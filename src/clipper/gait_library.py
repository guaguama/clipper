"""Convert an mj-nlp gait library (periodic strides) into a looping mjlab library.

Backs ``scripts/convert_gait_library.py``. Each mj-nlp node is one periodic stride,
``state`` ``(n+1, nq+nv)``. The closing state equals frame 0 up to the stride's planar
SE(2) displacement, to about 1e-16. The output is one mjlab motion NPZ per node, in
the schema of ``writers/mjlab.py``, plus a ``twist`` label. It is meant for a
twist-command task that **loops** each clip (``t = (t+1) % n_frames``) and switches
between clips at the same phase, like sesteban's ``LibraryMotionCommand``.

The generic ``mjlab`` writer is right for one-shot clips but wrong for loops in three
ways, and this module fixes each one:

1. **Frame count.** ``frame_blend`` samples ``arange(0, T, 1/50)``, i.e. ceil(50T)
   frames. When 50T is not an integer (1.41 s -> 70.5, 1.25 s -> 62.5) the loop is a
   half frame too long. Here the stride is resampled to exactly N = round(50T) frames
   over [0, T). That time-warps it by kappa = T / (N/50) (<= 0.8 %), and every velocity
   is scaled by kappa so it stays consistent with the positions.
2. **Velocities.** Finite differences are one-sided at the clip ends. On the planted
   library that left joint_vel 1.28 rad/s off at frame 0, against 0.13 rad/s in the
   interior, and sesteban's own G1 clips show 1.4-3.2 rad/s. Here velocities are the
   solver's own qvel, which is exact and periodic, resampled and pushed through
   ``mj_objectVelocity`` for the bodies (world frame, at the body origin, the same
   point a finite difference of ``xpos`` measures, and mjlab's ``body_link_*_vel_w``).
   That needs ``mjOBJ_XBODY``: ``mjOBJ_BODY`` reports the velocity at the body's COM
   (``xipos``), which on osl_fullbody's massless root sits 1 m from its origin. There is no seam by construction.
3. **Phase.** Nodes solved from different source clips start at unrelated points of the
   gait cycle. On ``clip_search/gait_library_walk_fwd_multi`` the four bands sit at
   -35 / -8 / 0 / +37 % of a stride. A phase-preserving clip switch would then jump a
   third of a stride. Clips are aligned on **normalised phase**, which is
   sesteban's ``library_to_npz._align_phase`` generalised to unequal stride lengths:
   - circular cross-correlation of the mean-removed foot-height profiles against a
     reference clip;
   - circular roll, with the wrapped block moved by the stride's SE(2) map so the
     roll creates no seam;
   - re-base so frame 0 has the root at xy = 0 and the **stride-mean heading** of the
     anatomical forward axis (``constants.FWD_AXIS_BODY``, -y on osl_fullbody) along +x.

A seam check then compares the runtime loop step (frame N-1 -> the SE(2)-mapped frame
0) with the clip's interior steps, for positions and velocities.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import functools

import mujoco
import numpy as np

from . import constants, mathx
from .foot_contact import _ANKLE_TABLES, _Foot
from .model import joint_names, load_model
from .qpos import joint_qpos_address
from .sources import mj_nlp_out
from .writers.mjlab import _fk_model, save_npz

@functools.lru_cache(maxsize=None)
def _models(model_id: str) -> tuple[mujoco.MjModel, mujoco.MjModel]:
    """(clip model, FK model), compiled once: an MSK compile takes about a second.

    Read-only here; nothing in this module mutates either model.
    """
    return load_model(model_id), _fk_model(model_id)


PHASE_BINS = 400  # normalised-phase resolution of the alignment (0.25 % of a stride)
SEAM_TOL = 1.5  # seam step must be <= SEAM_TOL x the largest interior step


# --------------------------------------------------------------------------- #
# SE(2) maps on the ground plane, as 3x3 homogeneous matrices on (x, y).
# --------------------------------------------------------------------------- #
def se2(yaw: float, t: np.ndarray) -> np.ndarray:
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, t[0]], [s, c, t[1]], [0.0, 0.0, 1.0]])


def se2_yaw(m: np.ndarray) -> float:
    return float(np.arctan2(m[1, 0], m[0, 0]))


@dataclass
class Clip:
    """One looping clip in mjlab arrays, plus its stride map.

    ``stride`` maps the world pose at frame k to the pose at frame k + N (the next
    loop), as an SE(2) matrix. Bodies follow ``_fk_model``'s order (body 0 = root).
    """

    name: str
    joint_pos: np.ndarray  # (N, J)
    joint_vel: np.ndarray  # (N, J)
    body_pos_w: np.ndarray  # (N, B, 3)
    body_quat_w: np.ndarray  # (N, B, 4)
    body_lin_vel_w: np.ndarray  # (N, B, 3)
    body_ang_vel_w: np.ndarray  # (N, B, 3)
    stride: np.ndarray  # (3, 3)
    twist: np.ndarray  # (3,) commanded label
    kappa: float = 1.0  # velocity scale from the exact-period time warp
    stride_T: float = 0.0  # source stride period (s)
    source: str = ""
    phase_shift: int = 0  # frames rolled by the alignment
    extra: dict[str, np.ndarray] = field(default_factory=dict)

    @property
    def n(self) -> int:
        return int(self.joint_pos.shape[0])

    def transformed(self, m: np.ndarray, rows: slice | None = None) -> None:
        """Apply the planar map `m` in place to every body pose/velocity (or `rows`)."""
        r = slice(None) if rows is None else rows
        yaw = se2_yaw(m)
        rz = mathx.rot_z(yaw)
        pos = self.body_pos_w[r]
        xy = pos[..., :2] @ m[:2, :2].T + m[:2, 2]
        self.body_pos_w[r] = np.concatenate([xy, pos[..., 2:]], axis=-1)
        self.body_quat_w[r] = mathx.quat_mul(mathx.yaw_quat(yaw), self.body_quat_w[r])
        self.body_lin_vel_w[r] = self.body_lin_vel_w[r] @ rz.T
        self.body_ang_vel_w[r] = self.body_ang_vel_w[r] @ rz.T

    def apply_world(self, a: np.ndarray) -> None:
        """Move the whole clip by the planar map `a`; the stride map is conjugated."""
        self.transformed(a)
        self.stride = a @ self.stride @ np.linalg.inv(a)


# --------------------------------------------------------------------------- #
# Loading + exact-period resample + solver velocities
# --------------------------------------------------------------------------- #
def _joint_columns(model: mujoco.MjModel, model_id: str) -> tuple[np.ndarray, np.ndarray]:
    """qpos and qvel addresses of the model's joints, in `joint_names` order."""
    names = joint_names(model_id)
    qadr, vadr = [], []
    for jn in names:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jn)
        qadr.append(joint_qpos_address(model, jn))
        vadr.append(int(model.jnt_dofadr[jid]))
    return np.asarray(qadr), np.asarray(vadr)


def _heading_track(quat: np.ndarray, fwd: tuple[float, float, float]) -> np.ndarray:
    return np.unwrap(mathx.heading_from_quat(quat, fwd))


def load_node(path: Path, model_id: str, out_fps: float) -> Clip:
    """Load one mj-nlp node and return it as an exact-period mjlab `Clip`."""
    model = _models(model_id)[0]
    traj = mj_nlp_out.load(path, model, model_id)
    raw = np.load(path)
    state = np.asarray(raw["state"], dtype=np.float64)
    if state.shape[1] != model.nq + model.nv:
        raise ValueError(
            f"{path.name}: state width {state.shape[1]} != nq+nv "
            f"{model.nq + model.nv} of {model_id!r}."
        )
    # The source maps qpos by joint name; qvel is read positionally, which is only
    # valid when that mapping was the identity (same model, same joint order).
    if not np.array_equal(traj.qpos, state[:, : model.nq]):
        raise ValueError(
            f"{path.name}: the solve's qpos layout differs from {model_id!r}'s, so "
            "its qvel cannot be read positionally."
        )
    if traj.twist is None:
        raise ValueError(f"{path.name}: no `twist` label; not a gait-library node.")
    qpos_src = traj.qpos
    qvel_src = state[:, model.nq :]
    dt_src = 1.0 / traj.fps
    n_src = qpos_src.shape[0]
    T = (n_src - 1) * dt_src
    fwd = constants.FWD_AXIS_BODY[model_id]

    # The stride map, from frame 0 to the closing state (= frame 0 of the next loop).
    psi0 = float(mathx.heading_from_quat(qpos_src[0, 3:7], fwd))
    psi1 = float(mathx.heading_from_quat(qpos_src[-1, 3:7], fwd))
    p0, p1 = qpos_src[0, :2], qpos_src[-1, :2]
    rot = se2(psi1 - psi0, np.zeros(2))
    stride = se2(psi1 - psi0, p1 - rot[:2, :2] @ p0)

    # Exact-period resample: N frames on [0, T), closing state excluded.
    n_out = int(np.floor(T * out_fps + 0.5))
    if n_out < 3:
        raise ValueError(f"{path.name}: stride {T:.3f} s gives only {n_out} frames.")
    x = np.arange(n_out) * (T / n_out) / dt_src
    i0 = np.floor(x).astype(np.intp)
    i1 = i0 + 1  # x < n_src - 1 always, so i1 <= n_src - 1
    b = x - i0
    kappa = T / (n_out / out_fps)
    qpos = mathx.lerp(qpos_src[i0], qpos_src[i1], b[:, None])
    qpos[:, 3:7] = mathx.slerp(qpos_src[i0, 3:7], qpos_src[i1, 3:7], b)
    qvel = mathx.lerp(qvel_src[i0], qvel_src[i1], b[:, None]) * kappa

    clip = _kinematics(model_id, qpos, qvel, path.stem, stride)
    clip.twist = np.asarray(traj.twist, dtype=np.float64)
    clip.kappa = kappa
    clip.stride_T = T
    clip.source = str(path)
    return clip


def _kinematics(
    model_id: str,
    qpos: np.ndarray,
    qvel: np.ndarray,
    name: str,
    stride: np.ndarray,
) -> Clip:
    """Body poses and velocities from (qpos, qvel) on the mjlab body set."""
    model, fk = _models(model_id)
    if fk.nq != model.nq or fk.nv != model.nv:
        raise ValueError(f"{model_id}: FK model layout differs from the clip model's.")
    qadr, vadr = _joint_columns(model, model_id)
    data = mujoco.MjData(fk)
    n, nb = qpos.shape[0], fk.nbody - 1
    pos = np.empty((n, nb, 3))
    quat = np.empty((n, nb, 4))
    lin = np.empty((n, nb, 3))
    ang = np.empty((n, nb, 3))
    res = np.zeros(6)
    for k in range(n):
        data.qpos[:] = qpos[k]
        data.qvel[:] = qvel[k]
        mujoco.mj_kinematics(fk, data)
        mujoco.mj_comPos(fk, data)
        mujoco.mj_comVel(fk, data)
        pos[k] = data.xpos[1:]
        quat[k] = data.xquat[1:]
        for bid in range(1, fk.nbody):
            # world-oriented 6D velocity at the body origin: [ang(3), lin(3)].
            # XBODY, not BODY: BODY is evaluated at the COM (xipos).
            mujoco.mj_objectVelocity(fk, data, mujoco.mjtObj.mjOBJ_XBODY, bid, res, 0)
            ang[k, bid - 1] = res[:3]
            lin[k, bid - 1] = res[3:]
    return Clip(
        name=name,
        joint_pos=qpos[:, qadr],
        joint_vel=qvel[:, vadr],
        body_pos_w=pos,
        body_quat_w=quat,
        body_lin_vel_w=lin,
        body_ang_vel_w=ang,
        stride=stride,
        twist=np.zeros(3),
    )


# --------------------------------------------------------------------------- #
# Phase alignment + re-base
# --------------------------------------------------------------------------- #
def foot_body_indices(model_id: str) -> list[int]:
    """Body-array indices (body id - 1) of one body per foot, for phase profiles."""
    fk = _models(model_id)[1]
    out = []
    for bodies in constants.FOOT_CONTACT_BODIES[model_id].values():
        out.append(mujoco.mj_name2id(fk, mujoco.mjtObj.mjOBJ_BODY, bodies[0]) - 1)
    return out


def phase_profile(body_pos_w: np.ndarray, feet: list[int], bins: int = PHASE_BINS) -> np.ndarray:
    """Mean-removed foot heights on `bins` normalised-phase samples, shape (F, bins).

    Heights are invariant under the planar stride map, so frame N is frame 0 and the
    periodic interpolation is exact at the wrap.
    """
    z = body_pos_w[:, feet, 2].T  # (F, N)
    n = z.shape[1]
    x = np.arange(bins) / bins * n
    i0 = np.floor(x).astype(np.intp)
    b = x - i0
    p = z[:, i0 % n] * (1 - b) + z[:, (i0 + 1) % n] * b
    return p - p.mean(axis=1, keepdims=True)


def phase_lag(profile: np.ndarray, ref: np.ndarray) -> float:
    """Phase (fraction of a stride, in (-0.5, 0.5]) to roll `profile` onto `ref`."""
    bins = ref.shape[1]
    f = np.fft.rfft(profile, axis=1)
    g = np.fft.rfft(ref, axis=1)
    # cc[s] = sum_t roll(profile, s)[t] * ref[t]
    cc = np.fft.irfft(np.conj(f) * g, n=bins, axis=1).sum(axis=0)
    s = int(np.argmax(cc))
    if s > bins // 2:
        s -= bins
    return s / bins


def roll_clip(clip: Clip, s: int) -> None:
    """Circularly roll `clip` by `s` frames (frame k -> k + s), seam-free.

    Wrapped frames come from the neighbouring stride, so they are moved by the
    stride map (s > 0: the previous stride, inverse map; s < 0: the next one).
    """
    n = clip.n
    s = int(s) % n
    if s == 0:
        return
    s = s if s <= n // 2 else s - n
    for key in (
        "joint_pos", "joint_vel", "body_pos_w", "body_quat_w",
        "body_lin_vel_w", "body_ang_vel_w",
    ):
        setattr(clip, key, np.roll(getattr(clip, key), s, axis=0))
    if s > 0:
        clip.transformed(np.linalg.inv(clip.stride), slice(0, s))
    else:
        clip.transformed(clip.stride, slice(n + s, n))
    clip.phase_shift = s


def rebase(clip: Clip, model_id: str) -> None:
    """Root at xy = 0 at frame 0, stride-mean heading along +x."""
    fwd = constants.FWD_AXIS_BODY[model_id]
    psi = _heading_track(clip.body_quat_w[:, 0], fwd)
    psi_mean = mathx.mean_heading(psi, psi[0] + se2_yaw(clip.stride))
    a = se2(-psi_mean, np.zeros(2))
    a = se2(-psi_mean, -(a[:2, :2] @ clip.body_pos_w[0, 0, :2]))
    clip.apply_world(a)


# --------------------------------------------------------------------------- #
# Seam check
# --------------------------------------------------------------------------- #
def seam_report(clip: Clip) -> dict[str, tuple[float, float]]:
    """(seam step, largest interior step) per quantity, for the runtime wrap.

    The wrap goes from frame N-1 to frame 0 of the *next* loop, i.e. frame 0 moved
    by the stride map (joints are unmoved).
    """
    nxt = Clip(
        name="", joint_pos=clip.joint_pos[:1].copy(), joint_vel=clip.joint_vel[:1].copy(),
        body_pos_w=clip.body_pos_w[:1].copy(), body_quat_w=clip.body_quat_w[:1].copy(),
        body_lin_vel_w=clip.body_lin_vel_w[:1].copy(),
        body_ang_vel_w=clip.body_ang_vel_w[:1].copy(), stride=clip.stride, twist=clip.twist,
    )
    nxt.transformed(clip.stride)
    out = {}
    for key, arr, first in (
        ("joint_pos", clip.joint_pos, nxt.joint_pos),
        ("joint_vel", clip.joint_vel, nxt.joint_vel),
        ("root_pos", clip.body_pos_w[:, 0], nxt.body_pos_w[:, 0]),
        ("body_lin_vel", clip.body_lin_vel_w, nxt.body_lin_vel_w),
        ("body_ang_vel", clip.body_ang_vel_w, nxt.body_ang_vel_w),
    ):
        inner = np.abs(np.diff(arr, axis=0)).reshape(arr.shape[0] - 1, -1).max(axis=1)
        seam = float(np.abs(first[0] - arr[-1]).max())
        out[key] = (seam, float(inner.max()))
    return out


# --------------------------------------------------------------------------- #
# Idle clip
# --------------------------------------------------------------------------- #
def idle_clip(model_id: str, n_frames: int, keyframe: str = "stand") -> tuple[Clip, dict]:
    """The `keyframe` pose held still, feet lowered onto z = 0, facing +x at the origin."""
    model = _models(model_id)[0]
    kid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, keyframe)
    if kid < 0:
        raise ValueError(f"{model_id!r} has no keyframe {keyframe!r}.")
    qpos = np.array(model.key_qpos[kid], dtype=np.float64)
    data = mujoco.MjData(model)
    data.qpos[:] = qpos
    mujoco.mj_kinematics(model, data)
    lows = {}
    for tag, bodies in constants.FOOT_CONTACT_BODIES[model_id].items():
        foot = _Foot(model, data, bodies, _ANKLE_TABLES[tag][model_id])
        lows[tag] = float(foot.world(data)[:, 2].min())
    dz = -min(lows.values())
    qpos[2] += dz
    gaps = {tag: z - min(lows.values()) for tag, z in lows.items()}
    qs = np.repeat(qpos[None], n_frames, axis=0)
    clip = _kinematics(model_id, qs, np.zeros((n_frames, model.nv)), "idle", np.eye(3))
    rebase(clip, model_id)
    return clip, {"dz": dz, "foot_gap": gaps}


# --------------------------------------------------------------------------- #
# Library
# --------------------------------------------------------------------------- #
def reference_index(clips: list[Clip]) -> int:
    """sesteban's reference choice: straight, forward, nearest mid forward speed."""
    fwd = [i for i, c in enumerate(clips) if c.twist[0] > 1e-6]
    if not fwd:
        return 0
    vx = [clips[i].twist[0] for i in fwd]
    mid = 0.5 * (min(vx) + max(vx))
    return min(
        fwd,
        key=lambda i: (
            abs(clips[i].twist[2]) > 1e-6 or abs(clips[i].twist[1]) > 1e-6,
            abs(clips[i].twist[0] - mid),
            clips[i].name,
        ),
    )


def align_library(
    clips: list[Clip], model_id: str, ref_profile: np.ndarray | None = None
) -> list[dict]:
    """Phase-align every clip to the reference profile (in place); return a report."""
    feet = foot_body_indices(model_id)
    if ref_profile is None:
        ref_profile = phase_profile(clips[reference_index(clips)].body_pos_w, feet)
    report = []
    for c in clips:
        lag = phase_lag(phase_profile(c.body_pos_w, feet), ref_profile)
        roll_clip(c, int(round(lag * c.n)))
        resid = phase_lag(phase_profile(c.body_pos_w, feet), ref_profile)
        report.append({"name": c.name, "lag": lag, "shift": c.phase_shift, "residual": resid})
    return report


def profile_from_npz(path: Path, model_id: str) -> np.ndarray:
    """Phase profile of an already-converted mjlab clip (an `--align-to` reference)."""
    data = np.load(path)
    return phase_profile(np.asarray(data["body_pos_w"], np.float64), foot_body_indices(model_id))


def write_clip(clip: Clip, out_dir: Path, fps: float) -> Path:
    extra = {
        "twist": clip.twist.astype(np.float32),
        "twist_effective": (clip.twist * clip.kappa).astype(np.float32),
        "stride_T": np.array(clip.stride_T),
        "n_frames": np.array(clip.n),
        "phase_shift": np.array(clip.phase_shift),
        # Planar map frame k -> frame k + N (the closing state the N frames omit), so a
        # viewer can chain loops exactly. Ignored by mjlab's loaders.
        "stride_se2": np.asarray(clip.stride, dtype=np.float64),
        "source": np.array(clip.source),
    }
    path = out_dir / f"{clip.name}.npz"
    save_npz(
        path, fps,
        joint_pos=clip.joint_pos, joint_vel=clip.joint_vel,
        body_pos_w=clip.body_pos_w, body_quat_w=clip.body_quat_w,
        body_lin_vel_w=clip.body_lin_vel_w, body_ang_vel_w=clip.body_ang_vel_w,
        extra=extra,
    )
    return path
