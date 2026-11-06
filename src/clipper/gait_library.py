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
   third of a stride. By default (``phase="touchdown"``, the gait library scheme) frame
   0 is the **touchdown of the OSL foot** (``constants.TOUCHDOWN_FOOT``):
   - the start of the foot's longest cyclic run with its lowest sole point below
     ``TOUCHDOWN_TOL`` = 5 mm, which is mj-nlp's own stance rule
     (``osl_gait.cyclic_longest_run`` on ``slip_contact_tol``);
   - found on the solver's own samples (200 Hz) and refined to the threshold crossing,
     so the resample simply starts there; samples past the period come from the next
     stride, moved by the stride's SE(2) map, so the shift creates no seam;
   - it is an absolute event, so every node and every release lands on the same phase 0
     with no reference clip.
   The old ``phase="xcorr"`` (circular cross-correlation of foot heights against a
   reference clip, sesteban's ``library_to_npz._align_phase`` generalised to unequal
   strides) is kept for comparison. Either way the clip is then re-based so frame 0 has
   the root at xy = 0 and the **stride-mean heading** of the anatomical forward axis
   (``constants.FWD_AXIS_BODY``, -y on osl_fullbody) along +x.

Before any of that, each node's **loop closure** is checked (closing state = frame 0
moved by the stride map); a seam check then compares the runtime loop step (frame N-1
-> the SE(2)-mapped frame 0) with the clip's interior steps.

**Release folders** (the gait library scheme): one flat folder ``<name>_vNNN/`` with a
``library.json`` manifest, ``idle.npz`` and one ``<family>_vx±NNNNN_vy±NNNNN_wz±NNNNN.npz``
per twist (thousandths). `read_release` checks the folder against its manifest and the
names against the twist labels; the converted folder keeps every file name.
"""

from __future__ import annotations

import datetime
import functools
import json
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

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
# A foot is in contact while its lowest sole point is below this (m): mj-nlp's
# `slip_contact_tol`, which its own stance windows and QA use. A run below it counts as a
# contact only if it reaches the floor (z <= 0); one that doesn't is a swing skim.
TOUCHDOWN_TOL = 0.005
# Loop closure: closing state vs frame 0 moved by the stride map. mj-nlp nodes close to
# ~1e-15 (periodicity is a hard constraint), so anything near this is a broken node.
CLOSURE_TOL_POS = 1e-6  # m / rad
CLOSURE_TOL_VEL = 1e-4  # m/s / rad/s
PHASES = ("touchdown", "xcorr", "none")
# Where joint_vel / root qvel come from (body velocities always follow from FK of it):
#   solver  - mj-nlp's own qvel, resampled and scaled by kappa (clipper's original choice);
#   central - 0.5 (d(q[k-1] -> q[k]) + d(q[k] -> q[k+1])) / dt at the output rate, which is
#             what musclemimic trains on: its TrajectoryHandler always overwrites a clip's
#             qvel with `recompute_trajectory_velocities` (handler.py:476);
#   forward - d(q[k] -> q[k+1]) / dt, the qvel clipper's musclemimic writer stores (and
#             musclemimic then discards).
# d() is mj_differentiatePos; the loop wraps through the stride map, so no end is one-sided.
VELOCITIES = ("solver", "central", "forward")

MANIFEST = "library.json"
IDLE_STEM = "idle"
NODE_RE = re.compile(
    r"^(?P<family>[a-z][a-z0-9]*(?:_[a-z0-9]+)*?)"
    r"_vx(?P<vx>[+-]\d{5})_vy(?P<vy>[+-]\d{5})_wz(?P<wz>[+-]\d{5})$"
)
RELEASE_RE = re.compile(r"_v\d{3}$")


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
    # Where frame 0 sits in the source stride, as a fraction of it in [0, 1) (the
    # touchdown time / T, or the xcorr roll). Written as the `phase_shift` extra.
    phase_shift: float = 0.0
    velocity: str = "solver"  # qvel source (VELOCITIES); "zero" for the idle
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


def _check_free_root(model: mujoco.MjModel, model_id: str) -> None:
    if model.njnt == 0 or model.jnt_type[0] != mujoco.mjtJoint.mjJNT_FREE or model.jnt_qposadr[0]:
        raise ValueError(f"{model_id}: the root must be a free joint at qpos[0:7].")


def map_state(m: np.ndarray, qpos: np.ndarray, qvel: np.ndarray | None = None):
    """Move (qpos, qvel) rows by the planar map `m` (root xy, root yaw, root linvel).

    A free joint's qvel is [world linear | body angular], so only its linear part turns.
    """
    yaw = se2_yaw(m)
    q = np.array(qpos, dtype=np.float64, copy=True)
    q[..., :2] = q[..., :2] @ m[:2, :2].T + m[:2, 2]
    q[..., 3:7] = mathx.quat_mul(mathx.yaw_quat(yaw), q[..., 3:7])
    if qvel is None:
        return q
    v = np.array(qvel, dtype=np.float64, copy=True)
    v[..., :2] = v[..., :2] @ m[:2, :2].T
    return q, v


def stride_map(qpos_first: np.ndarray, qpos_last: np.ndarray, model_id: str) -> np.ndarray:
    """SE(2) map taking the root's planar pose at `qpos_first` to that at `qpos_last`."""
    fwd = constants.FWD_AXIS_BODY[model_id]
    psi0 = float(mathx.heading_from_quat(qpos_first[3:7], fwd))
    psi1 = float(mathx.heading_from_quat(qpos_last[3:7], fwd))
    rot = se2(psi1 - psi0, np.zeros(2))
    return se2(psi1 - psi0, qpos_last[:2] - rot[:2, :2] @ qpos_first[:2])


def closure_error(qpos: np.ndarray, qvel: np.ndarray, stride: np.ndarray) -> tuple[float, float]:
    """(position, velocity) max error of the closing state vs frame 0 moved by `stride`.

    Root xy and heading match by construction of `stride`; this measures root z and
    tilt, the joints, and every velocity.
    """
    q_pred, v_pred = map_state(stride, qpos[0], qvel[0])
    qa, qb = qpos[-1, 3:7], q_pred[3:7]
    quat = min(np.linalg.norm(qa - qb), np.linalg.norm(qa + qb))
    pos = max(abs(qpos[-1, 2] - q_pred[2]), quat, float(np.abs(qpos[-1, 7:] - q_pred[7:]).max(initial=0.0)))
    vel = float(np.abs(qvel[-1] - v_pred).max())
    return float(pos), vel


@dataclass
class Node:
    """One mj-nlp node as solved: the stride's samples including the closing state."""

    path: Path
    qpos: np.ndarray  # (n+1, nq)
    qvel: np.ndarray  # (n+1, nv)
    dt: float
    twist: np.ndarray
    stride: np.ndarray  # (3, 3)

    @property
    def period(self) -> float:
        return (self.qpos.shape[0] - 1) * self.dt


def read_node(path: Path, model_id: str) -> Node:
    """Load an mj-nlp node file and check it (layout, twist label, loop closure).

    Raises ValueError naming the file when the loop does not close: a broken node must
    stop the conversion, never be dropped or patched.
    """
    model = _models(model_id)[0]
    _check_free_root(model, model_id)
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
    qpos, qvel = traj.qpos, state[:, model.nq :]
    stride = stride_map(qpos[0], qpos[-1], model_id)
    pos_err, vel_err = closure_error(qpos, qvel, stride)
    if pos_err > CLOSURE_TOL_POS or vel_err > CLOSURE_TOL_VEL:
        raise ValueError(
            f"{path.name}: the loop does not close (closing state vs frame 0 under the "
            f"stride map: pos {pos_err:.2e} > {CLOSURE_TOL_POS:g} or vel {vel_err:.2e} > "
            f"{CLOSURE_TOL_VEL:g}). Fix or drop it in mj-nlp; clipper never drops nodes."
        )
    return Node(
        path=Path(path), qpos=qpos, qvel=qvel, dt=1.0 / traj.fps,
        twist=np.asarray(traj.twist, dtype=np.float64), stride=stride,
    )


# --------------------------------------------------------------------------- #
# OSL touchdown (phase 0)
# --------------------------------------------------------------------------- #
@functools.lru_cache(maxsize=None)
def _touchdown_foot(model_id: str):
    """(model, data, sampled foot surface) of the phase-0 foot, labels from `stand`."""
    if model_id not in constants.TOUCHDOWN_FOOT:
        raise ValueError(f"{model_id!r} has no constants.TOUCHDOWN_FOOT entry.")
    tag = constants.TOUCHDOWN_FOOT[model_id]
    model = _models(model_id)[0]
    data = mujoco.MjData(model)
    kid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "stand")
    if kid >= 0:
        data.qpos[:] = model.key_qpos[kid]
    mujoco.mj_kinematics(model, data)
    bodies = constants.FOOT_CONTACT_BODIES[model_id][tag]
    return model, data, _Foot(model, data, bodies, _ANKLE_TABLES[tag][model_id])


def sole_height(qpos: np.ndarray, model_id: str) -> np.ndarray:
    """Lowest point of the phase-0 foot's collision geoms per qpos row (m, floor z=0)."""
    model, data, foot = _touchdown_foot(model_id)
    h = np.empty(len(qpos))
    for k, q in enumerate(qpos):
        data.qpos[:] = q
        mujoco.mj_kinematics(model, data)
        h[k] = foot.world(data)[:, 2].min()
    return h


@dataclass
class Touchdown:
    phase: float  # touchdown time / period, in [0, 1)
    stance: float  # fraction of the stride from touchdown to lift-off (lifts included)
    lifts: list[float]  # peak heights (m) of lifts off the floor during stance
    skims: int  # swing dips below `tol` that never reach the floor (ignored)


def _cyclic_runs(on: np.ndarray) -> list[tuple[int, int]]:
    """(start, length) of every maximal run of True in `on`, read as a loop (not all True)."""
    n = on.size
    k0 = int(np.flatnonzero(~on)[0])
    r = np.roll(on, -k0)
    runs, i = [], 0
    while i < n:
        if r[i]:
            j = i
            while j < n and r[j]:
                j += 1
            runs.append(((i + k0) % n, j - i))
            i = j
        else:
            i += 1
    return runs


def touchdown_phase(height: np.ndarray, tol: float = TOUCHDOWN_TOL) -> Touchdown:
    """OSL touchdown from one stride of sole heights (n periodic samples, no closing state).

    A contact is a run of samples with the sole below `tol` (mj-nlp's stance rule) that
    reaches the floor. A run that stays above the floor is a swing skim: the foot passes
    within `tol` at the end of swing, rises again, then plants. It is part of the swing.
    The swing is the longest gap between contacts. The touchdown is its end, i.e. the
    start of the real stance, refined to the sub-sample time the height crosses `tol`.
    Any other gap is a lift during stance (the foot leaves the floor by more than `tol`
    and comes back down). Lifts are reported, and frame 0 stays at the first contact.
    Detect on the solver's own samples: at a coarser rate a short lift can vanish.
    """
    n = height.size
    on = height < tol
    if not on.any():
        raise ValueError(f"the foot never comes within {1000 * tol:.0f} mm of the floor")
    if on.all():
        raise ValueError(f"the foot never lifts above {1000 * tol:.0f} mm")
    runs = _cyclic_runs(on)
    contacts = [(s, ln) for s, ln in runs if height[(s + np.arange(ln)) % n].min() <= 0.0]
    if not contacts:
        raise ValueError("the foot never reaches the floor")
    # gap after each contact: (length, first gap sample, start of the next contact)
    gaps = [
        ((b[0] - (a[0] + a[1])) % n, (a[0] + a[1]) % n, b[0])
        for a, b in zip(contacts, contacts[1:] + contacts[:1])
    ]
    swing = max(range(len(gaps)), key=lambda k: gaps[k][0])
    start = gaps[swing][2]
    h0, h1 = height[(start - 1) % n], height[start]
    frac = (h0 - tol) / (h0 - h1) if h0 > h1 else 1.0
    phase = float(((start - 1 + frac) / n) % 1.0)
    lifts = [
        float(height[(e + np.arange(g)) % n].max())
        for k, (g, e, _) in enumerate(gaps) if k != swing
    ]
    return Touchdown(
        phase=phase, stance=1.0 - gaps[swing][0] / n, lifts=lifts,
        skims=len(runs) - len(contacts),
    )


def loop_velocities(
    qpos: np.ndarray, stride: np.ndarray, model_id: str, fps: float, scheme: str
) -> np.ndarray:
    """Finite-difference qvel of a looping clip played at `fps` ("central" / "forward").

    The frame after N-1 is frame 0 moved by `stride`, the one before 0 is frame N-1
    moved back, so the loop has no one-sided end. Uses mj_differentiatePos, as musclemimic
    does (free joint: world linear, body angular velocity).
    """
    model = _models(model_id)[0]
    n, dt = qpos.shape[0], 1.0 / fps
    nxt = map_state(stride, qpos[:1])[0]
    fwd = np.empty((n, model.nv))  # fwd[k] = d(q[k] -> q[k+1]) / dt
    for k in range(n):
        mujoco.mj_differentiatePos(model, fwd[k], dt, qpos[k], qpos[k + 1] if k + 1 < n else nxt)
    if scheme == "forward":
        return fwd
    if scheme == "central":
        # fwd[k-1] = d(q[k-1] -> q[k]); for k = 0 that is d(stride^-1 q[N-1] -> q[0]), which
        # equals fwd[N-1] up to rotating the free joint's world-frame linear part back.
        prev = np.roll(fwd, 1, axis=0)
        prev[0] = map_state(np.linalg.inv(stride), np.zeros(model.nq), fwd[-1])[1]
        return 0.5 * (prev + fwd)
    raise ValueError(f"no finite-difference scheme {scheme!r}")


def resample_node(
    node: Node, model_id: str, out_fps: float, t0: float = 0.0, velocity: str = "solver"
) -> Clip:
    """Resample `node` to an exact-period `Clip` of N frames starting at time `t0`.

    N = round(T * out_fps) frames on [t0, t0 + T). Samples past the period come from the
    next stride (the node's samples moved by the stride map), so any `t0` gives a
    seam-free loop. `velocity` picks the qvel (see VELOCITIES): the solver's, scaled by
    kappa = T / (N / out_fps), or finite differences of the resampled frames at out_fps
    (which already play at that rate, so no kappa).
    """
    if velocity not in VELOCITIES:
        raise ValueError(f"velocity must be one of {VELOCITIES}, not {velocity!r}")
    T, dt = node.period, node.dt
    n_src = node.qpos.shape[0] - 1  # samples per stride
    n_out = int(np.floor(T * out_fps + 0.5))
    if n_out < 3:
        raise ValueError(f"{node.path.name}: stride {T:.3f} s gives only {n_out} frames.")
    t0 = float(t0) % T
    q_next, v_next = map_state(node.stride, node.qpos, node.qvel)
    qpos_src = np.concatenate([node.qpos[:-1], q_next])  # [0, 2T], 2 n_src + 1 rows
    qvel_src = np.concatenate([node.qvel[:-1], v_next])
    x = (t0 + np.arange(n_out) * (T / n_out)) / dt
    i0 = np.minimum(np.floor(x).astype(np.intp), 2 * n_src - 1)
    i1 = i0 + 1
    b = x - i0
    kappa = T / (n_out / out_fps)
    qpos = mathx.lerp(qpos_src[i0], qpos_src[i1], b[:, None])
    qpos[:, 3:7] = mathx.slerp(qpos_src[i0, 3:7], qpos_src[i1, 3:7], b)
    if velocity == "solver":
        qvel = mathx.lerp(qvel_src[i0], qvel_src[i1], b[:, None]) * kappa
    else:
        qvel = loop_velocities(qpos, node.stride, model_id, out_fps, velocity)

    clip = _kinematics(model_id, qpos, qvel, node.path.stem, node.stride)
    clip.velocity = velocity
    clip.twist = node.twist.copy()
    clip.kappa = kappa
    clip.stride_T = T
    clip.source = str(node.path)
    clip.phase_shift = t0 / T
    return clip


def load_node(path: Path, model_id: str, out_fps: float, phase: str = "touchdown") -> Clip:
    """Load, check and resample one mj-nlp node into an exact-period mjlab `Clip`.

    phase="touchdown" starts the clip at the OSL touchdown (`touchdown_phase`, on the
    solver's samples); "xcorr" / "none" start it at the node's own frame 0 (the xcorr
    roll happens later, in `align_library`).
    """
    if phase not in PHASES:
        raise ValueError(f"phase must be one of {PHASES}, not {phase!r}")
    node = read_node(path, model_id)
    t0 = 0.0
    if phase == "touchdown":
        try:
            ph = touchdown_phase(sole_height(node.qpos[:-1], model_id)).phase
        except ValueError as e:
            raise ValueError(f"{Path(path).name}: no OSL touchdown: {e}") from None
        t0 = ph * node.period
    return resample_node(node, model_id, out_fps, t0)


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


def roll_clip(clip: Clip, s: int) -> int:
    """Circularly roll `clip` by `s` frames (frame k -> k + s), seam-free.

    Wrapped frames come from the neighbouring stride, so they are moved by the
    stride map (s > 0: the previous stride, inverse map; s < 0: the next one).
    """
    n = clip.n
    s = int(s) % n
    if s == 0:
        return 0
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
    clip.phase_shift = float((clip.phase_shift - s / n) % 1.0)
    return s


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
    clip.velocity = "zero"
    return clip, {"dz": dz, "foot_gap": gaps}


def idle_from_node(path: Path, model_id: str, n_frames: int) -> tuple[Clip, dict]:
    """The gait library scheme's idle: one frame of a solved standing node, held still.

    mj-nlp's idle node is a static stance solved as a short periodic "stride", so its
    samples still sway slightly. The frame with the smallest |qvel| is held for
    `n_frames` with zero velocities, re-based to the origin facing +x.
    """
    node = read_node(path, model_id)
    if np.any(np.abs(node.twist) > 1e-9):
        raise ValueError(f"{Path(path).name}: idle node has twist {node.twist.tolist()}, not 0.")
    k = int(np.argmin(np.linalg.norm(node.qvel[:-1], axis=1)))
    nv = node.qvel.shape[1]
    qs = np.repeat(node.qpos[k][None], n_frames, axis=0)
    clip = _kinematics(model_id, qs, np.zeros((n_frames, nv)), IDLE_STEM, np.eye(3))
    rebase(clip, model_id)
    clip.source = str(path)
    clip.velocity = "zero"
    info = {
        "frame": k,
        "frame_qvel": float(np.linalg.norm(node.qvel[k])),
        "max_qvel": float(np.abs(node.qvel).max()),
        "sole_min": float(sole_height(node.qpos[k : k + 1], model_id)[0]),
    }
    return clip, info


# --------------------------------------------------------------------------- #
# Release folders (gait library scheme)
# --------------------------------------------------------------------------- #
def node_tag(twist) -> str:
    """``vx+01000_vy+00000_wz-00250``: the twist in thousandths (mm/s, mrad/s)."""
    v = [int(round(1000.0 * float(x))) for x in np.asarray(twist).reshape(3)]
    return "_".join(f"{a}{x:+06d}" for a, x in zip(("vx", "vy", "wz"), v))


def check_node_name(stem: str, twist) -> list[str]:
    """Problems with a node file name against its twist label (empty = fine).

    ``idle`` must carry twist 0. Others must match ``<family>_vx±NNNNN_vy±NNNNN_wz±NNNNN``
    with the label rounded to thousandths, and the known families their sign rules:
    walk_fwd vx > 0, walk_bck vx < 0, walk_turn vx = vy = 0. New families (run_fwd, ...)
    are checked against the pattern only.
    """
    t = np.asarray(twist, dtype=np.float64).reshape(3)
    if stem == IDLE_STEM:
        return [] if np.all(np.abs(t) < 1e-9) else [f"{stem}: idle twist {t.tolist()} != 0"]
    m = NODE_RE.match(stem)
    if not m:
        return [f"{stem}: not <family>_vx±NNNNN_vy±NNNNN_wz±NNNNN"]
    probs = []
    named = np.array([int(m[a]) for a in ("vx", "vy", "wz")])
    label = np.round(1000.0 * t).astype(int)
    if not np.array_equal(named, label):
        probs.append(f"{stem}: name says {node_tag(named / 1000)}, twist label is {node_tag(t)}")
    fam = m["family"]
    if fam.endswith("_arc"):
        probs.append(f"{stem}: arcs belong in the _fwd/_bck family (wz != 0), not {fam}")
    rules = {
        "walk_fwd": (label[0] > 0, "vx > 0"),
        "walk_bck": (label[0] < 0, "vx < 0"),
        "walk_turn": (label[0] == 0 and label[1] == 0, "vx = vy = 0"),
    }
    if fam in rules and not rules[fam][0]:
        probs.append(f"{stem}: family {fam} needs {rules[fam][1]}")
    return probs


@dataclass
class ReleaseInput:
    """One input folder: its node files, idle node and manifest (None = legacy folder)."""

    root: Path
    name: str
    manifest: dict | None
    nodes: list[Path]
    idle: Path | None
    notes: list[str] = field(default_factory=list)


def read_release(root: Path, glob: str | None = None) -> ReleaseInput:
    """Read a gait-library folder, checking it against the scheme.

    With a ``library.json``: the node files are exactly the manifest's ``nodes`` (plus
    ``idle.npz``); a listed file that is missing or an unlisted ``*.npz`` is an error,
    since a release holds only accepted nodes. Without one (an mj-nlp working folder
    from before the scheme): `glob` (default ``*_vx*.npz``, ``seed_*`` skipped) plus
    ``idle.npz`` if present. File names are not checked here (see `check_node_name`,
    which needs the twist labels).
    """
    root = Path(root)
    if not root.is_dir():
        raise ValueError(f"{root}: not a folder")
    notes = []
    mpath = root / MANIFEST
    idle = root / f"{IDLE_STEM}.npz"
    idle = idle if idle.exists() else None
    if mpath.exists():
        manifest = json.loads(mpath.read_text())
        if glob is not None:
            notes.append(f"--glob ignored: {MANIFEST} lists the nodes")
        listed = [str(n["file"]) for n in manifest.get("nodes", [])]
        dup = sorted({f for f in listed if listed.count(f) > 1})
        on_disk = {f.name for f in root.glob("*.npz")}
        missing = sorted(set(listed) - on_disk)
        extra = sorted(on_disk - set(listed) - {f"{IDLE_STEM}.npz"})
        errs = []
        if dup:
            errs.append(f"listed twice: {dup}")
        if missing:
            errs.append(f"listed but missing: {missing}")
        if extra:
            errs.append(f"not in the manifest (a release holds only accepted nodes): {extra}")
        if errs:
            raise ValueError(f"{root}: folder and {MANIFEST} disagree; " + "; ".join(errs))
        nodes = [root / f for f in listed if f != f"{IDLE_STEM}.npz"]
        name = str(manifest.get("release") or root.name)
        if name != root.name:
            notes.append(f"{MANIFEST} names release {name!r}, folder is {root.name!r}")
        if idle is None:
            raise ValueError(f"{root}: a release must hold {IDLE_STEM}.npz")
    else:
        manifest = None
        g = glob or "*_vx*.npz"
        nodes = sorted(
            f for f in root.glob(g) if not f.name.startswith("seed_") and f.stem != IDLE_STEM
        )
        name = root.name
        notes.append(f"no {MANIFEST}: legacy mj-nlp folder, nodes = {g!r}")
    if not RELEASE_RE.search(name):
        notes.append(f"{name!r} is not a release name (<name>_vNNN)")
    return ReleaseInput(root=root, name=name, manifest=manifest, nodes=nodes, idle=idle, notes=notes)


@dataclass
class Converted:
    """A converted library: moving clips (input order), the idle, a per-node report."""

    clips: list[Clip]
    idle: Clip | None
    idle_info: dict
    rows: list[dict]
    warnings: list[str]


def convert_release(
    rel: ReleaseInput,
    model_id: str,
    fps: float,
    phase: str = "touchdown",
    align_to: Path | None = None,
    idle_frames: int = 50,
    idle_keyframe: str | None = None,
    velocity: str = "solver",
    idle_node: Path | None = None,
) -> Converted:
    """Run the scheme's per-node steps on a release folder, in memory.

    a. every node's loop closure (all failures are collected, then raised together);
    b./c. exact-period resample starting at the OSL touchdown (or `phase` xcorr/none);
    d. re-base + FK; the seam is measured into the report rows (the caller decides).
    File names are checked against the twist labels: an error in a release (manifest
    present), a warning in a legacy folder. The idle is ``idle.npz`` when there is one,
    else the model keyframe `idle_keyframe` when given, else none.
    """
    if phase not in PHASES:
        raise ValueError(f"phase must be one of {PHASES}, not {phase!r}")
    if align_to is not None and phase != "xcorr":
        raise ValueError("align_to only applies to phase='xcorr'")
    warnings = list(rel.notes)
    nodes, errors = [], []
    for f in rel.nodes:
        try:
            nodes.append(read_node(f, model_id))
        except ValueError as e:
            errors.append(str(e))
    if errors:
        raise ValueError(f"{len(errors)} node(s) failed the checks:\n  " + "\n  ".join(errors))
    probs = [p for n in nodes for p in check_node_name(n.path.stem, n.twist)]
    if probs and rel.manifest is not None:
        raise ValueError("file names disagree with the scheme:\n  " + "\n  ".join(probs))
    if len(probs) > 5:
        probs = probs[:5] + [f"... and {len(probs) - 5} more name problems"]
    warnings += probs

    clips, errors, lifts = [], [], {}
    for n in nodes:
        t0 = 0.0
        if phase == "touchdown":
            try:
                td = touchdown_phase(sole_height(n.qpos[:-1], model_id))
            except ValueError as e:
                errors.append(f"{n.path.name}: no OSL touchdown: {e}")
                continue
            t0 = td.phase * n.period
            if td.lifts:
                lifts[n.path.stem] = td.lifts
        clips.append(resample_node(n, model_id, fps, t0, velocity))
    if errors:
        raise ValueError("\n  ".join(["touchdown detection failed:", *errors]))
    if lifts:
        warnings.append(
            f"{len(lifts)} node(s) lift the OSL foot off the floor during stance (above "
            f"{1000 * TOUCHDOWN_TOL:.0f} mm, then back down); frame 0 is the first "
            "contact: " + ", ".join(
                f"{k} ({'/'.join(f'{1000 * h:.1f}' for h in v)} mm)"
                for k, v in list(lifts.items())[:6]
            ) + (" ..." if len(lifts) > 6 else "")
        )

    xc = {}
    if phase == "xcorr" and clips:
        ref = profile_from_npz(align_to, model_id) if align_to else None
        for r in align_library(clips, model_id, ref):
            xc[r["name"]] = r
    for c in clips:
        rebase(c, model_id)

    rows = []
    for c in clips:
        seam = seam_report(c)
        rows.append({
            "name": c.name,
            "n_frames": c.n,
            "period_s": c.stride_T,
            "kappa": c.kappa,
            "phase_shift": c.phase_shift,
            "xcorr_residual": xc.get(c.name, {}).get("residual"),
            "lifts_m": lifts.get(c.name, []),
            "seam": {k: s_ / max(i_, 1e-9) for k, (s_, i_) in seam.items()},
        })

    idle, idle_info = None, {}
    idle_path = rel.idle if rel.idle is not None else idle_node
    if rel.idle is not None and idle_node is not None:
        warnings.append(f"idle_node ignored: the folder has its own {rel.idle.name}")
    if idle_path is not None:
        idle, idle_info = idle_from_node(idle_path, model_id, idle_frames)
        idle_info = {"from": str(idle_path) if idle_path != rel.idle else idle_path.name, **idle_info}
    elif idle_keyframe:
        idle, info = idle_clip(model_id, idle_frames, idle_keyframe)
        idle_info = {"from": f"keyframe {idle_keyframe!r}", "dz": info["dz"],
                     "foot_gap": info["foot_gap"]}
    return Converted(clips=clips, idle=idle, idle_info=idle_info, rows=rows, warnings=warnings)


def clipper_git_info() -> dict:
    """{"git_sha", "dirty"} of this clipper checkout (None when git is unavailable)."""
    def git(*a: str) -> str | None:
        try:
            r = subprocess.run(
                ["git", "-C", str(constants.REPO_ROOT), *a],
                capture_output=True, text=True, timeout=10, check=True,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        return r.stdout.strip()

    sha = git("rev-parse", "--short", "HEAD")
    status = git("status", "--porcelain")
    return {"git_sha": sha, "dirty": None if status is None else bool(status)}


def write_manifest(
    out_dir: Path, release: ReleaseInput, clipper_block: dict
) -> Path:
    """Copy the input manifest (or start one) and add clipper's block."""
    doc = dict(release.manifest) if release.manifest is not None else {"release": release.name}
    doc["clipper"] = {
        **clipper_git_info(),
        "converted": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "input": str(release.root.resolve()),
        **clipper_block,
    }
    path = out_dir / MANIFEST
    path.write_text(json.dumps(doc, indent=1) + "\n")
    return path


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
        shift = roll_clip(c, int(round(lag * c.n)))
        resid = phase_lag(phase_profile(c.body_pos_w, feet), ref_profile)
        report.append({"name": c.name, "lag": lag, "shift": shift, "residual": resid})
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
        "phase_shift": np.array(float(clip.phase_shift)),  # frame 0 / source stride
        # Planar map frame k -> frame k + N (the closing state the N frames omit), so a
        # viewer can chain loops exactly. Ignored by mjlab's loaders.
        "stride_se2": np.asarray(clip.stride, dtype=np.float64),
        "source": np.array(clip.source),
        "velocity_scheme": np.array(clip.velocity),
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
