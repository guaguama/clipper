"""Twist-driven playback of a looping gait library (pure numpy + MuJoCo, no GUI).

Backs ``scripts/visualize_gait_library.py``. The player mirrors what sesteban's
``LibraryMotionCommand`` (``tasks/crawling_fwd/mdp/commands.py``) and its blending
subclass (``tasks/crawling_fwd_blending``) feed a policy, so the viewer shows the
reference the policy actually tracks:

* **Selection.** The commanded twist ``[vx, vy, wz]`` snaps to the nearest library node
  under a per-axis weighted squared distance. Weights are ``1 / span**2``, rescaled so
  the largest is 1, like sesteban's ``span_normalized_weights``; an axis with no span
  gets weight 0. There is no interpolation between nodes.
* **Phase.** Normalised phase ``phi`` in ``[0, 1)`` survives a switch: clip ``c`` sits at
  fractional frame ``phi * N_c``. Frames are interpolated (lerp, slerp for the root).
  Past frame ``N-1``, the next frame is frame 0 moved by the clip's stride map.
* **Continuous travel.** Clip poses live in each clip's own frame (re-based, frame 0
  at the origin). A planar SE(2) anchor ``A`` places them in the world. On each
  loop, ``A <- A @ stride``. On a switch the new clip is re-anchored so its root
  planar pose (xy + forward-axis heading) equals the one on screen, so there is no
  teleport. sesteban instead re-anchors to the *robot*; with no robot here,
  re-anchoring to the displayed reference is the equivalent.
* **Crossfade.** Over ``blend_s`` seconds the outgoing clip keeps playing at the
  shared phase on its own anchor. The two world poses are blended: root position and
  joints by lerp, root quaternion by slerp.

``LibraryClip`` holds a clip in the *model's* qpos layout, so it is source-agnostic.
``load_library`` builds clips from converted mjlab folders (``convert_gait_library.py``
output, which must carry ``stride_se2``) or from mj-nlp release folders (converted in
memory with ``clipper.gait_library.convert_release``, the converter's own pipeline). Every
clip's frame 0 is the OSL touchdown (the gait library scheme), so clips from any number
of folders share phase 0; a converted folder whose library.json says otherwise (or that
has none) is reported at load.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .. import constants, mathx
from .. import gait_library as gl
from ..model import joint_names
from ..qpos import build_qpos

AXES = ("vx", "vy", "wz")


# --------------------------------------------------------------------------- #
# Clips
# --------------------------------------------------------------------------- #
@dataclass
class LibraryClip:
    """One looping clip: qpos in the model layout, in its own (re-based) frame."""

    name: str
    twist: np.ndarray  # (3,) commanded label
    twist_effective: np.ndarray  # (3,) label x kappa (exact-period time warp)
    qpos: np.ndarray  # (N, nq)
    stride: np.ndarray  # (3, 3) planar map frame k -> frame k + N
    source: str = ""

    @property
    def n(self) -> int:
        return int(self.qpos.shape[0])

    @property
    def is_idle(self) -> bool:
        return bool(np.all(np.abs(self.twist) < 1e-6))


def apply_se2(m: np.ndarray, qpos: np.ndarray) -> np.ndarray:
    """Move a qpos row (or rows) by the planar map `m`: root xy and root yaw only."""
    out = np.array(qpos, dtype=np.float64, copy=True)
    out[..., :2] = out[..., :2] @ m[:2, :2].T + m[:2, 2]
    out[..., 3:7] = mathx.quat_mul(mathx.yaw_quat(gl.se2_yaw(m)), out[..., 3:7])
    return out


def planar_pose(qpos: np.ndarray, fwd: tuple[float, float, float]) -> np.ndarray:
    """SE(2) matrix of a qpos row's root: xy plus the heading of the forward axis."""
    psi = float(mathx.heading_from_quat(qpos[3:7], fwd))
    return gl.se2(psi, qpos[:2])


def blend_qpos(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    """(1-t)*a + t*b on a free-base + hinge qpos row (root quat by slerp)."""
    out = mathx.lerp(a, b, t)
    out[3:7] = mathx.slerp(a[3:7], b[3:7], t)
    return out


def sample(clip: LibraryClip, x: float) -> np.ndarray:
    """qpos at fractional frame `x` in [0, N), clip frame, interpolating across the wrap."""
    k = int(np.floor(x)) % clip.n
    b = float(x - np.floor(x))
    q0 = clip.qpos[k]
    q1 = clip.qpos[k + 1] if k + 1 < clip.n else apply_se2(clip.stride, clip.qpos[0])
    if b == 0.0:
        return q0.copy()
    return blend_qpos(q0, q1, b)


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def _qpos_from_mjlab_arrays(model, model_id, body_pos_w, body_quat_w, joint_pos):
    names = joint_names(model_id)
    if joint_pos.shape[1] != len(names):
        raise ValueError(
            f"joint_pos has {joint_pos.shape[1]} joints; {model_id!r} has {len(names)}."
        )
    return build_qpos(
        model,
        np.asarray(body_pos_w[:, 0], np.float64),
        np.asarray(body_quat_w[:, 0], np.float64),
        {jn: np.asarray(joint_pos[:, j], np.float64) for j, jn in enumerate(names)},
    )


def _load_mjlab_dir(d: Path, model, model_id: str) -> list[LibraryClip]:
    out = []
    for f in sorted(d.glob("*.npz")):
        data = np.load(f)
        if "twist" not in data:
            raise ValueError(f"{f}: no `twist` key; not a gait-library clip.")
        if "stride_se2" not in data:
            raise ValueError(
                f"{f}: no `stride_se2` key (needed to chain loops). Re-run "
                "scripts/convert_gait_library.py on its source library."
            )
        twist = np.asarray(data["twist"], np.float64).reshape(3)
        eff = data["twist_effective"] if "twist_effective" in data else twist
        clip = LibraryClip(
            name=f.stem,
            twist=twist,
            twist_effective=np.asarray(eff, np.float64).reshape(3),
            qpos=_qpos_from_mjlab_arrays(
                model, model_id, data["body_pos_w"], data["body_quat_w"], data["joint_pos"]
            ),
            stride=np.asarray(data["stride_se2"], np.float64),
            source=str(f),
        )
        out.append(clip)
    return out


def _from_gl_clip(c: gl.Clip, model, model_id: str) -> LibraryClip:
    return LibraryClip(
        name=c.name,
        twist=np.asarray(c.twist, np.float64),
        twist_effective=np.asarray(c.twist, np.float64) * c.kappa,
        qpos=_qpos_from_mjlab_arrays(model, model_id, c.body_pos_w, c.body_quat_w, c.joint_pos),
        stride=c.stride.copy(),
        source=c.source,
    )


def _dedupe(items: list, twist_of, name_of) -> list:
    """Keep the first item per twist (rounded to 1e-6); report the dropped ones."""
    seen: dict[tuple, str] = {}
    kept = []
    for it in items:
        key = tuple(np.round(twist_of(it), 6))
        if key in seen:
            print(f"  duplicate twist {list(key)}: keeping {seen[key]}, dropping {name_of(it)}")
            continue
        seen[key] = name_of(it)
        kept.append(it)
    return kept


def load_library(
    dirs: list[Path],
    source: str,
    model_id: str,
    fps: float = constants.MJLAB_MOTION_FPS,
    idle: bool = False,
    glob: str | None = None,
) -> list[LibraryClip]:
    """Load one or more library folders as `LibraryClip`s (duplicates by twist dropped).

    Args:
        source: ``"mjlab"`` for converted folders (``convert_gait_library.py``), or
            ``"mj_nlp_out"`` for mj-nlp release folders (or legacy working folders),
            converted in memory exactly as ``convert_gait_library.py`` would: each node
            starts at the OSL touchdown, so folders combine without a shared reference.
        idle: mj_nlp_out only — add the model's ``stand`` keyframe as the zero-twist
            node when no folder has an ``idle.npz``.
        glob: mj_nlp_out legacy folders only (no ``library.json``) — node files.
    """
    model, _ = gl._models(model_id)

    if source == "mjlab":
        clips = []
        for d in dirs:
            got = _load_mjlab_dir(Path(d), model, model_id)
            print(f"  {d}: {len(got)} clips{_manifest_note(Path(d))}")
            clips.extend(got)
        clips = _dedupe(clips, lambda c: c.twist, lambda c: c.name)
        if idle:
            print("  note: --idle only applies to mj_nlp_out; mjlab folders carry their own idle.npz")
    elif source == "mj_nlp_out":
        clips = []
        for d in dirs:
            rel = gl.read_release(Path(d), glob)
            conv = gl.convert_release(rel, model_id, fps, phase="touchdown")
            kind = "release" if rel.manifest is not None else "legacy folder"
            print(f"  {d}: {kind}, {len(conv.clips)} nodes" + (" + idle.npz" if conv.idle else ""))
            for w in conv.warnings:
                print(f"    warning: {w}")
            for r in conv.rows:
                if any(v > gl.SEAM_TOL for v in r["seam"].values()):
                    print(f"    WARNING: {r['name']} fails the seam check (convert would refuse it)")
            raw = conv.clips + ([conv.idle] if conv.idle is not None else [])
            clips.extend(_from_gl_clip(c, model, model_id) for c in raw)
        clips = _dedupe(clips, lambda c: c.twist, lambda c: c.name)
        if idle and not any(c.is_idle for c in clips):
            clips.append(_from_gl_clip(gl.idle_clip(model_id, int(round(fps)))[0], model, model_id))
    else:
        raise ValueError(f"unsupported library source {source!r}")

    if not clips:
        raise ValueError(f"no clips found in {[str(d) for d in dirs]}")
    return clips


def _manifest_note(d: Path) -> str:
    """Describe a converted folder from its library.json; warn if not touchdown-phased.

    The phase is read from clipper's block rather than re-detected: on 50 Hz frames a
    short lift during stance can shift the detected touchdown by a few frames.
    """
    f = d / gl.MANIFEST
    if not f.exists():
        return (
            "\n  WARNING: no library.json, so converted before the release scheme: frame 0 "
            "is not the OSL touchdown. Re-convert with scripts/convert_gait_library.py."
        )
    try:
        doc = json.loads(f.read_text())
    except ValueError:
        return " (unreadable library.json)"
    block = doc.get("clipper") or {}
    note = f" (release {doc.get('release')}, clipper {block.get('git_sha')})"
    if block.get("phase") != "touchdown":
        note += (
            f"\n  WARNING: converted with phase {block.get('phase')!r}, not the OSL "
            "touchdown; it will not share phase 0 with other releases."
        )
    return note


# --------------------------------------------------------------------------- #
# Twist selection
# --------------------------------------------------------------------------- #
class TwistSelector:
    """Library twist ranges, nearest-node lookup, and stick -> command mapping."""

    def __init__(self, twists: np.ndarray):
        self.twists = np.asarray(twists, np.float64).reshape(-1, 3)
        self.lo = np.minimum(self.twists.min(axis=0), 0.0)
        self.hi = np.maximum(self.twists.max(axis=0), 0.0)
        span = self.hi - self.lo
        self.varies = span > 1e-9
        w = np.zeros(3)
        w[self.varies] = 1.0 / span[self.varies] ** 2
        if w.max() > 0:
            w /= w.max()
        self.weights = w

    def clamp(self, cmd) -> np.ndarray:
        cmd = np.clip(np.asarray(cmd, np.float64), self.lo, self.hi)
        cmd[~self.varies] = 0.0
        return cmd

    def nearest(self, cmd) -> int:
        d = ((self.twists - np.asarray(cmd, np.float64)) ** 2 * self.weights).sum(axis=1)
        return int(np.argmin(d))

    def from_stick(self, s) -> np.ndarray:
        """Stick values in [-1, 1] per axis -> command; +1 -> max, -1 -> min, 0 -> 0."""
        s = np.clip(np.asarray(s, np.float64), -1.0, 1.0)
        cmd = np.where(s >= 0.0, s * self.hi, -s * self.lo)
        cmd[~self.varies] = 0.0
        return cmd


# --------------------------------------------------------------------------- #
# Player
# --------------------------------------------------------------------------- #
class LibraryPlayer:
    """Loop the nearest clip to the command, re-anchoring and crossfading on switches."""

    def __init__(
        self,
        clips: list[LibraryClip],
        model_id: str,
        fps: float = constants.MJLAB_MOTION_FPS,
        blend_s: float = 0.4,
    ):
        self.clips = clips
        self.fwd = constants.FWD_AXIS_BODY[model_id]
        self.fps = float(fps)
        self.blend_s = float(blend_s)
        self.selector = TwistSelector(np.stack([c.twist for c in clips]))
        start = self.selector.nearest(np.zeros(3))
        self.cur = start
        self.anchor = np.eye(3)
        self.phase = 0.0
        self.prev: int | None = None
        self.prev_anchor = np.eye(3)
        self.alpha = 1.0  # blend progress: 1 = done
        self.loops = 0
        self.cmd = np.zeros(3)

    # -- queries ---------------------------------------------------------- #
    @property
    def clip(self) -> LibraryClip:
        return self.clips[self.cur]

    def _world(self, idx: int, anchor: np.ndarray) -> np.ndarray:
        c = self.clips[idx]
        return apply_se2(anchor, sample(c, self.phase * c.n))

    def qpos(self) -> np.ndarray:
        """World qpos on screen now (blended while a crossfade is running)."""
        cur = self._world(self.cur, self.anchor)
        if self.prev is None or self.alpha >= 1.0:
            return cur
        return blend_qpos(self._world(self.prev, self.prev_anchor), cur, self.alpha)

    # -- control ---------------------------------------------------------- #
    def command(self, cmd) -> bool:
        """Set the commanded twist; switch clips if the nearest node changed."""
        self.cmd = self.selector.clamp(cmd)
        idx = self.selector.nearest(self.cmd)
        if idx == self.cur:
            return False
        shown = planar_pose(self.qpos(), self.fwd)
        new = self.clips[idx]
        local = planar_pose(sample(new, self.phase * new.n), self.fwd)
        if self.blend_s > 0.0:
            self.prev, self.prev_anchor, self.alpha = self.cur, self.anchor, 0.0
        else:
            self.prev, self.alpha = None, 1.0
        self.cur = idx
        self.anchor = shown @ np.linalg.inv(local)
        return True

    def step(self, dt: float) -> np.ndarray:
        """Advance by `dt` seconds of motion time; return the world qpos."""
        self.phase += dt * self.fps / self.clip.n
        while self.phase >= 1.0:
            self.phase -= 1.0
            self.anchor = self.anchor @ self.clip.stride
            if self.prev is not None and self.alpha < 1.0:
                self.prev_anchor = self.prev_anchor @ self.clips[self.prev].stride
            self.loops += 1
        if self.prev is not None and self.alpha < 1.0:
            self.alpha = min(1.0, self.alpha + dt / self.blend_s)
            if self.alpha >= 1.0:
                self.prev = None
        return self.qpos()
