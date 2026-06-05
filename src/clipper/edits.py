"""Pure trajectory edits: crop, global height offset, and standing-pose padding.

Each function takes a `Trajectory` and returns a NEW one (qpos is copied, never
mutated in place); `fps`, `model`, and `source` carry over and `name` records the
edit. Edits operate in the trajectory's own qpos layout
(``[base_pos(3), base_quat_wxyz(4), joints(DOF)]``), so they are model-agnostic and
compose in the pipeline order ``crop -> apply_height_offset -> pad_standing``.
"""

from __future__ import annotations

import numpy as np

from . import constants, mathx
from .trajectory import Trajectory


def crop(traj: Trajectory, start: int | None = None, stop: int | None = None) -> Trajectory:
    """Crop to the inclusive 0-based frame range ``[start, stop]``.

    ``start``/``stop`` default to the first/last frame. Bounds match the indices
    printed by the scrub viewer's ``f`` marker (0-based ``frame k/N``).

    Raises:
        ValueError: if the range is out of bounds or inverted.
    """
    n = traj.num_frames
    start = 0 if start is None else int(start)
    stop = n - 1 if stop is None else int(stop)
    if not (0 <= start <= stop <= n - 1):
        raise ValueError(
            f"crop range [{start}, {stop}] invalid for {n} frames "
            f"(need 0 <= start <= stop <= {n - 1})."
        )
    return Trajectory(
        qpos=traj.qpos[start : stop + 1].copy(),
        fps=traj.fps,
        model=traj.model,
        source=traj.source,
        name=f"{traj.name}_crop{start}-{stop}",
    )


def apply_height_offset(traj: Trajectory, dz: float) -> Trajectory:
    """Add `dz` (meters) to the base z of every frame (matches set_pose z_offset)."""
    q = traj.qpos.copy()
    q[:, 2] += float(dz)
    name = traj.name if dz == 0.0 else f"{traj.name}_z{dz:+g}"
    return Trajectory(
        qpos=q, fps=traj.fps, model=traj.model, source=traj.source, name=name
    )


def _standing_qpos(traj: Trajectory, anchor: int) -> np.ndarray:
    """Home standing qpos, inheriting the anchor frame's xy + yaw (kept upright)."""
    stand = constants.home_qpos(traj.model).copy()
    stand[0:2] = traj.qpos[anchor, 0:2]  # inherit ground xy
    stand[3:7] = mathx.yaw_only_quat(traj.qpos[anchor, 3:7])  # inherit heading only
    return stand


def _blend_frames(a: np.ndarray, b: np.ndarray, n_blend: int) -> np.ndarray:
    """`n_blend` interior frames from a -> b (exclusive of both endpoints).

    lerp for base position + joints, slerp for the base quaternion.
    """
    if n_blend <= 0:
        return np.zeros((0, a.shape[0]), dtype=np.float64)
    t = (np.arange(1, n_blend + 1) / (n_blend + 1.0))[:, None]
    out = mathx.lerp(a[None, :], b[None, :], t)  # (n_blend, nq)
    out[:, 3:7] = mathx.slerp(
        np.broadcast_to(a[3:7], (n_blend, 4)),
        np.broadcast_to(b[3:7], (n_blend, 4)),
        t[:, 0],
    )
    return out


def pad_standing(
    traj: Trajectory,
    pre_static: float = 1.0,
    pre_blend: float = 0.5,
    post_static: float = 1.0,
    post_blend: float = 0.5,
) -> Trajectory:
    """Prepend/append a standing pose with a slerp/lerp transition at each end.

    The standing pose is the model's home keyframe with the adjacent end frame's
    xy + yaw grafted on (upright, at nominal home height). Durations are in
    seconds and converted to frame counts via the trajectory fps.

    Args:
        traj: Trajectory to pad.
        pre_static: Seconds held standing before the motion.
        pre_blend: Seconds blending standing -> first motion frame.
        post_static: Seconds held standing after the motion.
        post_blend: Seconds blending last motion frame -> standing.
    """
    fps = traj.fps
    n_pre_s = int(round(pre_static * fps))
    n_pre_b = int(round(pre_blend * fps))
    n_post_s = int(round(post_static * fps))
    n_post_b = int(round(post_blend * fps))

    q = traj.qpos
    stand_front = _standing_qpos(traj, 0)
    stand_back = _standing_qpos(traj, traj.num_frames - 1)

    segments = []
    if n_pre_s:
        segments.append(np.broadcast_to(stand_front, (n_pre_s, q.shape[1])).copy())
    segments.append(_blend_frames(stand_front, q[0], n_pre_b))
    segments.append(q.copy())
    segments.append(_blend_frames(q[-1], stand_back, n_post_b))
    if n_post_s:
        segments.append(np.broadcast_to(stand_back, (n_post_s, q.shape[1])).copy())

    return Trajectory(
        qpos=np.concatenate(segments, axis=0),
        fps=fps,
        model=traj.model,
        source=traj.source,
        name=f"{traj.name}_pad",
    )
