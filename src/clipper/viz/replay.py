"""Pure-MuJoCo replay of a clipper `Trajectory` in the passive viewer.

Replay is purely kinematic: each frame's qpos is written and `mj_forward` is
called (no dynamics). `set_pose` carries a `z_offset` argument now so the later
height-editing / scrubbing feature can reuse it unchanged.
"""

from __future__ import annotations

import math
import time

import mujoco
import mujoco.viewer
import numpy as np

from ..trajectory import Trajectory
from .camera import center_on_pelvis
from .markers import draw_markers


def _default_render_flags_off(viewer: "mujoco.viewer.Handle") -> None:
    """Default shadows and reflections off in the passive viewer.

    The flags are written to `user_scn`, which MuJoCo applies edge-triggered (only
    when a value changes), so this sets the *initial* state while leaving the
    viewer's own Shadow/Reflection checkboxes free to turn them back on.
    """
    flags = viewer.user_scn.flags
    flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0
    flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = 0


def grid_offsets(n: int, spacing: float) -> np.ndarray:
    """Return ``(n, 2)`` xy offsets for `n` clones on a near-square grid.

    Row-major, ``ceil(sqrt(n))`` columns, neighbours `spacing` m apart, and the
    grid's bounding box centered on (0, 0) so the camera's pelvis lookat stays in
    the middle. ``n == 1`` gives a single zero offset.
    """
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)
    idx = np.arange(n)
    xy = np.stack([idx % cols, idx // cols], axis=1).astype(np.float64)
    xy -= [(cols - 1) / 2.0, (rows - 1) / 2.0]
    return xy * spacing


def grid_radius(offsets: np.ndarray | None) -> float:
    """Distance from the grid center to its farthest clone (0 for no clones)."""
    if offsets is None:
        return 0.0
    return float(np.linalg.norm(offsets, axis=1).max())


def set_pose(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    traj: Trajectory,
    frame: int,
    z_offset: float = 0.0,
    joint_offsets: dict[int, float] | None = None,
    clone_offsets: np.ndarray | None = None,
) -> None:
    """Place the model at trajectory frame `frame`, with optional pose offsets.

    Args:
        z_offset: meters added to the base height (qpos[2]).
        joint_offsets: optional ``{qpos_address: delta_radians}`` added to specific
            joints (e.g. the scrub viewer's right-ankle correction). ``None`` leaves
            the pose untouched, so replay / video output are unaffected.
        clone_offsets: ``(N, 2)`` xy offsets from `grid_offsets` when `model` was
            built with ``load_model(..., clones=N)``. The single pose (offsets
            included) is tiled N times and each copy's base shifted by its row.
    """
    qpos = traj.qpos[frame].copy()
    qpos[2] += z_offset
    if joint_offsets:
        for adr, delta in joint_offsets.items():
            qpos[adr] += delta
    if clone_offsets is None:
        data.qpos[:] = qpos
    else:
        blocks = np.tile(qpos, (len(clone_offsets), 1))
        blocks[:, :2] += clone_offsets
        data.qpos[:] = blocks.ravel()
    mujoco.mj_forward(model, data)


def draw_clone_markers(
    scn: "mujoco.MjvScene", points: np.ndarray, clone_offsets: np.ndarray | None
) -> None:
    """Draw a frame's markers once per clone, each shifted by its grid offset."""
    if clone_offsets is None:
        draw_markers(scn, points)
        return
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    for dx, dy in clone_offsets:
        draw_markers(scn, pts + [dx, dy, 0.0])


def replay(
    model: mujoco.MjModel,
    traj: Trajectory,
    fps: float | None = None,
    loop: bool = True,
    speed: float = 1.0,
    clone_offsets: np.ndarray | None = None,
) -> None:
    """Play `traj` back in a passive MuJoCo viewer.

    Args:
        model: Compiled model whose qpos layout matches `traj.qpos`.
        traj: Trajectory to play.
        fps: Playback rate; defaults to `traj.fps`.
        loop: Restart from frame 0 at the end instead of stopping.
        speed: Real-time multiplier (2.0 = twice as fast).
        clone_offsets: Grid offsets when `model` holds several copies; see
            `set_pose`.
    """
    fps = float(fps or traj.fps)
    dt = 1.0 / (fps * max(speed, 1e-6))
    n = traj.num_frames

    data = mujoco.MjData(model)
    print(
        f"Replaying '{traj.name}' [{traj.source} -> {traj.model}]: "
        f"{n} frames @ {fps:g} Hz ({traj.duration:.2f} s)"
        f"{' x%g' % speed if speed != 1.0 else ''}. Close the window to stop."
    )

    with mujoco.viewer.launch_passive(
        model, data, show_left_ui=False, show_right_ui=False
    ) as viewer:
        _default_render_flags_off(viewer)
        center_on_pelvis(  # one-time start frame; user owns it after
            viewer.cam, traj.qpos[0], traj.model, extra_distance=grid_radius(clone_offsets)
        )
        k = 0
        while viewer.is_running():
            tic = time.perf_counter()
            set_pose(model, data, traj, k, clone_offsets=clone_offsets)
            if traj.markers is not None:
                viewer.user_scn.ngeom = 0
                draw_clone_markers(viewer.user_scn, traj.markers[k], clone_offsets)
            viewer.sync()
            print(
                f"  frame {k + 1:>6d}/{n} "
                f"({100.0 * (k + 1) / n:5.1f}%)  t={k / fps:6.2f}s",
                end="\r",
            )
            k += 1
            if k >= n:
                if not loop:
                    break
                k = 0
            elapsed = time.perf_counter() - tic
            if elapsed < dt:
                time.sleep(dt - elapsed)
    print()
