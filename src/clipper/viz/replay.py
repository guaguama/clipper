"""Pure-MuJoCo replay of a clipper `Trajectory` in the passive viewer.

Replay is purely kinematic: each frame's qpos is written and `mj_forward` is
called (no dynamics). `set_pose` carries a `z_offset` argument now so the later
height-editing / scrubbing feature can reuse it unchanged.
"""

from __future__ import annotations

import time

import mujoco
import mujoco.viewer

from ..trajectory import Trajectory


def _default_render_flags_off(viewer: "mujoco.viewer.Handle") -> None:
    """Default shadows and reflections off in the passive viewer.

    The flags are written to `user_scn`, which MuJoCo applies edge-triggered (only
    when a value changes), so this sets the *initial* state while leaving the
    viewer's own Shadow/Reflection checkboxes free to turn them back on.
    """
    flags = viewer.user_scn.flags
    flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0
    flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = 0


def set_pose(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    traj: Trajectory,
    frame: int,
    z_offset: float = 0.0,
) -> None:
    """Place the model at trajectory frame `frame`, optionally raised by z_offset."""
    data.qpos[:] = traj.qpos[frame]
    data.qpos[2] += z_offset
    mujoco.mj_forward(model, data)


def replay(
    model: mujoco.MjModel,
    traj: Trajectory,
    fps: float | None = None,
    loop: bool = True,
    speed: float = 1.0,
) -> None:
    """Play `traj` back in a passive MuJoCo viewer.

    Args:
        model: Compiled model whose qpos layout matches `traj.qpos`.
        traj: Trajectory to play.
        fps: Playback rate; defaults to `traj.fps`.
        loop: Restart from frame 0 at the end instead of stopping.
        speed: Real-time multiplier (2.0 = twice as fast).
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
        k = 0
        while viewer.is_running():
            tic = time.perf_counter()
            set_pose(model, data, traj, k)
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
