"""Shared start-camera framing for clipper viewers.

Every viewer *starts* framed the same way: a free camera centered on the robot's
pelvis (``qpos[:3]``) at a fixed azimuth / elevation / distance. From there the
paths diverge by design — the offscreen video renderer re-pins the lookat to the
pelvis every frame (tracking, fixed world angle), while the interactive passive
viewers set this once and then hand the camera over to the user's mouse.

The lookat comes from the trajectory, not ``model.stat.center``, so the framing
is identical across models and source formats.
"""

from __future__ import annotations

import mujoco

# Reused from the prior --save-video defaults (model.py ``visual.global_`` angles
# and 1.5 x the 0.8 ``stat.extent``), so the video zoom/angle is unchanged.
AZIMUTH =   140.0
ELEVATION = -20.0
DISTANCE = 2.0


def center_on_pelvis(cam: "mujoco.MjvCamera", qpos_frame) -> None:
    """Point a free ``MjvCamera`` at the pelvis of this qpos frame.

    Azimuth / elevation / distance are held at the shared constants; only the
    lookat moves, so calling this per frame keeps the robot centered at a fixed
    world viewing angle.

    Args:
        cam: A free camera (``MjvCamera``); the passive viewer's ``viewer.cam``
            or a fresh ``mujoco.MjvCamera()`` for the offscreen renderer.
        qpos_frame: A trajectory frame ``[base_pos(3), base_quat(4), joints]``;
            the first 3 entries are the world pelvis position.
    """
    cam.azimuth = AZIMUTH
    cam.elevation = ELEVATION
    cam.distance = DISTANCE
    cam.lookat[:] = qpos_frame[:3]
