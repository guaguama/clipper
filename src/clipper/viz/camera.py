"""Shared start-camera framing for clipper viewers.

Every viewer *starts* framed the same way: a free camera centered on the robot's
pelvis (``qpos[:3]``) at a fixed azimuth / elevation / distance. From there the
paths diverge by design — the offscreen video renderer re-pins the lookat to the
pelvis every frame (tracking, fixed world angle), while the interactive passive
viewers set this once and then hand the camera over to the user's mouse.

The lookat comes from the trajectory, not ``model.stat.center``, so the framing
is identical across source formats. The angle/zoom preset, however, is chosen per
*model family*: the G1 robots and the musculoskeletal (MSK) models each get their
own ``CameraView`` so they can be tuned independently.
"""

from __future__ import annotations

from collections import namedtuple

import mujoco

from .. import constants

CameraView = namedtuple("CameraView", "azimuth elevation distance")

# G1 robots. Reused from the prior --save-video defaults (model.py
# ``visual.global_`` angles and 1.5 x the 0.8 ``stat.extent``); unchanged.
G1_VIEW = CameraView(azimuth=140.0, elevation=-20.0, distance=2.0)

# MSK models (osl_ka / myofullbody / myolegs). Tuned separately — adjust these
# three numbers to taste. The G1 preset framed the MSK models from behind and a
# touch too close, so this faces the front (azimuth flipped ~180°) and pulls back.
MSK_VIEW = CameraView(azimuth=-40.0, elevation=-20.0, distance=3.0)


def _view_for(model_id: str | None) -> CameraView:
    """Return the start-camera preset for a model id (G1_VIEW if unknown/None)."""
    if model_id is not None and model_id in constants.MSK_MODELS:
        return MSK_VIEW
    return G1_VIEW


def center_on_pelvis(
    cam: "mujoco.MjvCamera", qpos_frame, model_id: str | None = None
) -> None:
    """Point a free ``MjvCamera`` at the pelvis of this qpos frame.

    Azimuth / elevation / distance come from the model family's ``CameraView``
    (G1 vs MSK); only the lookat moves, so calling this per frame keeps the robot
    centered at a fixed world viewing angle.

    Args:
        cam: A free camera (``MjvCamera``); the passive viewer's ``viewer.cam``
            or a fresh ``mujoco.MjvCamera()`` for the offscreen renderer.
        qpos_frame: A trajectory frame ``[base_pos(3), base_quat(4), joints]``;
            the first 3 entries are the world pelvis position.
        model_id: Trajectory's target model; selects the preset. ``None`` falls
            back to the G1 preset.
    """
    view = _view_for(model_id)
    cam.azimuth = view.azimuth
    cam.elevation = view.elevation
    cam.distance = view.distance
    cam.lookat[:] = qpos_frame[:3]
