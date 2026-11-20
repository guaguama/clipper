"""Append world-fixed terrain and frame-aligned markers to any MuJoCo scene."""

from __future__ import annotations

import mujoco
import numpy as np

from ..trajectory import Trajectory
from .markers import draw_markers

TERRAIN_RGBA = np.array([0.38, 0.49, 0.58, 1.0], dtype=np.float32)


def draw_overlays(
    scn: mujoco.MjvScene,
    traj: Trajectory,
    frame: int,
    z_offset: float = 0.0,
) -> None:
    """Append after renderer.update_scene, or reset viewer.user_scn first.

    Terrain stays fixed when the scrubber shifts the pose; mocap markers follow
    the pose. Boxes are visual geoms only: they do not change FK or model layout.
    """
    if traj.terrain is not None:
        boxes = traj.terrain.boxes
        if scn.ngeom + len(boxes) > scn.maxgeom:
            raise ValueError(f"MuJoCo scene has no room for {len(boxes)} terrain boxes")
        base = scn.ngeom
        for i, box in enumerate(boxes):
            mujoco.mjv_initGeom(
                scn.geoms[base + i],
                mujoco.mjtGeom.mjGEOM_BOX,
                np.asarray(box.size),
                np.asarray(box.pos),
                box.rotation.ravel(),
                TERRAIN_RGBA,
            )
        scn.ngeom = base + len(boxes)
    if traj.markers is not None:
        points = traj.markers[frame] + np.array([0.0, 0.0, z_offset])
        draw_markers(scn, points)
