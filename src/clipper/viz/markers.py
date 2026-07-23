"""Overlay point markers (spheres) into a MuJoCo scene.

Used to draw a trajectory's optional ``markers`` (e.g. a retarget's source mocap
skeleton) on top of the robot. Works for either an interactive viewer's
``user_scn`` or the offscreen renderer's ``scene`` — both are ``MjvScene`` with a
``geoms`` array, an ``ngeom`` count, and a ``maxgeom`` capacity. The spheres are
(re)written into the leading geom slots each call, so callers invoke it per frame.
"""

from __future__ import annotations

import numpy as np
import mujoco

# Distinct, semi-transparent so the robot stays readable underneath.
DEFAULT_RGBA = (1.0, 0.55, 0.0, 0.75)  # orange
DEFAULT_RADIUS = 0.03  # meters

_IDENTITY_MAT = np.eye(3, dtype=np.float64).flatten()


def draw_markers(
    scn: "mujoco.MjvScene",
    points: np.ndarray,
    rgba: tuple[float, float, float, float] = DEFAULT_RGBA,
    radius: float = DEFAULT_RADIUS,
) -> None:
    """Draw ``points`` (K, 3) as spheres into the leading geom slots of ``scn``.

    For an interactive ``viewer.user_scn`` the geoms overlay the base scene; for an
    offscreen ``renderer.scene`` call this AFTER ``update_scene`` so the spheres are
    appended on top. Silently no-ops on empty input and clamps to ``scn.maxgeom``.
    """
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    k = min(len(pts), int(scn.maxgeom) - int(scn.ngeom))
    if k <= 0:
        return
    size = np.array([radius, 0.0, 0.0], dtype=np.float64)
    rgba_arr = np.asarray(rgba, dtype=np.float32)
    base = int(scn.ngeom)
    for i in range(k):
        mujoco.mjv_initGeom(
            scn.geoms[base + i],
            mujoco.mjtGeom.mjGEOM_SPHERE,
            size,
            pts[i],
            _IDENTITY_MAT,
            rgba_arr,
        )
    scn.ngeom = base + k
