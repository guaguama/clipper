"""MSK motion NPZ plus paired or explicitly supplied terrain JSON."""

from __future__ import annotations

from pathlib import Path

import mujoco

from .. import constants
from ..terrain import Terrain
from ..trajectory import Trajectory
from . import musclemimic


def load(
    path: str | Path,
    model: mujoco.MjModel,
    model_id: str,
    source: str = "terra",
    fps: float | None = None,
    *,
    terrain: str | Path | None = None,
) -> Trajectory:
    """Map motion by name and keep terrain in the same, unmodified world frame.

    ``terrain`` overrides the default ``<stem>_terrain.json`` next to the motion,
    so a MuscleMimic motion can use an existing terrain without a local sidecar.
    A terrain file is required; flat terrain is represented by ``{"boxes": []}``.
    """
    path = Path(path)
    if model_id not in constants.MSK_MODELS:
        raise ValueError(f"TERRA's MyoFullBody motion requires an MSK model, got {model_id!r}.")
    terrain_path = (
        Path(terrain).expanduser() if terrain is not None
        else path.with_name(f"{path.stem}_terrain.json")
    )
    if not terrain_path.is_file():
        raise FileNotFoundError(
            f"TERRA terrain file not found: {terrain_path}. Supply --terrain <JSON> "
            "or keep <motion_stem>_terrain.json next to the motion NPZ."
        )
    terrain_spec = Terrain.load(terrain_path)
    # Share the motion loader's knee-only cross-model adaptations. Its default
    # preserves solved spine/shoulders for both ordinary caches and TERRA files.
    traj = musclemimic.load(path, model, model_id, source=source, fps=fps)
    traj.terrain = terrain_spec
    return traj
