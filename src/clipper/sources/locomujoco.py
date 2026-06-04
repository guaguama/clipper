"""Loader for LocoMuJoCo-format npz trajectories (DefaultDatasets, Lafan1).

Verified schema (see clipper memory `clipper-data-sources`):
- ``qpos`` (N, 30) = base_pos(3) + base_quat_wxyz(4) + 23 joint angles (rad).
- Quaternion is already MuJoCo wxyz — no conversion.
- The 23 joints are in the exact `constants.G1_23DOF_JOINT_NAMES` order.
- ``frequency`` holds the frame rate (40 Hz for the current datasets).

These map to the g1_23dof model, but the by-name `build_qpos` lets them be
visualized on g1_29dof too (the 6 extra DOFs stay at 0).
"""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np

from .. import constants
from ..qpos import build_qpos
from ..trajectory import Trajectory

SOURCE_JOINT_NAMES = constants.G1_23DOF_JOINT_NAMES  # source DOF column order


def load(
    path: str | Path,
    model: mujoco.MjModel,
    model_id: str,
    source: str = "locomujoco",
    fps: float | None = None,
) -> Trajectory:
    """Load a LocoMuJoCo npz into a `Trajectory` in `model`'s qpos layout."""
    path = Path(path)
    data = np.load(path)
    raw = np.asarray(data["qpos"], dtype=np.float64)
    if raw.ndim != 2 or raw.shape[1] != 7 + len(SOURCE_JOINT_NAMES):
        raise ValueError(
            f"{path.name}: expected qpos (N, {7 + len(SOURCE_JOINT_NAMES)}); "
            f"got {raw.shape}"
        )

    base_pos = raw[:, 0:3]
    base_quat_wxyz = raw[:, 3:7]  # already wxyz
    joint_angles = {
        name: raw[:, 7 + i] for i, name in enumerate(SOURCE_JOINT_NAMES)
    }

    qpos = build_qpos(model, base_pos, base_quat_wxyz, joint_angles)

    if fps is None:
        fps = float(np.asarray(data["frequency"]).reshape(-1)[0])

    return Trajectory(
        qpos=qpos,
        fps=float(fps),
        model=model_id,
        source=source,
        name=path.stem,
    )
