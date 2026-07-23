"""Loader for holosoma retargeted npz trajectories.

Produced by holosoma's interaction-mesh retargeter (see ``../holosoma``); a clip is
an ``.npz`` with:

- ``qpos`` (T, 36) = base_pos(3) + base_quat_wxyz(4) + 29 joint angles (rad).
- ``human_joints`` (T, 22, 3) = the source SMPL body skeleton (world positions, m),
  carried through as a visualization overlay (`Trajectory.markers`).
- ``fps`` scalar (30 for the current retargets).
- ``cost`` scalar retarget residual (diagnostic; ignored).

holosoma targets the Unitree G1 29-DOF, and its 29-joint column order is identical to
``constants.G1_29DOF_JOINT_NAMES`` (the standard ``_joint``-suffixed G1 order); the
quaternion is already MuJoCo wxyz and positions are meters/z-up. So the qpos is already
clipper's canonical g1_29dof layout — no unit conversion, quaternion reorder, or floor
grounding. The by-name `build_qpos` still lets it target g1_23dof (extra waist/wrist
DOFs drop) or other models.
"""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np

from .. import constants
from ..qpos import build_qpos
from ..trajectory import Trajectory

SOURCE_JOINT_NAMES = constants.G1_29DOF_JOINT_NAMES  # source DOF column order


def load(
    path: str | Path,
    model: mujoco.MjModel,
    model_id: str,
    source: str = "holosoma",
    fps: float | None = None,
) -> Trajectory:
    """Load a holosoma retargeted npz into a `Trajectory` in `model`'s qpos layout."""
    path = Path(path)
    data = np.load(path)
    raw = np.asarray(data["qpos"], dtype=np.float64)
    if raw.ndim != 2 or raw.shape[1] != 7 + len(SOURCE_JOINT_NAMES):
        raise ValueError(
            f"{path.name}: expected qpos (N, {7 + len(SOURCE_JOINT_NAMES)}); "
            f"got {raw.shape} (object/interaction-task clips are not supported)"
        )

    base_pos = raw[:, 0:3]
    base_quat_wxyz = raw[:, 3:7]  # already wxyz
    joint_angles = {
        name: raw[:, 7 + i] for i, name in enumerate(SOURCE_JOINT_NAMES)
    }

    qpos = build_qpos(model, base_pos, base_quat_wxyz, joint_angles)

    if fps is None:
        if "fps" in data.files:
            fps = float(np.asarray(data["fps"]).reshape(-1)[0])
        else:
            fps = constants.HOLOSOMA_DEFAULT_FPS

    markers = None
    if "human_joints" in data.files:
        markers = np.asarray(data["human_joints"], dtype=np.float64)

    return Trajectory(
        qpos=qpos,
        fps=float(fps),
        model=model_id,
        source=source,
        name=path.stem,
        markers=markers,
    )
