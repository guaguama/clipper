"""Loader for bones_seed mocap CSVs.

Verified schema (see clipper memory `clipper-data-sources`):
- Header row, then 36 columns: frame, root_translate XYZ, root_rotate XYZ,
  29 joint DOFs (header suffix ``_dof``).
- Translation is in CENTIMETERS; already Z-up. Rotations/joints in DEGREES;
  root Euler order ``'xyz'``. No stored frame rate.
- The 29 joint columns are in `constants.G1_29DOF_JOINT_NAMES` order.

Per-row conversion: pos = cm/100, quat = R.from_euler('xyz', deg).as_quat() ->
wxyz, joints = deg2rad. The raw mocap world-z is loaded verbatim (no grounding),
so the trajectory reproduces the exact input height.
"""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from .. import constants
from ..qpos import build_qpos
from ..trajectory import Trajectory

SOURCE_JOINT_NAMES = constants.G1_29DOF_JOINT_NAMES  # source DOF column order
N_COLS = 1 + 3 + 3 + len(SOURCE_JOINT_NAMES)  # frame + pos + rot + joints = 36


def load(
    path: str | Path,
    model: mujoco.MjModel,
    model_id: str,
    source: str = "bones_seed",
    fps: float | None = None,
) -> Trajectory:
    """Load a bones_seed CSV into a `Trajectory` in `model`'s qpos layout."""
    path = Path(path)
    raw = np.loadtxt(path, delimiter=",", skiprows=1, dtype=np.float64)
    if raw.ndim == 1:
        raw = raw[None, :]
    if raw.shape[1] != N_COLS:
        raise ValueError(
            f"{path.name}: expected {N_COLS} columns; got {raw.shape[1]}"
        )

    base_pos = raw[:, 1:4] / 100.0  # cm -> m
    rot_deg = raw[:, 4:7]
    quat_xyzw = Rotation.from_euler("xyz", rot_deg, degrees=True).as_quat()
    base_quat_wxyz = quat_xyzw[:, [3, 0, 1, 2]]
    joint_rad = np.deg2rad(raw[:, 7:7 + len(SOURCE_JOINT_NAMES)])
    joint_angles = {
        name: joint_rad[:, i] for i, name in enumerate(SOURCE_JOINT_NAMES)
    }

    qpos = build_qpos(model, base_pos, base_quat_wxyz, joint_angles)

    if fps is None:
        fps = constants.BONES_SEED_DEFAULT_FPS

    return Trajectory(
        qpos=qpos,
        fps=float(fps),
        model=model_id,
        source=source,
        name=path.stem,
    )
