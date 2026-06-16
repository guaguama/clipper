"""Loader for clipper's own `unitree_rl_mjlab` output NPZ (source id ``unilab``).

This re-loads the motion-tracking NPZ produced by
``clipper.writers.unitree_rl_mjlab`` so a written clip can be visualized or
re-cropped — the loader/writer pair is now symmetric.

Schema (see ``writers/unitree_rl_mjlab.py``):
- ``joint_pos`` ``(N, DOF)`` — actuated joints in ``constants.JOINT_NAMES`` order.
- ``body_pos_w`` ``(N, 30, 3)`` / ``body_quat_w`` ``(N, 30, 4)`` — world body pose
  (quat wxyz); body 0 is the root, so its pose IS the free-joint base.
- ``fps`` ``(1,)`` — frame rate.

The NPZ does not record which model wrote it, but the DOF count disambiguates:
23 -> g1_23dof joints, 29 -> g1_29dof joints. The base pose is read from body 0
and the joints are scattered onto the requested model by name via `build_qpos`,
so a clip can also be loaded onto the other model (extra DOFs fill 0 / drop).
Velocities in the NPZ are derived data and are ignored on load.
"""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np

from .. import constants
from ..qpos import build_qpos
from ..trajectory import Trajectory

N_BODIES = 30

# DOF count -> source joint column order.
_JOINT_NAMES_BY_DOF: dict[int, tuple[str, ...]] = {
    len(constants.G1_23DOF_JOINT_NAMES): constants.G1_23DOF_JOINT_NAMES,
    len(constants.G1_29DOF_JOINT_NAMES): constants.G1_29DOF_JOINT_NAMES,
}


def load(
    path: str | Path,
    model: mujoco.MjModel,
    model_id: str,
    source: str = "unilab",
    fps: float | None = None,
) -> Trajectory:
    """Load a unitree_rl_mjlab NPZ into a `Trajectory` in `model`'s qpos layout."""
    path = Path(path)
    data = np.load(path)

    joint_pos = np.asarray(data["joint_pos"], dtype=np.float64)
    if joint_pos.ndim != 2:
        raise ValueError(
            f"{path.name}: expected joint_pos (N, DOF); got {joint_pos.shape}"
        )
    dof = joint_pos.shape[1]
    source_joint_names = _JOINT_NAMES_BY_DOF.get(dof)
    if source_joint_names is None:
        raise ValueError(
            f"{path.name}: joint_pos has {dof} DOF; expected one of "
            f"{sorted(_JOINT_NAMES_BY_DOF)}."
        )

    body_pos_w = np.asarray(data["body_pos_w"], dtype=np.float64)
    body_quat_w = np.asarray(data["body_quat_w"], dtype=np.float64)
    n = joint_pos.shape[0]
    if body_pos_w.shape != (n, N_BODIES, 3):
        raise ValueError(
            f"{path.name}: expected body_pos_w ({n}, {N_BODIES}, 3); "
            f"got {body_pos_w.shape}"
        )
    if body_quat_w.shape != (n, N_BODIES, 4):
        raise ValueError(
            f"{path.name}: expected body_quat_w ({n}, {N_BODIES}, 4); "
            f"got {body_quat_w.shape}"
        )

    base_pos = body_pos_w[:, 0]  # root body == free-joint base
    base_quat_wxyz = body_quat_w[:, 0]  # already wxyz
    joint_angles = {
        name: joint_pos[:, i] for i, name in enumerate(source_joint_names)
    }

    qpos = build_qpos(model, base_pos, base_quat_wxyz, joint_angles)

    if fps is None:
        fps = float(np.asarray(data["fps"]).reshape(-1)[0])

    return Trajectory(
        qpos=qpos,
        fps=float(fps),
        model=model_id,
        source=source,
        name=path.stem,
    )
