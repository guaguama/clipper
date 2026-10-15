"""Loader for clipper's own mjlab motion NPZ (source id ``mjlab``).

Re-loads the motion-tracking NPZ produced by `clipper.writers.mjlab` so a written
clip can be visualized or re-cropped — the loader/writer pair is symmetric. The same
file is what ``../unitree_rl_mjlab`` consumes (the two repos read the identical
arrays), so this also reads clips written for the fork, including older ones under
``outputs/unitree_rl_mjlab/``.

Schema (see ``writers/mjlab.py``):
- ``joint_pos`` ``(N, DOF)`` — the model's joints in qpos order.
- ``body_pos_w`` ``(N, B, 3)`` / ``body_quat_w`` ``(N, B, 4)`` — world body pose
  (quat wxyz); body 0 is the root, so its pose IS the free-joint base.
- ``fps`` ``(1,)`` — frame rate.

The NPZ does not record which model wrote it, so the joint order is resolved from the
DOF count, **the target model's own first**: DOF counts can collide across model
families (the former ``osl_ka`` and ``g1_23dof`` were both 23 DOF), and reading one as
the other would map onto joint names that match nothing, so
`build_qpos` would return an all-zero-joint clip with no exception. Only when the DOF
count is *not* the target's does the G1 table apply, which serves the cross-model case
(a 29-DOF clip viewed on g1_23dof and vice-versa; extra DOFs fill 0 / drop).

The base pose is read from body 0 rather than from the joint block, so the body count
never has to be known ahead of time — it varies by model (30 for either G1, 105 for
``osl_fullbody``). Velocities in the NPZ are derived data and are ignored on load.
"""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np

from .. import constants
from ..model import joint_names
from ..qpos import build_qpos
from ..trajectory import Trajectory

# DOF count -> source joint column order, for the cross-model G1 case only.
_JOINT_NAMES_BY_DOF: dict[int, tuple[str, ...]] = {
    len(constants.G1_23DOF_JOINT_NAMES): constants.G1_23DOF_JOINT_NAMES,
    len(constants.G1_29DOF_JOINT_NAMES): constants.G1_29DOF_JOINT_NAMES,
}


def load(
    path: str | Path,
    model: mujoco.MjModel,
    model_id: str,
    source: str = "mjlab",
    fps: float | None = None,
) -> Trajectory:
    """Load an mjlab motion NPZ into a `Trajectory` in `model`'s qpos layout."""
    path = Path(path)
    data = np.load(path)

    missing = [k for k in ("joint_pos", "body_pos_w", "body_quat_w") if k not in data]
    if missing:
        raise ValueError(
            f"{path.name}: not an mjlab motion NPZ (missing {missing}). "
            f"Keys present: {sorted(data.files)}."
        )

    joint_pos = np.asarray(data["joint_pos"], dtype=np.float64)
    if joint_pos.ndim != 2:
        raise ValueError(
            f"{path.name}: expected joint_pos (N, DOF); got {joint_pos.shape}"
        )
    n, dof = joint_pos.shape

    # The target model's own layout wins (see the module docstring).
    target_names = joint_names(model_id)
    if dof == len(target_names):
        source_joint_names = target_names
    else:
        source_joint_names = _JOINT_NAMES_BY_DOF.get(dof)
    if source_joint_names is None:
        raise ValueError(
            f"{path.name}: joint_pos has {dof} DOF; expected {len(target_names)} for "
            f"{model_id!r}, or one of {sorted(_JOINT_NAMES_BY_DOF)} for a cross-model "
            f"G1 clip."
        )

    body_pos_w = np.asarray(data["body_pos_w"], dtype=np.float64)
    body_quat_w = np.asarray(data["body_quat_w"], dtype=np.float64)
    if body_pos_w.ndim != 3 or body_pos_w.shape[0] != n or body_pos_w.shape[2] != 3:
        raise ValueError(
            f"{path.name}: expected body_pos_w ({n}, B, 3); got {body_pos_w.shape}"
        )
    if body_quat_w.ndim != 3 or body_quat_w.shape[0] != n or body_quat_w.shape[2] != 4:
        raise ValueError(
            f"{path.name}: expected body_quat_w ({n}, B, 4); got {body_quat_w.shape}"
        )

    base_pos = body_pos_w[:, 0]  # root body == free-joint base
    base_quat_wxyz = body_quat_w[:, 0]  # already wxyz
    joint_angles = {
        name: joint_pos[:, i] for i, name in enumerate(source_joint_names)
    }

    qpos = build_qpos(model, base_pos, base_quat_wxyz, joint_angles)

    if fps is None and "fps" in data:
        stored = np.atleast_1d(np.asarray(data["fps"], dtype=np.float64))
        if stored.size and stored.flat[0] > 0.0:
            fps = float(stored.flat[0])
    if fps is None:
        # The writer always stores `fps`; fall back to mjlab's tracking step rate.
        fps = constants.MJLAB_MOTION_FPS

    return Trajectory(
        qpos=qpos,
        fps=float(fps),
        model=model_id,
        source=source,
        name=path.stem,
    )
