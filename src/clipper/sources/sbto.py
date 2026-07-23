"""Loader for clipper's own `sbto` reference NPZ (source id ``sbto``).

Re-loads the NPZ written by ``clipper.writers.sbto`` (and consumed by the
``../sbto`` DynaRetarget/SBTO pipeline), so a converted/cropped reference clip can
be visualized or re-cropped — the loader/writer pair is symmetric.

Distinct from the ``sbto_out`` source, which imports sbto's *solved* output
(``best_trajectory.npz``). This one reads clipper's own reference input NPZ, whose
schema is (see ``writers/sbto.py``):
- ``qpos`` ``(N, 36)`` float64 — one full qpos row per frame, stored
  **OmniRetarget-native** ``[base_quat(4, wxyz), base_pos(3), joints(29)]`` (the
  writer flips ``[pos, quat] -> [quat, pos]`` so sbto loads it with default flags);
- ``fps`` — scalar frame rate (int).

The base flip is reversed here to recover clipper-canonical ``[pos, quat, joints]``.
Joints are in ``constants.G1_29DOF_JOINT_NAMES`` order and map onto the requested
``--model`` by joint name via `build_qpos` (a 29-DOF clip can be viewed on
g1_23dof and vice-versa).
"""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np

from .. import constants
from ..qpos import build_qpos
from ..trajectory import Trajectory

# qpos width (nq) -> source joint column order. The writer always emits the 36-col
# g1_29dof layout, but a 23-col variant is accepted for symmetry/future-proofing.
_JOINT_NAMES_BY_NQ: dict[int, tuple[str, ...]] = {
    7 + len(constants.G1_23DOF_JOINT_NAMES): constants.G1_23DOF_JOINT_NAMES,
    7 + len(constants.G1_29DOF_JOINT_NAMES): constants.G1_29DOF_JOINT_NAMES,
}


def load(
    path: str | Path,
    model: mujoco.MjModel,
    model_id: str,
    source: str = "sbto",
    fps: float | None = None,
) -> Trajectory:
    """Load a clipper sbto reference NPZ into a `Trajectory` in `model`'s layout."""
    path = Path(path)
    data = np.load(path)
    if "qpos" not in data:
        raise ValueError(
            f"{path.name}: no 'qpos' array; not a clipper sbto reference NPZ "
            f"(keys: {list(data.keys())})."
        )
    raw = np.asarray(data["qpos"], dtype=np.float64)
    if raw.ndim == 1:
        raw = raw[None, :]
    source_joint_names = _JOINT_NAMES_BY_NQ.get(raw.shape[1])
    if source_joint_names is None:
        raise ValueError(
            f"{path.name}: qpos width {raw.shape[1]}; expected one of "
            f"{sorted(_JOINT_NAMES_BY_NQ)}."
        )

    # Reverse the writer's base flip: stored [quat(4), pos(3)] -> [pos, quat].
    base_quat_wxyz = raw[:, 0:4]  # already wxyz
    base_pos = raw[:, 4:7]
    joint_angles = {
        name: raw[:, 7 + i] for i, name in enumerate(source_joint_names)
    }

    qpos = build_qpos(model, base_pos, base_quat_wxyz, joint_angles)

    if fps is None and "fps" in data:
        fps = float(np.asarray(data["fps"]).item())
    if fps is None:
        raise ValueError(
            f"{path.name}: no 'fps' in file; pass --fps to set the rate."
        )

    return Trajectory(
        qpos=qpos,
        fps=float(fps),
        model=model_id,
        source=source,
        name=path.stem,
    )
