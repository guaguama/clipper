"""Loader for `srb_kino` qpos CSVs (source id ``srb_kino``).

These are the headerless per-frame qpos CSVs fed directly into
``../unitree_rl_mjlab/scripts/csv_to_npz.py``. Schema:
- No header; ``(N, nq)`` rows: ``base_pos(3) + base_quat(4) + joints`` in
  ``g1_<dof>dof`` model order (29-DOF -> 36 cols, 23-DOF -> 30 cols).
- The quaternion is **xyzw** (scalar-last), unlike clipper's ``mj_nlp`` source
  which is wxyz. csv_to_npz reorders it the same way (``[:, [3, 0, 1, 2]]``,
  ``csv_to_npz.py``), so we match that to get MuJoCo wxyz.
- No timing information at all (no ``time.csv``, no timestamp column). The rate is
  not stored anywhere; the srb_kino producer emits at 50 Hz, so that is the
  default (`constants.SRB_KINO_DEFAULT_FPS`). Override with ``--fps``.

The qpos columns are in the model's qpos order, so they map onto the requested
``--model`` by joint name via `build_qpos` (a 29-DOF clip can be viewed on
g1_23dof and vice-versa). Read-only import; there is no matching writer.
"""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np

from .. import constants
from ..qpos import build_qpos
from ..trajectory import Trajectory

# qpos width (nq) -> source joint column order. Same mapping as `mj_nlp.py`.
_JOINT_NAMES_BY_NQ: dict[int, tuple[str, ...]] = {
    7 + len(constants.G1_23DOF_JOINT_NAMES): constants.G1_23DOF_JOINT_NAMES,
    7 + len(constants.G1_29DOF_JOINT_NAMES): constants.G1_29DOF_JOINT_NAMES,
}


def load(
    path: str | Path,
    model: mujoco.MjModel,
    model_id: str,
    source: str = "srb_kino",
    fps: float | None = None,
) -> Trajectory:
    """Load an srb_kino qpos CSV into a `Trajectory` in `model`'s qpos layout."""
    path = Path(path)
    raw = np.loadtxt(path, delimiter=",", dtype=np.float64)
    if raw.ndim == 1:
        raw = raw[None, :]
    source_joint_names = _JOINT_NAMES_BY_NQ.get(raw.shape[1])
    if source_joint_names is None:
        raise ValueError(
            f"{path.name}: qpos width {raw.shape[1]}; expected one of "
            f"{sorted(_JOINT_NAMES_BY_NQ)}."
        )

    base_pos = raw[:, 0:3]
    base_quat_wxyz = raw[:, 3:7][:, [3, 0, 1, 2]]  # xyzw -> wxyz (see csv_to_npz)
    joint_angles = {
        name: raw[:, 7 + i] for i, name in enumerate(source_joint_names)
    }

    qpos = build_qpos(model, base_pos, base_quat_wxyz, joint_angles)

    if fps is None:
        fps = constants.SRB_KINO_DEFAULT_FPS

    return Trajectory(
        qpos=qpos,
        fps=float(fps),
        model=model_id,
        source=source,
        name=path.stem,
    )
