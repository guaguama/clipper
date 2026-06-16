"""Loader for `mj-nlp` (sim-nlp) trajectory folders (source id ``mj_nlp``).

Re-loads the qpos/time CSV pair written by ``clipper.writers.mj_nlp`` (and used
by ``../mj-nlp/trajectories/playback.py``), so an mj-nlp clip can be visualized
or re-cropped — the loader/writer pair is symmetric.

Layout (see ``writers/mj_nlp.py``):
- ``qpos_<dof>dof.csv`` ``(N, nq)`` — full qpos per row, no header:
  ``base_pos(3) + base_quat(4, wxyz) + joints`` in ``g1_<dof>dof`` model order.
- ``time.csv`` ``(N,)`` — absolute time stamps (s); the frame rate is its sample
  period (``1 / median(diff(time))``), matching playback's own timing logic.

``--path`` may be the trajectory folder (the ``qpos_<dof>dof.csv`` matching the
target ``--model`` is preferred, else whichever variant is present) or a specific
``qpos_*dof.csv`` file. The qpos columns are in the model's qpos order (identical
between clipper and mj-nlp), so they map onto the requested model by joint name
via `build_qpos` — a 29-DOF clip can be viewed on g1_23dof and vice-versa.
"""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np

from .. import constants
from ..qpos import build_qpos
from ..trajectory import Trajectory

# qpos width (nq) -> source joint column order.
_JOINT_NAMES_BY_NQ: dict[int, tuple[str, ...]] = {
    7 + len(constants.G1_23DOF_JOINT_NAMES): constants.G1_23DOF_JOINT_NAMES,
    7 + len(constants.G1_29DOF_JOINT_NAMES): constants.G1_29DOF_JOINT_NAMES,
}


def _resolve_qpos_csv(path: Path, model_id: str) -> Path:
    """Pick the qpos CSV: a folder's dof-matched variant, or `path` itself."""
    if path.is_dir():
        dof = len(constants.JOINT_NAMES[model_id])
        preferred = path / f"qpos_{dof}dof.csv"
        if preferred.exists():
            return preferred
        candidates = sorted(path.glob("qpos_*dof.csv"))
        if not candidates:
            raise FileNotFoundError(f"no qpos_*dof.csv in {path}")
        return candidates[0]
    return path


def _fps_from_time_csv(time_path: Path) -> float | None:
    """Frame rate from time.csv as 1 / median sample period, or None if absent."""
    if not time_path.exists():
        return None
    t = np.atleast_1d(np.loadtxt(time_path, delimiter=","))
    if t.size < 2:
        return None
    dt = float(np.median(np.diff(t)))
    if dt <= 0.0:
        return None
    return 1.0 / dt


def load(
    path: str | Path,
    model: mujoco.MjModel,
    model_id: str,
    source: str = "mj_nlp",
    fps: float | None = None,
) -> Trajectory:
    """Load an mj-nlp trajectory into a `Trajectory` in `model`'s qpos layout."""
    path = Path(path)
    qpos_csv = _resolve_qpos_csv(path, model_id)

    raw = np.loadtxt(qpos_csv, delimiter=",", dtype=np.float64)
    if raw.ndim == 1:
        raw = raw[None, :]
    source_joint_names = _JOINT_NAMES_BY_NQ.get(raw.shape[1])
    if source_joint_names is None:
        raise ValueError(
            f"{qpos_csv.name}: qpos width {raw.shape[1]}; expected one of "
            f"{sorted(_JOINT_NAMES_BY_NQ)}."
        )

    base_pos = raw[:, 0:3]
    base_quat_wxyz = raw[:, 3:7]  # already wxyz
    joint_angles = {
        name: raw[:, 7 + i] for i, name in enumerate(source_joint_names)
    }

    qpos = build_qpos(model, base_pos, base_quat_wxyz, joint_angles)

    if fps is None:
        # time.csv sits beside the qpos CSV (or in the folder that was passed).
        fps = _fps_from_time_csv(qpos_csv.parent / "time.csv")
    if fps is None:
        raise ValueError(
            f"{qpos_csv.parent}: no usable time.csv; pass --fps to set the rate."
        )

    return Trajectory(
        qpos=qpos,
        fps=float(fps),
        model=model_id,
        source=source,
        name=qpos_csv.parent.name if path.is_dir() else qpos_csv.stem,
    )
