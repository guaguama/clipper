"""Loader for `mj-nlp` task-output npz files (source id ``mj_nlp_out``).

Reads the `.npz` a `../mj-nlp` example task writes into its own folder (e.g.
``examples/g1_tracking_mpc/tracking_<motion>.npz``) — the solved closed-loop MPC
result — so it can be visualized or re-cropped in clipper. This is a read-only
import; there is no matching writer.

Distinct from the ``mj_nlp`` source, which re-loads clipper's own qpos/time CSV
folder. This one parses mj-nlp's native task output, whose schema is (see
``../mj-nlp/src/file_utils.py`` ``save_trajectory``):
- ``state`` ``(N+1, nq+nv)`` float64 — full state per frame, ``[qpos | qvel]``.
- ``time`` ``(N+1,)`` — absolute time stamps (s); fps is its sample rate
  (``1 / median(diff(time))``, typically 100 Hz).
- also ``input``, ``model``, ``spline_type``, ``reference`` — unused here.

Only ``state`` is loaded (the solved rollout), not ``reference`` (the tracking
ghost). ``qvel`` is dropped: clipper's `Trajectory` stores qpos only and writers
derive velocities lazily.

The qpos columns are in the mj-nlp g1_29dof model order, identical to clipper's
``constants.G1_29DOF_JOINT_NAMES``, so they map onto the requested ``--model`` by
joint name via `build_qpos` (a 29-DOF clip can be viewed on g1_23dof and back).
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


def _fps_from_time(time: np.ndarray) -> float | None:
    """Frame rate from the time array as 1 / median sample period, or None."""
    t = np.atleast_1d(np.asarray(time, dtype=np.float64))
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
    source: str = "mj_nlp_out",
    fps: float | None = None,
) -> Trajectory:
    """Load an mj-nlp task-output npz into a `Trajectory` in `model`'s layout."""
    path = Path(path)
    data = np.load(path)
    if "state" not in data:
        raise ValueError(
            f"{path.name}: no 'state' array; not an mj-nlp task-output npz "
            f"(keys: {list(data.keys())})."
        )
    state = np.asarray(data["state"], dtype=np.float64)
    if state.ndim == 1:
        state = state[None, :]

    # state = [qpos | qvel]. For a free-base + hinge model nv = nq - 1, so the
    # total width W = 2*nq - 1 and nq = (W + 1) // 2.
    width = state.shape[1]
    nq = (width + 1) // 2
    if 2 * nq - 1 != width:
        raise ValueError(
            f"{path.name}: state width {width} is not a valid nq+nv for a "
            f"free-base model (expected 2*nq-1)."
        )
    source_joint_names = _JOINT_NAMES_BY_NQ.get(nq)
    if source_joint_names is None:
        raise ValueError(
            f"{path.name}: qpos width {nq} (from state width {width}); expected "
            f"one of {sorted(_JOINT_NAMES_BY_NQ)}."
        )

    qpos_raw = state[:, :nq]
    base_pos = qpos_raw[:, 0:3]
    base_quat_wxyz = qpos_raw[:, 3:7]  # already wxyz
    joint_angles = {
        name: qpos_raw[:, 7 + i] for i, name in enumerate(source_joint_names)
    }

    qpos = build_qpos(model, base_pos, base_quat_wxyz, joint_angles)

    if fps is None and "time" in data:
        fps = _fps_from_time(data["time"])
    if fps is None:
        raise ValueError(
            f"{path.name}: no usable 'time' array; pass --fps to set the rate."
        )

    return Trajectory(
        qpos=qpos,
        fps=float(fps),
        model=model_id,
        source=source,
        name=path.stem,
    )
