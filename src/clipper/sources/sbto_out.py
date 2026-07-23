"""Loader for sbto solver output (source id ``sbto_out``).

Reads the ``best_trajectory.npz`` the ``../sbto`` DynaRetarget/SBTO solver writes
into each run directory (e.g.
``sbto/outputs/<timestamp>__<motion>/best_trajectory.npz``) — the optimized,
dynamically-feasible trajectory — so it can be visualized or re-cropped in clipper.
This is a read-only import; there is no matching writer.

Distinct from the ``sbto`` source, which re-loads clipper's own sbto *reference*
input NPZ. This one parses sbto's native output, whose schema is (verified from a
solver run; see also ``../sbto/sbto/data/constants.py``):
- ``root_pos`` ``(N, 3)`` — base position (m).
- ``root_rot`` ``(N, 4)`` — base orientation, MuJoCo **wxyz**.
- ``dof_pos`` ``(N, ndof)`` — joint angles (rad), in the G1 29-DOF model order
  (``constants.G1_29DOF_JOINT_NAMES``).
- ``time`` ``(N,)`` — absolute time stamps (s); fps is its sample rate
  (``1 / median(diff(time))``, 100 Hz at sbto's 0.01 s sim step).
- also ``dof_vel/root_lin_vel/root_ang_vel/dof_pd_target/cost/step_knots`` — unused
  (clipper's `Trajectory` stores qpos only; writers derive velocities lazily).

Unlike the ``sbto`` reference NPZ, this output is already canonical MuJoCo
``[pos, quat_wxyz]`` (sbto un-flips the reference on load), so no base reorder is
needed. Joints map onto the requested ``--model`` by name via `build_qpos`.
"""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np

from .. import constants
from ..qpos import build_qpos
from ..trajectory import Trajectory

# sbto writes the optimized trajectory here inside each run directory.
_BEST_TRAJECTORY_NPZ = "best_trajectory.npz"

# joint count (ndof) -> source joint column order. sbto ships only the g1_29dof
# no-hands model; the 23-DOF entry is accepted for future-proofing.
_JOINT_NAMES_BY_NDOF: dict[int, tuple[str, ...]] = {
    len(constants.G1_23DOF_JOINT_NAMES): constants.G1_23DOF_JOINT_NAMES,
    len(constants.G1_29DOF_JOINT_NAMES): constants.G1_29DOF_JOINT_NAMES,
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
    source: str = "sbto_out",
    fps: float | None = None,
) -> Trajectory:
    """Load an sbto solver-output NPZ into a `Trajectory` in `model`'s layout."""
    path = Path(path)
    # Accept either a run directory or the npz itself.
    npz = path / _BEST_TRAJECTORY_NPZ if path.is_dir() else path
    data = np.load(npz)

    missing = [k for k in ("root_pos", "root_rot", "dof_pos") if k not in data]
    if missing:
        raise ValueError(
            f"{npz.name}: missing {missing}; not an sbto best_trajectory NPZ "
            f"(keys: {list(data.keys())})."
        )

    base_pos = np.asarray(data["root_pos"], dtype=np.float64)
    base_quat_wxyz = np.asarray(data["root_rot"], dtype=np.float64)  # already wxyz
    dof = np.asarray(data["dof_pos"], dtype=np.float64)
    if dof.ndim == 1:
        dof = dof[None, :]

    source_joint_names = _JOINT_NAMES_BY_NDOF.get(dof.shape[1])
    if source_joint_names is None:
        raise ValueError(
            f"{npz.name}: dof_pos width {dof.shape[1]}; expected one of "
            f"{sorted(_JOINT_NAMES_BY_NDOF)}."
        )
    joint_angles = {
        name: dof[:, i] for i, name in enumerate(source_joint_names)
    }

    qpos = build_qpos(model, base_pos, base_quat_wxyz, joint_angles)

    if fps is None and "time" in data:
        fps = _fps_from_time(data["time"])
    if fps is None:
        raise ValueError(
            f"{npz.name}: no usable 'time' array; pass --fps to set the rate."
        )

    # Prefer the descriptive run-directory name over the generic "best_trajectory".
    name = npz.parent.name if npz.stem == "best_trajectory" else npz.stem

    return Trajectory(
        qpos=qpos,
        fps=float(fps),
        model=model_id,
        source=source,
        name=name,
    )
