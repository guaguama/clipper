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
- ``twist`` ``(3,)`` (gait-library nodes only) — the commanded ``[vx, vy, wz]`` label,
  kept on ``Trajectory.twist`` (not ``twist_sim``/``native_twist``, which are measured).
- also ``input``, ``model``, ``spline_type``, ``reference`` — unused here.

Only ``state`` is loaded (the solved rollout), not ``reference`` (the tracking
ghost). ``qvel`` is dropped: clipper's `Trajectory` stores qpos only and writers
derive velocities lazily.

**Every active joint coupling is re-solved** (`couple`). mj-nlp solves on a model with
the same equalities as clipper's (osl_fullbody: all 44, checked 2026-11-14), but they
are soft (solref 0.02), so a rollout's followers miss their polynomial: up to 0.058 rad
on the intact knee of the optimized TERRA stairs, 0.029 rad on unicycle_v001, 0.026 rad
on the shoulder girdle. mjlab's physics enforces those couplings, so a reference that
breaks them is one the policy cannot reach. Unlike an import (``import_equality_joints``),
these coordinates were not solved independently of the couplings, and re-solving moves
bodies by millimetres. ``equality_joints=()`` reads the rollout untouched.

The qpos columns are in the solving model's own order, which is identical to
clipper's for every shared model (the G1s), so they map onto the requested ``--model`` by
joint name via `build_qpos` (a 29-DOF clip can be viewed on g1_23dof and back).

Which joint order a file is in is inferred from its width, and — as in the ``mj_nlp``
source — **the target model's own nq is tried first**, because widths collide across
model families (the former ``osl_ka`` and ``g1_23dof`` were both nq 30). Consulting the G1 table
first would map an MSK rollout onto G1 joint names, none of which match, so
`build_qpos` would drop every column and silently yield an all-zero-joint clip. The G1
table applies only when the width is *not* the target model's own, which is what serves
the cross-dof case (a 29-DOF rollout viewed on g1_23dof).
"""

from __future__ import annotations

from collections.abc import Collection
from pathlib import Path

import mujoco
import numpy as np

from .. import constants
from ..model import joint_names
from ..qpos import build_qpos, enforce_joint_equalities, enforce_joint_equality_velocities
from ..trajectory import Trajectory

# qpos width (nq) -> source joint column order, for the cross-dof G1 case only.
# Same mapping (and same lookup order) as `mj_nlp.py`.
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


def couple(
    model: mujoco.MjModel,
    qpos: np.ndarray,
    qvel: np.ndarray | None = None,
    *,
    equality_joints: Collection[str] | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Re-solve a rollout's coupled followers: qpos exactly, qvel by the chain rule.

    ``equality_joints`` None means every active joint equality (this source's policy);
    an empty collection returns the inputs unchanged. `qvel`, when given, must be in
    `model`'s dof layout; its followers become ``p'(driver) * driver rate``.
    """
    q = enforce_joint_equalities(model, qpos, joint_names=equality_joints)
    if qvel is None:
        return q, None
    return q, enforce_joint_equality_velocities(model, q, qvel, joint_names=equality_joints)


def load(
    path: str | Path,
    model: mujoco.MjModel,
    model_id: str,
    source: str = "mj_nlp_out",
    fps: float | None = None,
    *,
    equality_joints: Collection[str] | None = None,
) -> Trajectory:
    """Load an mj-nlp task-output npz into a `Trajectory` in `model`'s layout.

    ``equality_joints``: see `couple` (None re-solves every active coupling).
    """
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
    # The target model's own layout wins (see the module docstring): an identity
    # read, and the only correct reading when a width is ambiguous across models.
    if nq == model.nq:
        source_joint_names = joint_names(model_id)
    else:
        source_joint_names = _JOINT_NAMES_BY_NQ.get(nq)
    if source_joint_names is None:
        # The npz records the XML the solve ran on; naming it turns a width mismatch
        # into an obvious "wrong --model" diagnosis.
        solved_on = str(data["model"]) if "model" in data else "unknown"
        raise ValueError(
            f"{path.name}: qpos width {nq} (from state width {width}); expected "
            f"{model.nq} for {model_id!r}, or one of {sorted(_JOINT_NAMES_BY_NQ)} "
            f"for a cross-model G1 rollout. The npz was solved on {solved_on!r}."
        )

    qpos_raw = state[:, :nq]
    base_pos = qpos_raw[:, 0:3]
    base_quat_wxyz = qpos_raw[:, 3:7]  # already wxyz
    joint_angles = {
        name: qpos_raw[:, 7 + i] for i, name in enumerate(source_joint_names)
    }

    qpos = build_qpos(model, base_pos, base_quat_wxyz, joint_angles)
    qpos, _ = couple(model, qpos, equality_joints=equality_joints)

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
        twist=data["twist"] if "twist" in data else None,
    )
