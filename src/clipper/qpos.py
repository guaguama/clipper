"""Build a model-ordered qpos from named joint angles.

This is the single mechanism that decouples source joint ordering from the
target model's qpos layout and makes *any source -> any model* work:
joints are matched to the model by name, scattered to their qpos addresses, and
joints absent from the model are simply skipped (a 23-joint source on g1_29dof
leaves the 6 extra DOFs at their default 0; a 29-joint source on g1_23dof drops
the 6 it doesn't have).
"""

from __future__ import annotations

import mujoco
import numpy as np


def joint_qpos_address(model: mujoco.MjModel, joint_name: str) -> int | None:
    """Return the qpos index of a (1-DOF hinge/slide) joint, or None if absent."""
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    if jid < 0:
        return None
    return int(model.jnt_qposadr[jid])


def build_qpos(
    model: mujoco.MjModel,
    base_pos: np.ndarray,
    base_quat_wxyz: np.ndarray,
    joint_angles: dict[str, np.ndarray],
) -> np.ndarray:
    """Assemble a (N, nq) qpos array in `model`'s qpos layout.

    Args:
        model: Compiled MuJoCo model (a single free-joint floating base + hinges).
        base_pos: (N, 3) base position (meters).
        base_quat_wxyz: (N, 4) base orientation, MuJoCo wxyz order.
        joint_angles: mapping joint_name -> (N,) angles in radians. Names not
            present in `model` are ignored; model joints not provided stay 0.

    Returns:
        (N, model.nq) float64 array. The free joint occupies qpos[:, :7]
        (pos then wxyz quat); remaining entries default to 0.
    """
    base_pos = np.asarray(base_pos, dtype=np.float64)
    base_quat_wxyz = np.asarray(base_quat_wxyz, dtype=np.float64)
    n = base_pos.shape[0]
    if base_pos.shape != (n, 3):
        raise ValueError(f"base_pos must be (N, 3); got {base_pos.shape}")
    if base_quat_wxyz.shape != (n, 4):
        raise ValueError(f"base_quat_wxyz must be (N, 4); got {base_quat_wxyz.shape}")

    qpos = np.zeros((n, model.nq), dtype=np.float64)
    qpos[:, 0:3] = base_pos
    qpos[:, 3:7] = base_quat_wxyz

    for name, angles in joint_angles.items():
        adr = joint_qpos_address(model, name)
        if adr is None:
            continue  # joint not in this model — skip (cross-model fill/drop)
        qpos[:, adr] = np.asarray(angles, dtype=np.float64)

    return qpos


def enforce_joint_equalities(model: mujoco.MjModel, qpos: np.ndarray) -> np.ndarray:
    """Return a copy of `qpos` with every active joint coupling solved exactly.

    MSK knees carry coupler joints (``knee_angle_rotation3_l`` etc.) tied to the
    knee angle by joint equalities, ``q1 - q1_ref = sum_k c_k (q2 - q2_ref)^k`` with
    the refs from ``qpos0`` (``q2`` term absent when the equality has no second
    joint). A clip retargeted on a different model carries *that* model's coupler
    values: the ``MyoFullBody`` cache predates myo_sim 0.2.3, whose
    ``knee_angle_rotation3_l`` polycoef has the opposite sign of the one in
    ``osl_fullbody``, so copied values miss by up to 0.47 rad. Recomputing the
    couplers from the target model's own polynomials fixes that, and fills couplers
    the source never had (``myolegs``' right knee from the OSL_KA cache).
    """
    qpos = np.array(qpos, dtype=np.float64)
    eqs = [
        e for e in range(model.neq)
        if model.eq_type[e] == mujoco.mjtEq.mjEQ_JOINT and model.eq_active0[e]
    ]
    driven = {int(model.eq_obj1id[e]) for e in eqs}
    for e in eqs:
        j1, j2 = int(model.eq_obj1id[e]), int(model.eq_obj2id[e])
        if j2 in driven:  # a single pass is only exact when no driver is itself driven
            raise ValueError(
                f"chained joint equality: {model.joint(j2).name} drives "
                f"{model.joint(j1).name} but is itself driven."
            )
        a1 = model.jnt_qposadr[j1]
        c = model.eq_data[e, :5]
        if j2 < 0:
            qpos[:, a1] = model.qpos0[a1] + c[0]
            continue
        a2 = model.jnt_qposadr[j2]
        x = qpos[:, a2] - model.qpos0[a2]
        qpos[:, a1] = model.qpos0[a1] + c[0] + x * (c[1] + x * (c[2] + x * (c[3] + x * c[4])))
    return qpos
