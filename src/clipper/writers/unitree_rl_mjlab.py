"""Write a `Trajectory` to the `unitree_rl_mjlab` motion-tracking NPZ format.

Recreates the output of upstream ``unitree_rl_mjlab/scripts/csv_to_npz.py`` using
only MuJoCo + numpy (no torch/mjlab/CUDA). The NPZ holds, per output frame:

* ``joint_pos`` / ``joint_vel`` ``(N, DOF)`` — the actuated joints in model order;
* ``body_pos_w`` / ``body_quat_w`` ``(N, 30, 3|4)`` — world body pose (wxyz);
* ``body_lin_vel_w`` / ``body_ang_vel_w`` ``(N, 30, 3)`` — world body velocities;
* ``fps`` ``(1,)``.

There is no root key: the root is body index 0. Body kinematics come from MuJoCo
forward kinematics on the model's *scene* XML, which has the full 30-body set for
both g1_29dof and g1_23dof (the 23-DOF scene pads with 6 dummy bodies). Velocities
are finite differences (linear/joint via ``np.gradient``; angular via an SO3
central difference), matching the upstream computation.
"""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np

from .. import constants, mathx
from ..qpos import build_qpos, joint_qpos_address
from ..trajectory import Trajectory

# Robot bodies = all compiled bodies minus the world body (index 0).
N_BODIES = 30


def _so3_derivative(quats: np.ndarray, dt: float) -> np.ndarray:
    """Angular velocity (world) from a quaternion sequence; shape (T, ..., 3).

    Central difference in SO3: ``omega[i] = axis_angle(q[i+1] * conj(q[i-1])) / (2 dt)``
    with the first/last samples repeated (matches upstream ``_so3_derivative``).
    """
    q_prev, q_next = quats[:-2], quats[2:]
    q_rel = mathx.quat_mul(q_next, mathx.quat_conj(q_prev))
    omega = mathx.axis_angle_from_quat(q_rel) / (2.0 * dt)
    return np.concatenate([omega[:1], omega, omega[-1:]], axis=0)


def write(
    traj: Trajectory,
    out_dir: Path | None = None,
    output_fps: float | None = None,
    name: str | None = None,
    overwrite: bool = True,
) -> Path:
    """Write `traj` as a unitree_rl_mjlab motion NPZ and return the output path.

    Args:
        traj: Trajectory in its model's qpos layout.
        out_dir: Output directory; defaults to
            ``<repo>/outputs/unitree_rl_mjlab/<model>``.
        output_fps: Target fps; resample (lerp/slerp) only if set and != traj.fps.
            ``None`` keeps the trajectory's own fps with no resampling.
        name: Output file stem; defaults to ``traj.name``.
        overwrite: If False, raise when the target file already exists.
    """
    model_id = traj.model
    joint_names = constants.JOINT_NAMES[model_id]
    dof = len(joint_names)

    # --- split the trajectory qpos into base + per-joint columns (by name) ---
    bare = mujoco.MjModel.from_xml_path(str(constants.ROBOT_XML[model_id]))
    if bare.nq != 7 + dof:
        raise ValueError(
            f"{model_id}: expected bare nq=={7 + dof}, got {bare.nq}."
        )
    base_pos = traj.qpos[:, 0:3].copy()
    base_quat = traj.qpos[:, 3:7].copy()
    dof_pos = np.stack(
        [traj.qpos[:, joint_qpos_address(bare, jn)] for jn in joint_names], axis=1
    )

    # --- optional resample to output_fps (lerp pos/joints, slerp base quat) ---
    if output_fps is not None and float(output_fps) != float(traj.fps):
        i0, i1, blend, n_out = mathx.frame_blend(traj.num_frames, traj.fps, output_fps)
        base_pos = mathx.lerp(base_pos[i0], base_pos[i1], blend[:, None])
        dof_pos = mathx.lerp(dof_pos[i0], dof_pos[i1], blend[:, None])
        base_quat = mathx.slerp(base_quat[i0], base_quat[i1], blend)
        out_fps = float(output_fps)
    else:
        out_fps = float(traj.fps)

    n = base_pos.shape[0]
    if n < 3:
        raise ValueError(
            f"need >= 3 output frames for finite-difference velocities, got {n}. "
            "Use a wider crop or --pad-standing."
        )
    dt = 1.0 / out_fps

    # --- joint outputs (real DOF only, in model order) ---
    joint_pos = dof_pos.astype(np.float32)
    joint_vel = np.gradient(dof_pos, dt, axis=0).astype(np.float32)

    # --- body kinematics via FK on the scene model (30-body parity) ---
    fk_model = mujoco.MjModel.from_xml_path(str(constants.SCENE_XML[model_id]))
    if fk_model.nbody != N_BODIES + 1:
        raise ValueError(
            f"{model_id}: scene model has {fk_model.nbody} bodies, "
            f"expected {N_BODIES + 1} (world + {N_BODIES})."
        )
    fk_qpos = build_qpos(
        fk_model,
        base_pos,
        base_quat,
        {jn: dof_pos[:, j] for j, jn in enumerate(joint_names)},
    )
    data = mujoco.MjData(fk_model)
    body_pos_w = np.empty((n, N_BODIES, 3), dtype=np.float64)
    body_quat_w = np.empty((n, N_BODIES, 4), dtype=np.float64)
    for k in range(n):
        data.qpos[:] = fk_qpos[k]
        mujoco.mj_forward(fk_model, data)
        body_pos_w[k] = data.xpos[1:]
        body_quat_w[k] = data.xquat[1:]

    if not np.allclose(body_pos_w[:, 0], base_pos, atol=1e-6):
        raise AssertionError("root body (index 0) does not match base position.")

    # --- body velocities (world frame, finite difference) ---
    body_lin_vel_w = np.gradient(body_pos_w, dt, axis=0)
    body_ang_vel_w = _so3_derivative(body_quat_w, dt)

    # --- save ---
    if out_dir is None:
        out_dir = constants.REPO_ROOT / "outputs" / "unitree_rl_mjlab" / model_id
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = name if name is not None else traj.name
    path = out_dir / f"{stem}.npz"
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} exists (pass overwrite=True to replace).")

    np.savez(
        path,
        fps=np.array([out_fps], dtype=np.float64),
        joint_pos=joint_pos,
        joint_vel=joint_vel,
        body_pos_w=body_pos_w.astype(np.float32),
        body_quat_w=body_quat_w.astype(np.float32),
        body_lin_vel_w=body_lin_vel_w.astype(np.float32),
        body_ang_vel_w=body_ang_vel_w.astype(np.float32),
    )
    return path
