"""Write a `Trajectory` to the mjlab motion-tracking NPZ format.

Registered as the ``mjlab`` output format (matching the ``mjlab`` source id). This
is the same artifact the fork ``../unitree_rl_mjlab`` consumes — verified, not
assumed: the two repos' ``scripts/csv_to_npz.py`` differ only in which scene they
build and where they put the result (the fork takes a ``--robot`` argument and
saves locally; mjlab is G1-29dof-only and uploads to a W&B registry), and their
tracking ``MotionLoader``s read the identical six arrays. So one writer serves both,
and the format was renamed ``unilab`` -> ``mjlab`` after the upstream name.

Recreates the output of ``csv_to_npz.py`` using only MuJoCo + numpy (no
mjlab/torch/warp/CUDA), which also removes the intermediate retargeted CSV, the
converter run and the registry upload from the loop: clipper writes the final NPZ.
Per output frame:

* ``joint_pos`` / ``joint_vel`` ``(N, DOF)`` — the model's joints in qpos order;
* ``body_pos_w`` / ``body_quat_w`` ``(N, B, 3|4)`` — world body pose (quat wxyz);
* ``body_lin_vel_w`` / ``body_ang_vel_w`` ``(N, B, 3)`` — world body velocities;
* ``fps`` ``(1,)``.

There is no root key: the root is body index 0. Velocities are finite differences
(linear/joint via ``np.gradient``; angular via an SO3 central difference), matching
the upstream computation.

**Frame rate is load-bearing and nothing downstream checks it.** The tracking task's
``MotionLoader`` never reads the ``fps`` key — it advances one motion frame per
environment step (``time_steps += 1``), and both repos' tracking envs are
``timestep=0.005`` with ``decimation=4``, i.e. ``step_dt = 0.02 s``. A motion NPZ is
therefore *implicitly* 50 Hz, and one written at any other rate trains a policy
against a reference playing at the wrong speed, silently. So this writer resamples to
``constants.MJLAB_MOTION_FPS`` (50) by default — matching ``csv_to_npz``'s own
``--output-fps`` default — instead of inheriting the clip's rate. Pass an explicit
``--output-fps`` only for an env whose ``step_dt`` differs.

**Body block.** ``body_*_w`` must match the body set of the ``Entity`` the consuming
repo builds, in its order (``Entity.body_names`` is ``spec.bodies[1:]``, i.e. every
body but the world, in spec order). For the G1 that is the vendored *scene* XML's 30
bodies — verified against a real unitree_rl_mjlab artifact, and the ``g1_23dof``
scene deliberately pads 6 dummy bodies to reach the same 30; mjlab's own ``g1.xml``
was checked to carry the same 30 in the same order. Every other model has no scene variant and uses the model exactly
as clipper loads it, whose body list was verified to equal its robot XML's.

**Gait libraries are not written here.** sesteban's twist-command tasks loop each
clip and need a ``twist`` label, exact-period frame counts, seam-free velocities and
a shared phase. ``clipper.gait_library`` (``scripts/convert_gait_library.py``) builds
those and saves them through ``save_npz`` below, so the schema stays identical.
"""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np

from .. import constants, mathx
from ..model import joint_names, load_model
from ..qpos import build_qpos, joint_qpos_address
from ..trajectory import Trajectory


def _so3_derivative(quats: np.ndarray, dt: float) -> np.ndarray:
    """Angular velocity (world) from a quaternion sequence; shape (T, ..., 3).

    Central difference in SO3: ``omega[i] = axis_angle(q[i+1] * conj(q[i-1])) / (2 dt)``
    with the first/last samples repeated (matches upstream ``_so3_derivative``).
    """
    q_prev, q_next = quats[:-2], quats[2:]
    q_rel = mathx.quat_mul(q_next, mathx.quat_conj(q_prev))
    omega = mathx.axis_angle_from_quat(q_rel) / (2.0 * dt)
    return np.concatenate([omega[:1], omega, omega[-1:]], axis=0)


def _fk_model(model_id: str) -> mujoco.MjModel:
    """The model whose bodies define the NPZ's body axis (see the module docstring).

    G1 models resolve to their scene XML, which is what the upstream entity carries
    (30 bodies for both DOF counts, the 23-DOF one padded). Everything else — the MSK
    family — uses the model as clipper loads it, which is the same XML mjlab would
    vendor and whose body list matches it exactly.
    """
    if model_id in constants.SCENE_XML:
        return mujoco.MjModel.from_xml_path(str(constants.SCENE_XML[model_id]))
    return load_model(model_id)


def write(
    traj: Trajectory,
    out_dir: Path | None = None,
    output_fps: float | None = None,
    name: str | None = None,
    overwrite: bool = True,
) -> Path:
    """Write `traj` as an mjlab motion NPZ and return the output path.

    Args:
        traj: Trajectory in its model's qpos layout.
        out_dir: Output directory; defaults to ``<repo>/outputs/mjlab/<model>``.
        output_fps: Target fps; resample (lerp/slerp) when it differs from the
            trajectory's rate. ``None`` means ``constants.MJLAB_MOTION_FPS`` (50),
            mjlab's tracking step rate — not the clip's own rate.
        name: Output file stem; defaults to ``traj.name``.
        overwrite: If False, raise when the target file already exists.
    """

    model_id = traj.model
    names = joint_names(model_id)
    dof = len(names)

    # --- split the trajectory qpos into base + per-joint columns (by name) ---
    # The loaded model, not a bare XML compile: the MSK models only reach their
    # trajectory nq after load_model's finger-joint removal.
    model = load_model(model_id)
    if model.nq != 7 + dof:
        raise ValueError(f"{model_id}: expected nq=={7 + dof}, got {model.nq}.")
    if traj.qpos.shape[1] != model.nq:
        raise ValueError(
            f"{model_id}: expected qpos width {model.nq}, got {traj.qpos.shape[1]}."
        )
    base_pos = traj.qpos[:, 0:3].copy()
    base_quat = traj.qpos[:, 3:7].copy()
    dof_pos = np.stack(
        [traj.qpos[:, joint_qpos_address(model, jn)] for jn in names], axis=1
    )

    # --- resample to the target rate (lerp pos/joints, slerp base quat) ---
    target_fps = (
        constants.MJLAB_MOTION_FPS if output_fps is None else float(output_fps)
    )
    if target_fps != float(traj.fps):
        i0, i1, blend, _ = mathx.frame_blend(traj.num_frames, traj.fps, target_fps)
        base_pos = mathx.lerp(base_pos[i0], base_pos[i1], blend[:, None])
        dof_pos = mathx.lerp(dof_pos[i0], dof_pos[i1], blend[:, None])
        base_quat = mathx.slerp(base_quat[i0], base_quat[i1], blend)
    out_fps = target_fps

    n = base_pos.shape[0]
    if n < 3:
        raise ValueError(
            f"need >= 3 output frames for finite-difference velocities, got {n} "
            f"at {out_fps:g} Hz. Use a wider crop or --pad-standing."
        )
    dt = 1.0 / out_fps

    # --- joint outputs (real DOF only, in model order) ---
    joint_pos = dof_pos.astype(np.float32)
    joint_vel = np.gradient(dof_pos, dt, axis=0).astype(np.float32)

    # --- body kinematics via FK on the entity's body set ---
    fk_model = _fk_model(model_id)
    n_bodies = fk_model.nbody - 1  # every body but the world, matching Entity
    fk_qpos = build_qpos(
        fk_model,
        base_pos,
        base_quat,
        {jn: dof_pos[:, j] for j, jn in enumerate(names)},
    )
    data = mujoco.MjData(fk_model)
    body_pos_w = np.empty((n, n_bodies, 3), dtype=np.float64)
    body_quat_w = np.empty((n, n_bodies, 4), dtype=np.float64)
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
        out_dir = constants.REPO_ROOT / "outputs" / "mjlab" / model_id
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = name if name is not None else traj.name
    path = out_dir / f"{stem}.npz"
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} exists (pass overwrite=True to replace).")

    save_npz(
        path,
        out_fps,
        joint_pos=joint_pos,
        joint_vel=joint_vel,
        body_pos_w=body_pos_w,
        body_quat_w=body_quat_w,
        body_lin_vel_w=body_lin_vel_w,
        body_ang_vel_w=body_ang_vel_w,
    )
    return path


def save_npz(
    path: Path,
    fps: float,
    *,
    joint_pos: np.ndarray,
    joint_vel: np.ndarray,
    body_pos_w: np.ndarray,
    body_quat_w: np.ndarray,
    body_lin_vel_w: np.ndarray,
    body_ang_vel_w: np.ndarray,
    extra: dict[str, np.ndarray] | None = None,
) -> None:
    """Save the six mjlab motion arrays (float32) + ``fps`` + any ``extra`` keys.

    Shared with `clipper.gait_library`, which computes the same arrays its own way
    (periodic resample, solver velocities) but must emit the identical schema.
    """
    np.savez(
        path,
        fps=np.array([fps], dtype=np.float64),
        joint_pos=np.asarray(joint_pos, dtype=np.float32),
        joint_vel=np.asarray(joint_vel, dtype=np.float32),
        body_pos_w=np.asarray(body_pos_w, dtype=np.float32),
        body_quat_w=np.asarray(body_quat_w, dtype=np.float32),
        body_lin_vel_w=np.asarray(body_lin_vel_w, dtype=np.float32),
        body_ang_vel_w=np.asarray(body_ang_vel_w, dtype=np.float32),
        **(extra or {}),
    )
