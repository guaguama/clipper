"""Write a `Trajectory` to the musclemimic cache NPZ format (round-trip).

Reproduces the schema musclemimic reads at train/runtime (and that its GMR
retargeter writes), so a cropped/edited clip is directly consumable again. Mirrors
``musclemimic/scripts/project_gmr_cache_to_osl_ka.py``:

* ``qpos`` (N, nq) / ``qvel`` (N, nv) — qvel by per-step ``mj_differentiatePos``;
* ``xpos/xquat/cvel/subtree_com`` (N, nbody, ...) — body FK from ``mj_forward``;
* ``site_xpos`` (N, nsite, 3) / ``site_xmat`` (N, nsite, 9) — FK at the ``*_mimic``
  tracking sites only (injected by ``model.load_model``);
* static model metadata (``njnt, jnt_type, joint_names, nbody, body_*, site_*``);
* ``frequency``, ``split_points`` ``[0, N]``, ``metadata``.

Velocities/FK are derived here (never carried in clipper's canonical Trajectory),
matching the rest of the writers. Quaternions are MuJoCo ``wxyz`` throughout.
"""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np

from .. import constants, mathx
from ..model import load_model
from ..trajectory import Trajectory


def _resample_qpos(qpos: np.ndarray, in_fps: float, out_fps: float) -> np.ndarray:
    """Resample a full qpos array: lerp positions/joints, slerp the base quat."""
    i0, i1, blend, _ = mathx.frame_blend(qpos.shape[0], in_fps, out_fps)
    out = mathx.lerp(qpos[i0], qpos[i1], blend[:, None])
    out[:, 3:7] = mathx.slerp(qpos[i0, 3:7], qpos[i1, 3:7], blend)
    return out


def _mimic_site_ids(model: mujoco.MjModel, model_id: str) -> tuple[list[int], list[str]]:
    """Site ids + names for the model's mimic sites, in MSK_MIMIC_SITES dict order."""
    ids: list[int] = []
    names: list[str] = []
    for site_name in constants.MSK_MIMIC_SITES.get(model_id, {}).values():
        sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
        if sid >= 0:
            ids.append(sid)
            names.append(site_name)
    return ids, names


def _qvel_from_qpos(model: mujoco.MjModel, qpos: np.ndarray, dt: float) -> np.ndarray:
    """Finite-difference qvel via ``mj_differentiatePos`` (last frame duplicated)."""
    n = qpos.shape[0]
    qvel = np.zeros((n, model.nv), dtype=np.float64)
    for t in range(n - 1):
        mujoco.mj_differentiatePos(model, qvel[t], dt, qpos[t], qpos[t + 1])
    if n >= 2:
        qvel[-1] = qvel[-2]
    return qvel


def _static_model_metadata(model: mujoco.MjModel, site_ids, site_names) -> dict:
    """Static model arrays copied from the compiled model (matches the generator)."""
    joint_names = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(model.njnt)
    ]
    body_names = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) for i in range(model.nbody)
    ]
    site_ids = np.asarray(site_ids, dtype=np.int64)
    return dict(
        njnt=np.int32(model.njnt),
        jnt_type=np.asarray(model.jnt_type[: model.njnt], dtype=np.int32),
        joint_names=np.asarray(joint_names),
        nbody=np.int32(model.nbody),
        body_names=np.asarray(body_names),
        body_rootid=np.asarray(model.body_rootid[: model.nbody], dtype=np.int32),
        body_weldid=np.asarray(model.body_weldid[: model.nbody], dtype=np.int32),
        body_mocapid=np.asarray(model.body_mocapid[: model.nbody], dtype=np.int32),
        body_pos=np.asarray(model.body_pos[: model.nbody], dtype=np.float32),
        body_quat=np.asarray(model.body_quat[: model.nbody], dtype=np.float32),
        body_ipos=np.asarray(model.body_ipos[: model.nbody], dtype=np.float32),
        body_iquat=np.asarray(model.body_iquat[: model.nbody], dtype=np.float32),
        nsite=np.int32(len(site_names)),
        site_names=np.asarray(site_names),
        site_bodyid=np.asarray(model.site_bodyid[site_ids], dtype=np.int32)
        if len(site_ids)
        else np.zeros((0,), dtype=np.int32),
        site_pos=np.asarray(model.site_pos[site_ids], dtype=np.float32)
        if len(site_ids)
        else np.zeros((0, 3), dtype=np.float32),
        site_quat=np.asarray(model.site_quat[site_ids], dtype=np.float32)
        if len(site_ids)
        else np.zeros((0, 4), dtype=np.float32),
    )


def write(
    traj: Trajectory,
    out_dir: Path | None = None,
    output_fps: float | None = None,
    name: str | None = None,
    overwrite: bool = True,
) -> Path:
    """Write `traj` as a musclemimic cache NPZ and return the output path.

    Args:
        traj: Trajectory in its MSK model's qpos layout.
        out_dir: Output directory; defaults to ``<repo>/outputs/musclemimic/<model>``.
        output_fps: Target fps; resample (lerp/slerp) only if set and != traj.fps.
        name: Output file stem; defaults to ``traj.name``.
        overwrite: If False, raise when the target file already exists.
    """
    model_id = traj.model
    if model_id not in constants.MSK_MODELS:
        raise ValueError(
            f"musclemimic writer targets MSK models {sorted(constants.MSK_MODELS)}; "
            f"got {model_id!r}."
        )

    model = load_model(model_id)
    qpos = traj.qpos
    if qpos.shape[1] != model.nq:
        raise ValueError(
            f"{model_id}: trajectory qpos width {qpos.shape[1]} != model nq {model.nq}."
        )

    # --- optional resample ---
    if output_fps is not None and float(output_fps) != float(traj.fps):
        qpos = _resample_qpos(qpos, traj.fps, float(output_fps))
        out_fps = float(output_fps)
    else:
        out_fps = float(traj.fps)

    n = qpos.shape[0]
    if n < 2:
        raise ValueError(f"need >= 2 frames for velocities, got {n}.")
    dt = 1.0 / out_fps

    qvel = _qvel_from_qpos(model, qpos, dt)

    # --- per-frame FK: body kinematics (all bodies) + mimic-site kinematics ---
    site_ids, site_names = _mimic_site_ids(model, model_id)
    nmimic = len(site_ids)
    data = mujoco.MjData(model)
    xpos = np.empty((n, model.nbody, 3), dtype=np.float32)
    xquat = np.empty((n, model.nbody, 4), dtype=np.float32)
    cvel = np.empty((n, model.nbody, 6), dtype=np.float32)
    subtree_com = np.empty((n, model.nbody, 3), dtype=np.float32)
    site_xpos = np.empty((n, nmimic, 3), dtype=np.float32)
    site_xmat = np.empty((n, nmimic, 9), dtype=np.float32)
    for t in range(n):
        data.qpos[:] = qpos[t]
        data.qvel[:] = qvel[t]
        mujoco.mj_forward(model, data)
        xpos[t] = data.xpos
        xquat[t] = data.xquat
        cvel[t] = data.cvel
        subtree_com[t] = data.subtree_com
        if nmimic:
            site_xpos[t] = data.site_xpos[site_ids]
            site_xmat[t] = data.site_xmat[site_ids]

    static = _static_model_metadata(model, site_ids, site_names)
    metadata = {"projected_from": traj.source, "projection_script": "clipper"}

    # --- save ---
    if out_dir is None:
        out_dir = constants.REPO_ROOT / "outputs" / "musclemimic" / model_id
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = name if name is not None else traj.name
    path = out_dir / f"{stem}.npz"
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} exists (pass overwrite=True to replace).")

    np.savez_compressed(
        path,
        qpos=qpos.astype(np.float32),
        qvel=qvel.astype(np.float32),
        frequency=np.float32(out_fps),
        split_points=np.array([0, n], dtype=np.int64),
        metadata=np.array(metadata, dtype=object),
        xpos=xpos,
        xquat=xquat,
        cvel=cvel,
        subtree_com=subtree_com,
        site_xpos=site_xpos,
        site_xmat=site_xmat,
        **static,
    )
    return path
