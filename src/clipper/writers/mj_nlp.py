"""Write a `Trajectory` to the `mj-nlp` (sim-nlp) trajectory CSV format.

The downstream ``../mj-nlp`` repo plays reference clips from a per-trajectory
folder holding two plain MuJoCo-standard CSVs (see ``mj-nlp/trajectories/playback.py``):

* ``qpos_<dof>dof.csv`` ``(N, nq)`` — one full qpos row per frame, no header:
  ``base_pos(3) + base_quat(4, wxyz) + joints`` in the target model's own order;
* ``time.csv`` ``(N,)`` — absolute time stamp (s) for each qpos row.
* Optional ``terrain.json`` — static world-frame boxes and provenance in TERRA's
  TerrainSpec schema, with an implicit floor at z=0. Absent for legacy motion-only
  clips; an empty boxes list explicitly declares flat terrain.

clipper's ``Trajectory.qpos`` is already in the target model's qpos layout, and the
qpos ordering is identical between clipper and mj-nlp for every shared model, so the
array is written out directly with no re-ordering. Verified for the G1 (same joint
order/types, only a ``robot/`` namespace prefix differs). The base quaternion is already
MuJoCo wxyz, matching playback's assumption.

``dof`` comes from ``model.joint_names`` rather than the G1-only
``constants.JOINT_NAMES``, so this writer serves the MSK models too.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .. import constants, mathx
from ..model import joint_names
from ..trajectory import Trajectory


def write(
    traj: Trajectory,
    out_dir: Path | None = None,
    output_fps: float | None = None,
    name: str | None = None,
    overwrite: bool = True,
) -> Path:
    """Write qpos/time CSVs and optional terrain; return the clip directory.

    Args:
        traj: Trajectory in its model's qpos layout.
        out_dir: Parent directory; defaults to
            ``<repo>/outputs/mj_nlp/<model>``. The clip is written to a
            ``<stem>/`` subfolder inside it.
        output_fps: Target fps; resample (lerp/slerp) only if set and != traj.fps.
            ``None`` keeps the trajectory's own fps with no resampling.
        name: Output folder stem; defaults to ``traj.name``.
        overwrite: If False, raise when a target CSV or terrain sidecar exists.
            If True, replace the motion and terrain together, removing an old
            terrain sidecar when the new trajectory has no terrain.
    """
    model_id = traj.model
    dof = len(joint_names(model_id))

    qpos = traj.qpos
    if qpos.shape[1] != 7 + dof:
        raise ValueError(
            f"{model_id}: expected qpos width {7 + dof}, got {qpos.shape[1]}."
        )

    # --- optional resample to output_fps (lerp base pos/joints, slerp base quat) ---
    if output_fps is not None and float(output_fps) != float(traj.fps):
        i0, i1, blend, _ = mathx.frame_blend(traj.num_frames, traj.fps, output_fps)
        base_pos = mathx.lerp(qpos[i0, 0:3], qpos[i1, 0:3], blend[:, None])
        base_quat = mathx.slerp(qpos[i0, 3:7], qpos[i1, 3:7], blend)
        joints = mathx.lerp(qpos[i0, 7:], qpos[i1, 7:], blend[:, None])
        qpos = np.concatenate([base_pos, base_quat, joints], axis=1)
        out_fps = float(output_fps)
    else:
        out_fps = float(traj.fps)

    n = qpos.shape[0]
    times = np.arange(n, dtype=np.float64) / out_fps

    # --- resolve the per-clip output directory ---
    if out_dir is None:
        out_dir = constants.REPO_ROOT / "outputs" / "mj_nlp" / model_id
    stem = name if name is not None else traj.name
    clip_dir = Path(out_dir) / stem
    qpos_path = clip_dir / f"qpos_{dof}dof.csv"
    time_path = clip_dir / "time.csv"
    terrain_path = clip_dir / "terrain.json"
    if not overwrite and any(p.exists() for p in (qpos_path, time_path, terrain_path)):
        raise FileExistsError(
            f"{clip_dir} already has trajectory files (pass overwrite=True to replace)."
        )
    clip_dir.mkdir(parents=True, exist_ok=True)

    np.savetxt(qpos_path, qpos, delimiter=",")
    np.savetxt(time_path, times, delimiter=",")
    if traj.terrain is not None:
        traj.terrain.save(terrain_path)
    else:
        # Never attach the previous clip's obstacles to replacement motion.
        terrain_path.unlink(missing_ok=True)
    return clip_dir
