"""Write a `Trajectory` to the `sbto` (DynaRetarget / Sampling-Based Trajectory
Optimization) reference-motion NPZ format.

The downstream ``../sbto`` repo loads a reference clip from a single ``.npz``
(``sbto/utils/extract_ref.py::load_npz_reference``) and reads exactly two keys:

* ``qpos`` ``(T, Nq)`` ``float64`` — one full MuJoCo qpos row per frame;
* ``fps`` — scalar frame rate, cast with ``int(...)``.

Everything else sbto needs (velocities, time, forward kinematics) is derived
internally, so positions are the only input. Any other array in the file is
ignored.

sbto targets the **Unitree G1 29-DoF, no-hands** model
(``sbto/models/unitree_g1/g1_mjx_no_hands.xml``), whose joint order is identical
to clipper's ``constants.G1_29DOF_JOINT_NAMES``. We therefore always emit a
36-column ``g1_29dof``-ordered qpos (base 7 + 29 joints); a ``g1_23dof`` clip is
name-remapped into that layout with its 6 absent joints left at 0.

**Free-joint layout:** sbto's default config (``task.cfg_ref``) uses
``flip_quat_pos=True`` and ``quat_wxyz=True`` — i.e. it expects the base free
joint stored **OmniRetarget-native** ``[quat(wxyz), pos]`` and flips it to
MuJoCo's ``[pos, quat]`` on load. This writer emits that native layout so clips
run through sbto with **no extra CLI flags**. (Quaternion elements stay wxyz;
only the pos/quat block order is swapped relative to clipper's canonical qpos.)

Robot-only clips (clipper never emits an object free joint) → run sbto with the
``task=g1/robot_ref`` task, e.g.::

    python3 sbto/main.py task=g1/robot_ref \\
        task.cfg_ref.motion_path=<this_file>.npz
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np

from .. import constants, mathx
from ..trajectory import Trajectory

# Models sbto can accept: only the G1 (its sole shipped model is g1_29dof-no-hands).
_SUPPORTED_MODELS = ("g1_29dof", "g1_23dof")


def _to_g1_29dof_qpos(traj: Trajectory) -> np.ndarray:
    """Return `traj.qpos` in the 36-col g1_29dof layout `[base(7), 29 joints]`.

    ``g1_29dof`` is used verbatim; ``g1_23dof`` is remapped by joint name into the
    29-DoF ordering (joints absent from the 23-DoF model — ``waist_roll``,
    ``waist_pitch``, and the L/R ``wrist_pitch``/``wrist_yaw`` — stay 0).
    """
    model_id = traj.model
    if model_id not in _SUPPORTED_MODELS:
        raise ValueError(
            f"sbto writer only supports G1 models {_SUPPORTED_MODELS} "
            f"(sbto ships a single G1 29-DoF model); got {model_id!r}."
        )

    qpos = traj.qpos
    src_names = constants.JOINT_NAMES[model_id]
    if qpos.shape[1] != 7 + len(src_names):
        raise ValueError(
            f"{model_id}: expected qpos width {7 + len(src_names)}, "
            f"got {qpos.shape[1]}."
        )

    if model_id == "g1_29dof":
        return qpos

    # g1_23dof -> g1_29dof: gather by joint name, zero-fill the 6 missing joints.
    src_col = {name: 7 + i for i, name in enumerate(src_names)}
    n = qpos.shape[0]
    joints = np.zeros((n, len(constants.G1_29DOF_JOINT_NAMES)), dtype=np.float64)
    for i, name in enumerate(constants.G1_29DOF_JOINT_NAMES):
        col = src_col.get(name)
        if col is not None:
            joints[:, i] = qpos[:, col]
    return np.concatenate([qpos[:, :7], joints], axis=1)


def write(
    traj: Trajectory,
    out_dir: Path | None = None,
    output_fps: float | None = None,
    name: str | None = None,
    overwrite: bool = True,
) -> Path:
    """Write `traj` as an sbto reference NPZ; return the written file path.

    Args:
        traj: Trajectory in a G1 model's qpos layout (``g1_29dof`` or ``g1_23dof``).
        out_dir: Parent directory; defaults to ``<repo>/outputs/sbto/<model>``.
        output_fps: Target fps; resample (lerp/slerp) only if set and != traj.fps.
            ``None`` keeps the trajectory's own fps with no resampling.
        name: Output file stem; defaults to ``traj.name``.
        overwrite: If False, raise when the target NPZ already exists.
    """
    # Assemble the 36-col g1_29dof-ordered qpos (base_pos(3), base_quat_wxyz(4), 29 joints).
    qpos = _to_g1_29dof_qpos(traj)

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

    # --- reorder base to OmniRetarget-native [quat(wxyz), pos] (sbto default flags) ---
    sbto_qpos = np.concatenate(
        [qpos[:, 3:7], qpos[:, 0:3], qpos[:, 7:]], axis=1
    )

    # sbto casts fps with int(...); store an integer and warn on truncation loss.
    out_fps_int = int(round(out_fps))
    if abs(out_fps - out_fps_int) > 1e-6:
        warnings.warn(
            f"sbto stores fps as an integer; {out_fps} will be written as "
            f"{out_fps_int}.",
            stacklevel=2,
        )

    # --- resolve the output file path ---
    if out_dir is None:
        out_dir = constants.REPO_ROOT / "outputs" / "sbto" / traj.model
    stem = name if name is not None else traj.name
    out_path = Path(out_dir) / f"{stem}.npz"
    if not overwrite and out_path.exists():
        raise FileExistsError(
            f"{out_path} already exists (pass overwrite=True to replace)."
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(out_path, qpos=sbto_qpos, fps=np.int64(out_fps_int))
    return out_path
