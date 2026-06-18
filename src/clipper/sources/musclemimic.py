"""Loader for musclemimic retargeted clips (``~/.musclemimic/caches/AMASS/...``).

These ``.npz`` are self-describing LocoMuJoCo-style files written by musclemimic's
GMR retargeting (and by clipper's own round-trip writer). The fields we read:

- ``qpos`` (N, nq): for a free-base model, ``[base_pos(3), base_quat_wxyz(4),
  joints]``; the quaternion is MuJoCo ``wxyz`` already — no conversion.
- ``frequency``: scalar Hz (100 for the current AMASS retargets).
- ``joint_names`` (njnt): names in qpos order, the free base named ``root`` first.
- ``jnt_type``: MuJoCo joint types; ``jnt_type[0] == 0`` marks a free base.

Because the file carries its own ``joint_names``, one loader serves every MSK model
— joints are mapped *by name* into the target model via ``build_qpos`` (joints the
model lacks are dropped, e.g. finger DOFs or the OSL ``socket_*`` joints).

The myo_sim ``myolegs`` model has no cache of its own and is driven by the
``MyoLeg80_OSL_KA`` cache: source joint names are first remapped through
``constants.MYOLEGS_OSL_ALIASES`` (``osl_knee_angle_r -> knee_angle_r`` etc.) so the
prosthetic right-leg angles land on the biological joints. See [[clipper-data-sources]].
"""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np

from .. import constants
from ..qpos import build_qpos
from ..trajectory import Trajectory


def load(
    path: str | Path,
    model: mujoco.MjModel,
    model_id: str,
    source: str = "musclemimic",
    fps: float | None = None,
) -> Trajectory:
    """Load a musclemimic cache npz into a `Trajectory` in `model`'s qpos layout."""
    path = Path(path)
    data = np.load(path, allow_pickle=True)
    raw = np.asarray(data["qpos"], dtype=np.float64)
    if raw.ndim != 2:
        raise ValueError(f"{path.name}: expected 2-D qpos, got shape {raw.shape}")

    joint_names = [str(j) for j in np.asarray(data["joint_names"]).reshape(-1)]
    jnt_type = np.asarray(data["jnt_type"]).reshape(-1)
    free_base = len(jnt_type) > 0 and int(jnt_type[0]) == int(mujoco.mjtJoint.mjJNT_FREE)
    if not free_base:
        raise ValueError(
            f"{path.name}: only free-base MSK clips are supported "
            f"(jnt_type[0]={None if not len(jnt_type) else int(jnt_type[0])})."
        )

    # Free joint occupies qpos[:, :7]; the remaining columns are the 1-DOF joints
    # in joint_names[1:] order.
    base_pos = raw[:, 0:3]
    base_quat_wxyz = raw[:, 3:7]  # MuJoCo wxyz, no conversion
    dof_names = joint_names[1:]
    if raw.shape[1] != 7 + len(dof_names):
        raise ValueError(
            f"{path.name}: qpos width {raw.shape[1]} != 7 + {len(dof_names)} joints."
        )

    # Drive myo_sim `myolegs` from the OSL_KA cache by renaming prosthetic joints.
    aliases = constants.MYOLEGS_OSL_ALIASES if model_id == "myolegs" else {}
    joint_angles = {
        aliases.get(name, name): raw[:, 7 + i] for i, name in enumerate(dof_names)
    }

    qpos = build_qpos(model, base_pos, base_quat_wxyz, joint_angles)

    if fps is None:
        freq = data["frequency"] if "frequency" in data.files else None
        fps = (
            float(np.asarray(freq).reshape(-1)[0])
            if freq is not None
            else constants.MUSCLEMIMIC_DEFAULT_FPS
        )

    return Trajectory(
        qpos=qpos,
        fps=float(fps),
        model=model_id,
        source=source,
        name=path.stem,
    )
