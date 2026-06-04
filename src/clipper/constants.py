"""Plain-python constants for the Unitree G1 models used by clipper.

These are extracted (no ``mjlab`` dependency) from unitree_rl_mjlab so clipper's
visualization and NPZ writer stay compatible with it:

- Joint orderings are copied verbatim from ``scripts/csv_to_npz.py`` (the order in
  which DOFs appear in the intermediate CSV and the output NPZ ``joint_pos``).
- The standing "home" pose is the ``HOME_KEYFRAME`` from the ``*_constants.py``
  files (identical for both models), used by the standing-pose padding feature.

Quaternion convention here is MuJoCo's ``wxyz`` (the clipper canonical format).
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

# --------------------------------------------------------------------------- #
# Asset paths (vendored from unitree_rl_mjlab; 29-DOF XMLs renamed to g1_29dof).
# --------------------------------------------------------------------------- #
REPO_ROOT: Path = Path(__file__).resolve().parents[2]
_XML_DIR: Path = REPO_ROOT / "assets" / "robots" / "unitree_g1" / "xmls"

# Bare robot (floating base, no floor) — preferred for kinematic playback.
G1_29DOF_XML: Path = _XML_DIR / "g1_29dof.xml"
G1_23DOF_XML: Path = _XML_DIR / "g1_23dof.xml"
# Scene wrappers (add a ground plane + actuators/sensors) — nicer for viewing.
G1_29DOF_SCENE_XML: Path = _XML_DIR / "scene_g1_29dof.xml"
G1_23DOF_SCENE_XML: Path = _XML_DIR / "scene_g1_23dof.xml"

# --------------------------------------------------------------------------- #
# Joint orderings (verbatim from csv_to_npz.py:337-395).
# --------------------------------------------------------------------------- #
G1_29DOF_JOINT_NAMES: tuple[str, ...] = (
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)

G1_23DOF_JOINT_NAMES: tuple[str, ...] = (
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
)

# Registry keyed by the clipper `--model` value.
JOINT_NAMES: dict[str, tuple[str, ...]] = {
    "g1_29dof": G1_29DOF_JOINT_NAMES,
    "g1_23dof": G1_23DOF_JOINT_NAMES,
}

ROBOT_XML: dict[str, Path] = {
    "g1_29dof": G1_29DOF_XML,
    "g1_23dof": G1_23DOF_XML,
}

SCENE_XML: dict[str, Path] = {
    "g1_29dof": G1_29DOF_SCENE_XML,
    "g1_23dof": G1_23DOF_SCENE_XML,
}

MODELS: tuple[str, ...] = ("g1_29dof", "g1_23dof")

# --------------------------------------------------------------------------- #
# Trajectory sources (clipper `--source` values; no autodetection).
# --------------------------------------------------------------------------- #
# DefaultDatasets and Lafan1 share the LocoMuJoCo-format npz loader; bones_seed
# has its own CSV loader. The natural target model for each source is noted, but
# any source can be visualized on any model via the by-name qpos builder.
SOURCES: tuple[str, ...] = ("default_datasets", "lafan1", "bones_seed")

# bones_seed CSVs carry no frame rate; this is the assumed default (override via
# the `--fps` CLI flag). Documented as a guess, not a verified capture rate.
BONES_SEED_DEFAULT_FPS: float = 30.0

# --------------------------------------------------------------------------- #
# Standing "home" pose (HOME_KEYFRAME, identical for both models).
# --------------------------------------------------------------------------- #
HOME_BASE_POS: tuple[float, float, float] = (0.0, 0.0, 0.8)
HOME_BASE_QUAT_WXYZ: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)

# (regex matched against joint name, value in radians). First match wins;
# joints matching nothing default to 0.0. Mirrors the mjlab HOME_KEYFRAME dict.
_HOME_JOINT_RULES: tuple[tuple[str, float], ...] = (
    ("left_shoulder_roll_joint", 0.18),
    ("right_shoulder_roll_joint", -0.18),
    (r".*_hip_pitch_joint", -0.1),
    (r".*_knee_joint", 0.3),
    (r".*_ankle_pitch_joint", -0.2),
    (r".*_shoulder_pitch_joint", 0.35),
    (r".*_elbow_joint", 0.87),
)


def resolve_home_joint_pos(joint_names: tuple[str, ...] | list[str]) -> np.ndarray:
    """Return the home joint angles (radians) ordered to match `joint_names`."""
    out = np.zeros(len(joint_names), dtype=np.float64)
    for i, name in enumerate(joint_names):
        for pattern, value in _HOME_JOINT_RULES:
            if re.fullmatch(pattern, name):
                out[i] = value
                break
    return out


def home_qpos(model: str) -> np.ndarray:
    """Full free-joint home qpos: [base_pos(3), base_quat_wxyz(4), joints(DOF)]."""
    if model not in JOINT_NAMES:
        raise ValueError(f"Unknown model {model!r}; expected one of {MODELS}.")
    joints = resolve_home_joint_pos(JOINT_NAMES[model])
    return np.concatenate(
        [np.asarray(HOME_BASE_POS), np.asarray(HOME_BASE_QUAT_WXYZ), joints]
    ).astype(np.float64)
