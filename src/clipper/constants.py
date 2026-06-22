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
# Musculoskeletal (MSK) models, vendored under assets/msk (see clipper memory).
# Each main XML already <include>s its own scene (floor/lights/skybox/cameras),
# so it is loaded directly — no programmatic floor injection (see model.py).
# --------------------------------------------------------------------------- #
_MSK_DIR: Path = REPO_ROOT / "assets" / "msk"
MYOLEGS_OSL_KA_XML: Path = _MSK_DIR / "myoLeg80_OSL_KA" / "myolegs_OSL_KA.xml"
MYOFULLBODY_XML: Path = _MSK_DIR / "musclemimic_models" / "body" / "myofullbody.xml"
MYOLEGS_XML: Path = _MSK_DIR / "myo_sim" / "leg" / "myolegs.xml"

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
    "osl_ka": MYOLEGS_OSL_KA_XML,
    "myofullbody": MYOFULLBODY_XML,
    "myolegs": MYOLEGS_XML,
}

SCENE_XML: dict[str, Path] = {
    "g1_29dof": G1_29DOF_SCENE_XML,
    "g1_23dof": G1_23DOF_SCENE_XML,
}

# MSK models are loaded via assets/msk XMLs that already bundle a scene; they
# carry no separate bare/scene split and use the musculoskeletal source/writer.
MSK_MODELS: frozenset[str] = frozenset(
    ("osl_ka", "myofullbody", "myolegs")
)

MODELS: tuple[str, ...] = ("g1_29dof", "g1_23dof", *sorted(MSK_MODELS))

# Right-ankle joint per model — the single hinge used by the scrub viewer's R-ankle
# slider and crop's `--ankle-offset` (a global degree delta added to that joint, to
# flatten the OSL prosthetic foot driven by raw biological measurements). Resolved
# to a qpos address at runtime via `qpos.joint_qpos_address`. Intentionally limited
# to `osl_ka`: this correction is prosthesis-specific, so other models get no R-ankle
# slider (scrub) and a clear error from `--ankle-offset` (crop).
RIGHT_ANKLE_JOINT: dict[str, str] = {
    "osl_ka": "osl_ankle_angle_r",
}

# --------------------------------------------------------------------------- #
# Trajectory sources (clipper `--source` values; no autodetection).
# --------------------------------------------------------------------------- #
# DefaultDatasets and Lafan1 share the LocoMuJoCo-format npz loader; bones_seed
# has its own CSV loader; `unilab` re-loads clipper's own unitree_rl_mjlab output
# NPZ and `mj_nlp` re-loads its mj-nlp qpos/time CSV folder (so written clips can
# be re-visualized / re-cropped). The natural target model for each source is
# noted, but any source can be visualized on any model via the by-name qpos builder.
SOURCES: tuple[str, ...] = (
    "default_datasets",
    "lafan1",
    "bones_seed",
    "unilab",
    "mj_nlp",
    # musclemimic retargeted clips in ~/.musclemimic/caches (self-describing npz);
    # the matching round-trip writer re-emits this schema. See sources/musclemimic.py.
    "musclemimic",
)

# bones_seed CSVs carry no frame rate, but the source capture rate is 120 Hz —
# verified from the dataset's seed metadata (move_duration_frames / temporal-label
# event seconds clusters tightly at 120 across ~142k clips). Override via `--fps`.
BONES_SEED_DEFAULT_FPS: float = 120.0

# musclemimic caches store `frequency` (100 Hz for the current AMASS retargets);
# used only as a fallback if a file is missing it. Override via `--fps`.
MUSCLEMIMIC_DEFAULT_FPS: float = 100.0

# --------------------------------------------------------------------------- #
# MSK joint remap: the myo_sim `myolegs` model has no retargeted trajectories of
# its own, so it is driven by the `MyoLeg80_OSL_KA` cache (a full-body clip with
# the upper body removed). The biological left leg shares joint names verbatim;
# the right leg's prosthetic knee/ankle were renamed but hold the same biological
# values, so we rename them back. The 4 `socket_*` DOFs have no biological
# counterpart and are dropped (negligible); right-knee coupler DOFs stay 0 (a
# minor visual approximation). Verified by FK: feet grounded, right knee mirrors
# the left with matching sign.
MYOLEGS_OSL_ALIASES: dict[str, str] = {
    "osl_knee_angle_r": "knee_angle_r",
    "osl_ankle_angle_r": "ankle_angle_r",
}

# --------------------------------------------------------------------------- #
# Mimic-site definitions (body_name -> site_name), replicated verbatim from
# musclemimic's per-env `body2sites_for_mimic` dicts. These `*_mimic` sites are
# NOT in the bare XMLs — musclemimic injects them at build time; clipper does the
# same in model.load_model so the round-trip writer can reproduce the cache's
# site_xpos/site_xmat block. `myolegs` mirrors OSL_KA's 9-site layout but uses the
# biological right-leg bodies (no OSL assemblies).
# MSK joints to delete from the spec before compiling, per model. musclemimic
# builds MyoFullBody with `disable_fingers=True` by default, removing these 40
# finger joints (nq 129->89, nv 128->88, njnt 123->83; finger bodies stay, so
# nbody is unchanged at 102) — matching the retargeted caches' layout exactly.
# Removing the joints is sufficient for cache parity; muscles/tendons aren't in
# the cache schema and don't affect kinematics, so they are left in place.
MSK_REMOVE_JOINTS: dict[str, tuple[str, ...]] = {
    "myofullbody": tuple(
        f"{j}_{side}"
        for side in ("r", "l")
        for j in (
            "cmc_flexion", "cmc_abduction", "mp_flexion", "ip_flexion",
            "mcp2_flexion", "mcp2_abduction", "mcp3_flexion", "mcp3_abduction",
            "mcp4_flexion", "mcp4_abduction", "mcp5_flexion", "mcp5_abduction",
            "md2_flexion", "md3_flexion", "md4_flexion", "md5_flexion",
            "pm2_flexion", "pm3_flexion", "pm4_flexion", "pm5_flexion",
        )
    ),
}

MSK_MIMIC_SITES: dict[str, dict[str, str]] = {
    "osl_ka": {
        "pelvis": "pelvis_mimic",
        "femur_l": "left_hip_mimic",
        "tibia_l": "left_knee_mimic",
        "talus_l": "left_ankle_mimic",
        "toes_l": "left_toes_mimic",
        "femur_r": "right_hip_mimic",
        "osl_knee_assembly": "right_knee_mimic",
        "osl_ankle_assembly": "right_ankle_mimic",
        "osl_foot_assembly": "right_toes_mimic",
    },
    "myofullbody": {
        "pelvis": "pelvis_mimic",
        "lumbar1": "upper_body_mimic",
        "head": "head_mimic",
        "humerus_l": "left_shoulder_mimic",
        "ulna_l": "left_elbow_mimic",
        "lunate_l": "left_hand_mimic",
        "humerus_r": "right_shoulder_mimic",
        "ulna_r": "right_elbow_mimic",
        "lunate_r": "right_hand_mimic",
        "femur_l": "left_hip_mimic",
        "tibia_l": "left_knee_mimic",
        "talus_l": "left_ankle_mimic",
        "toes_l": "left_toes_mimic",
        "femur_r": "right_hip_mimic",
        "tibia_r": "right_knee_mimic",
        "talus_r": "right_ankle_mimic",
        "toes_r": "right_toes_mimic",
    },
    "myolegs": {
        "pelvis": "pelvis_mimic",
        "femur_l": "left_hip_mimic",
        "tibia_l": "left_knee_mimic",
        "talus_l": "left_ankle_mimic",
        "toes_l": "left_toes_mimic",
        "femur_r": "right_hip_mimic",
        "tibia_r": "right_knee_mimic",
        "talus_r": "right_ankle_mimic",
        "toes_r": "right_toes_mimic",
    },
}

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
