# clipper
![Clipper Logo](assets/clipper.png)

View, edit, and convert reference trajectories from a variety of mocap sources
into formats suitable for downstream motion-tracking repos.

- **Supported Sources**: `LocoMujoCo DefaultDatasets` (`.npz`),
  `Lafan1` (`.npz`), `bones_seed` (`.csv`), `unilab` (`.npz`), `mj_nlp` (qpos/time CSV folder).
- **Supported Outputs**: `unilab` (motion NPZ for
  [`unitree_rl_mjlab`](../unitree_rl_mjlab)),
  `mj_nlp` (MuJoCo qpos + time CSV pair)

**Entry points** (`scripts/`):

- `visualize_trajectory.py` — MuJoCo replay or interactive slider scrubbing.
- `crop_trajectory.py` — crop + height-offset + optional standing-pose padding,
  with target-specific output writers (and optional mp4 export).

## Setup

```bash
conda env create -f environment.yml
conda activate clipper
pip install -e .
```

Creates env "clipper" (Python 3.12). `requirements.txt` mirrors pip dependencies for non-conda installs.

### Source data (`~/.g1mocap`)

The three sources are read in place from `~/.g1mocap` with this layout:

```
~/.g1mocap/
├── DefaultDatasets/      # LocoMuJoCo .npz   (source: default_datasets, 40 Hz → g1_23dof)
├── Lafan1/               # LocoMuJoCo .npz   (source: lafan1,           40 Hz → g1_23dof)
└── bones_seed/           # Bones Studio "seed" (source: bones_seed,    120 Hz → g1_29dof)
    ├── csv/<date>/*.csv
    └── metadata/
```

`DefaultDatasets` + `Lafan1` are the Unitree-G1 retargets from
[`robfiras/loco-mujoco-datasets`](https://huggingface.co/datasets/robfiras/loco-mujoco-datasets); `bones_seed` is the G1 CSVs + metadata from
[`bones-studio/seed`](https://huggingface.co/datasets/bones-studio/seed). Recreate the source directory with the Hugging Face CLI (Assumes you
have huggingface_hub installed and set up. May require access approval for each dataset first):

```bash
mkdir -p ~/.g1mocap

# --- DefaultDatasets + Lafan1 (LocoMuJoCo G1 .npz) ---
hf download robfiras/loco-mujoco-datasets --repo-type dataset \
    --include "DefaultDatasets/mocap/UnitreeG1/*.npz" "Lafan1/mocap/UnitreeG1/*.npz" \
    --local-dir ~/.g1mocap/_locomujoco
mkdir -p ~/.g1mocap/DefaultDatasets ~/.g1mocap/Lafan1
mv ~/.g1mocap/_locomujoco/DefaultDatasets/mocap/UnitreeG1/*.npz ~/.g1mocap/DefaultDatasets/
mv ~/.g1mocap/_locomujoco/Lafan1/mocap/UnitreeG1/*.npz          ~/.g1mocap/Lafan1/
rm -rf ~/.g1mocap/_locomujoco

# --- bones_seed (Bones Studio "seed": G1 CSVs + metadata; ~51 GB — large!) ---
hf download bones-studio/seed --repo-type dataset \
    --include "g1.tar.gz" "metadata/*" \
    --local-dir ~/.g1mocap/bones_seed
tar -xzf ~/.g1mocap/bones_seed/g1.tar.gz --strip-components=1 -C ~/.g1mocap/bones_seed
rm ~/.g1mocap/bones_seed/g1.tar.gz        # → ~/.g1mocap/bones_seed/csv/<date>/*.csv
```

> `bones_seed` is ~51 GB across ~142k CSVs delivered as one `g1.tar.gz`, so it must be fetched whole and extracted.

## Visualizing Trajectories

`scripts/visualize_trajectory.py` is used to replay a reference motion in MuJoCo (`--path`, `--source`, and `--model` are required). Examples:

```bash
# LocoMuJoCo-format npz (DefaultDatasets / Lafan1) -> 23-DOF G1
python scripts/visualize_trajectory.py \
    --path ~/.g1mocap/DefaultDatasets/walk.npz --source default_datasets --model g1_23dof
python scripts/visualize_trajectory.py \
    --path ~/.g1mocap/Lafan1/dance1_subject2.npz --source lafan1 --model g1_23dof

# bones_seed CSV -> 29-DOF G1 (cm->m, deg->rad, auto-grounded; plays at 120 Hz)
python scripts/visualize_trajectory.py \
    --path <bones_seed.csv> --source bones_seed --model g1_29dof

# unitree_rl_mjlab format
python scripts/visualize_trajectory.py \
    --path outputs/unitree_rl_mjlab/g1_29dof/<clip>.npz --source unilab --model g1_29dof
```

| flag | meaning |
|---|---|
| `--path` | path to reference trajectory (required) |
| `--source` | `default_datasets` \| `lafan1` \| `bones_seed` \| `unilab` \| `mj_nlp` (required) |
| `--model` | `g1_29dof` \| `g1_23dof` (required) |
| `--mode` | `replay` (default) \| `scrub` (interactive slider + keyboard control) |
| `--fps` | playback rate override; defaults to the source's own fps |
| `--speed` | real-time multiplier (e.g. `2` = 2×) |
| `--no-loop` | play once instead of looping |
| `--no-floor` | hide the ground plane |

Any source can be played on either model — joints are matched by name, so e.g. a
23-joint Lafan1 motion on `g1_29dof` simply leaves the extra waist/wrist DOFs at 0.

### Scrub mode

`--mode scrub` opens the MuJoCo viewer plus a small slider window for scrubbing 
through trajectory frames and adjusting global ground-clipping offset. Drag the sliders, 
or use the keyboard with the **slider window focused**. Additionally, current frame and 
z-offset can be marked and printed for reference when cropping.

| key | action |
|---|---|
| `a` / `d` | step −1 / +1 frame |
| `w` / `s` | jump −10 / +10 frames |
| `q` / `e` | raise / lower the global z-offset (±5 mm) |
| `r` | reset the z-offset to 0 |
| `f` | print `frame k/N  t=..s  z=+..m` (mark a clip's start/end for cropping) |

Close either window to quit.

## Cropping & converting trajectories

`crop_trajectory.py` runs **load → crop → height-offset → (optional) standing pad →
write**, producing a downstream motion file under `outputs/` (e.g.
`outputs/unitree_rl_mjlab/<model>/` for `unilab`, `outputs/mj_nlp/<model>/` for `mj_nlp`). The standing
pose pad prepends and appends a standing pose to the trajectory. Examples:

```bash
# Crop Lafan1 trajectory between frames 100 and 150 -> unitree_rl_mjlab NPZ (29-DOF)
python scripts/crop_trajectory.py \
    --path ~/.g1mocap/Lafan1/dance1_subject1.npz --source lafan1 --model g1_29dof \
    --start 100 --stop 150 --format unilab
# -> outputs/unitree_rl_mjlab/g1_29dof/dance1_subject1_crop100-150.npz

# Raise 3 cm, pad standing at both ends, resample to 50 Hz, preview first
python scripts/crop_trajectory.py \
    --path <bones_seed.csv> --source bones_seed --model g1_29dof --format unilab \
    --start 50 --stop 200 --height-offset 0.03 --pad-standing --output-fps 50 --visualize
```

| flag | meaning |
|---|---|
| `--path` | path to reference trajectory (required) |
| `--source` | `default_datasets` \| `lafan1` \| `bones_seed` \| `unilab` \| `mj_nlp` (required) |
| `--model` | `g1_29dof` \| `g1_23dof` (required) |
| `--start` / `--stop` | crop bounds (0-based, inclusive; default full clip) |
| `--height-offset` | global z added to every frame (m) |
| `--pad-standing` | add a standing pose + blended transition at each end (off by default) |
| `--pre-static`/`--pre-blend`/`--post-static`/`--post-blend` | pad durations (s); defaults 1.0 / 0.5 |
| `--output-fps` | resample (lerp + slerp) to this rate; default keeps the source fps |
| `--format` | output format (`unilab` \| `mj_nlp`) (required) |
| `--name` | output file stem (default derived from the source + crop range) |
| `--visualize` / `--save-video` | replay the final clip / render it to an mp4 |

## Musculoskeletal (MSK) models

In addition to the Unitree G1, clipper supports three musculoskeletal models from
[`musclemimic`](../../01_projpegleg/musclemimic) and
[`myo_sim`](../../01_projpegleg/myo_sim), vendored under `assets/msk/`:

| `--model` | model | nq | base | source repo |
|---|---|---|---|---|
| `osl_ka` | MyoLeg80 with right-leg OSL prosthetic | 30 | free | musclemimic |
| `myofullbody` | full-body MyoSkeleton (fingers disabled) | 89 | free | musclemimic |
| `myolegs` | MyoLegs (biological, myo_sim) | 35 | free | myo_sim |

These use a single new source/output format, **`musclemimic`**, for the retargeted
clips in `~/.musclemimic/caches/AMASS/<Model>/` — self-describing LocoMuJoCo-style
`.npz` (embed `qpos`, `qvel`, `frequency` = 100 Hz, `joint_names`, body/site FK).
Joints are mapped **by name** into the target model, so clips load regardless of the
model's exact DOF set.

```bash
# Visualize an OSL_KA walk clip
python scripts/visualize_trajectory.py \
    --path ~/.musclemimic/caches/AMASS/MyoLeg80_OSL_KA/gmr/KIT/7/WalkingStraightForwards01_poses.npz \
    --source musclemimic --model osl_ka

# Crop a full-body clip and write a round-trip musclemimic NPZ
python scripts/crop_trajectory.py \
    --path ~/.musclemimic/caches/AMASS/MyoFullBody/<clip>.npz \
    --source musclemimic --model myofullbody --start 30 --stop 200 --format musclemimic
# -> outputs/musclemimic/myofullbody/<clip>_crop30-200.npz
```

Add `musclemimic` to the `--source` choices (visualize + crop) and to the `--format`
choices (crop). The `musclemimic` writer reproduces the full cache schema (velocities
via `mj_differentiatePos`, body + mimic-site FK via `mj_forward`), so cropped clips are
re-loadable by clipper and consumable by musclemimic.

**Notes / caveats**
- **`myolegs` has no retargeted clips of its own** — it is driven by the
  `MyoLeg80_OSL_KA` cache (a full-body clip with the upper body removed). The
  prosthetic right knee/ankle are renamed back to the biological joints
  (`osl_knee_angle_r → knee_angle_r`, `osl_ankle_angle_r → ankle_angle_r`); the 4
  `socket_*` DOFs are dropped and the right-knee coupler DOFs stay at 0 (a minor
  visual approximation). Pass an `MyoLeg80_OSL_KA` clip with `--model myolegs`.
- **`myofullbody` ships with fingers disabled** (the 40 finger joints are removed at
  load, matching musclemimic's `disable_fingers=True` default) so nq = 89 matches the
  caches. The standing-pose pad for MSK models seeds from the model's first keyframe.
- MSK XMLs bundle their own scene (floor/lights/cameras), so `--no-floor` is a no-op
  for them; viewers use their own pelvis-centered free camera and ignore scene cameras.

**Prosthesis right-ankle correction.** The OSL joints are driven by raw biological
measurements, so the prosthetic foot may not sit fully flat. In `scrub` mode an
**R-ankle** slider (keys `z` / `c` = ±1°) applies a global right-ankle offset; press
`f` to print the current value alongside the frame/z marker. Bake the value you found
into a clip with `crop_trajectory.py --ankle-offset <deg>` (adds it to the right-ankle
joint of every frame). This is `osl_ka`-only (joint `osl_ankle_angle_r`): other models
get no R-ankle slider and `--ankle-offset` errors on them.

## Assets

`assets/robots/unitree_g1/` is imported from `unitree_rl_mjlab`
(`src/assets/robots/unitree_g1/`):

| clipper file | upstream file |
|---|---|
| `xmls/g1_29dof.xml` | `g1.xml` (29-DOF, bare) |
| `xmls/g1_23dof.xml` | `g1_23dof.xml` (23-DOF, bare) |
| `xmls/scene_g1_29dof.xml` | `scene_g1.xml` (29-DOF + floor) |
| `xmls/scene_g1_23dof.xml` | `scene_g1_23dof.xml` (23-DOF + floor) |
| `xmls/assets/*.STL` | same (38 meshes) |
