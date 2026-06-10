# clipper

View, edit, and convert reference trajectories from a variety of mocap sources
into formats suitable for downstream motion-tracking repos.

- **Sources** (read in place from `~/.g1mocap`): `DefaultDatasets` (`.npz`),
  `Lafan1` (`.npz`), `bones_seed` (`.csv`).
- **First target**: [`unitree_rl_mjlab`](../unitree_rl_mjlab) motion NPZ format.

**Entry points** (`scripts/`) — both take a **required** `--source` and `--model`
(`g1_29dof` | `g1_23dof`); there is no source autodetection:

- `visualize_trajectory.py` — pure-MuJoCo replay or interactive slider scrubbing.
- `crop_trajectory.py` — crop + height-offset + optional standing-pose padding,
  with target-specific output writers (and optional mp4 export).

## Setup

```bash
conda env create -f environment.yml      # creates env "clipper" (Python 3.12)
conda activate clipper
pip install -e .                          # install the clipper package
```

`requirements.txt` mirrors the pip deps for non-conda installs. The stack is
lean / pure-MuJoCo (no `mjlab` / `torch` / CUDA).

### Source data (`~/.g1mocap`)

The three sources are read in place from `~/.g1mocap` with this layout:

```
~/.g1mocap/
├── DefaultDatasets/      # LocoMuJoCo .npz   (source: default_datasets, 40 Hz → g1_23dof)
├── Lafan1/               # LocoMuJoCo .npz   (source: lafan1,           40 Hz → g1_23dof)
└── bones_seed/           # Bones Studio "seed" (source: bones_seed,    120 Hz → g1_29dof)
    ├── csv/<date>/*.csv  #   ~142k motion CSVs (~51 GB)
    └── metadata/         #   per-clip catalog (parquet + temporal-label jsonl)
```

`DefaultDatasets` + `Lafan1` are the Unitree-G1 retargets from
[`robfiras/loco-mujoco-datasets`](https://huggingface.co/datasets/robfiras/loco-mujoco-datasets)
(under `<set>/mocap/UnitreeG1/`); `bones_seed` is the G1 CSVs + metadata from
[`bones-studio/seed`](https://huggingface.co/datasets/bones-studio/seed). Recreate the
folder exactly with the Hugging Face CLI:

```bash
pip install -U "huggingface_hub[cli]"     # provides `huggingface-cli` (newer hub: `hf`)
mkdir -p ~/.g1mocap

# --- DefaultDatasets + Lafan1 (LocoMuJoCo G1 .npz) ---
huggingface-cli download robfiras/loco-mujoco-datasets --repo-type dataset \
    --include "DefaultDatasets/mocap/UnitreeG1/*.npz" "Lafan1/mocap/UnitreeG1/*.npz" \
    --local-dir ~/.g1mocap/_locomujoco
mkdir -p ~/.g1mocap/DefaultDatasets ~/.g1mocap/Lafan1
mv ~/.g1mocap/_locomujoco/DefaultDatasets/mocap/UnitreeG1/*.npz ~/.g1mocap/DefaultDatasets/
mv ~/.g1mocap/_locomujoco/Lafan1/mocap/UnitreeG1/*.npz          ~/.g1mocap/Lafan1/
rm -rf ~/.g1mocap/_locomujoco

# --- bones_seed (Bones Studio "seed": G1 CSVs + metadata; ~51 GB — large!) ---
huggingface-cli download bones-studio/seed --repo-type dataset \
    --include "g1.tar.gz" "metadata/*" \
    --local-dir ~/.g1mocap/bones_seed
tar -xzf ~/.g1mocap/bones_seed/g1.tar.gz --strip-components=1 -C ~/.g1mocap/bones_seed
rm ~/.g1mocap/bones_seed/g1.tar.gz        # → ~/.g1mocap/bones_seed/csv/<date>/*.csv
```

> `bones_seed` is ~51 GB across ~142k CSVs delivered as one `g1.tar.gz`, so it must be
> fetched whole; clipper then reads it lazily (one `--path` at a time). `--strip-components=1`
> drops the archive's leading `g1/` so the CSVs land at `bones_seed/csv/<date>/`.

## Visualizing trajectories

Replay a reference motion in MuJoCo. `--source` and `--model` are required (no
autodetection):

```bash
# LocoMuJoCo-format npz (DefaultDatasets / Lafan1) -> 23-DOF G1
python scripts/visualize_trajectory.py \
    --path ~/.g1mocap/DefaultDatasets/walk.npz --source default_datasets --model g1_23dof
python scripts/visualize_trajectory.py \
    --path ~/.g1mocap/Lafan1/dance1_subject2.npz --source lafan1 --model g1_23dof

# bones_seed CSV -> 29-DOF G1 (cm->m, deg->rad, auto-grounded; plays at 120 Hz)
python scripts/visualize_trajectory.py \
    --path <bones_seed.csv> --source bones_seed --model g1_29dof
```

| flag | meaning |
|---|---|
| `--source` | `default_datasets` \| `lafan1` \| `bones_seed` (required) |
| `--model` | `g1_29dof` \| `g1_23dof` (required) |
| `--mode` | `replay` (default) \| `scrub` (interactive slider + keyboard control) |
| `--fps` | playback rate override; defaults to the source's own fps |
| `--speed` | real-time multiplier (e.g. `2` = 2×) |
| `--no-loop` | play once instead of looping |
| `--no-floor` | hide the ground plane |

Any source can be played on either model — joints are matched by name, so e.g. a
23-joint Lafan1 motion on `g1_29dof` simply leaves the extra waist/wrist DOFs at 0.

### Scrub mode

`--mode scrub` opens the MuJoCo viewer plus a small slider window (a **Frame**
slider showing `frame k / N  t = ..s`, and a **Z-height** slider for a global
ground-clipping offset). Drag the sliders, or use the keyboard with the **3D
window focused** — the slider handles track the keys:

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
write**, producing a downstream motion file under `outputs/<format>/<model>/`. Frame
indices are **0-based, inclusive** — the same numbers the scrub `f` marker prints.

```bash
# Crop Lafan1 frames 100..400 -> unitree_rl_mjlab NPZ (29-DOF)
python scripts/crop_trajectory.py \
    --path ~/.g1mocap/Lafan1/dance1_subject1.npz --source lafan1 --model g1_29dof \
    --start 100 --stop 400
# -> outputs/unitree_rl_mjlab/g1_29dof/dance1_subject1_crop100-400.npz

# Raise 3 cm, pad standing at both ends, resample to 50 Hz, preview first
python scripts/crop_trajectory.py \
    --path <bones_seed.csv> --source bones_seed --model g1_29dof \
    --start 50 --stop 600 --height-offset 0.03 --pad-standing --output-fps 50 --visualize
```

| flag | meaning |
|---|---|
| `--start` / `--stop` | crop bounds (0-based, inclusive; default full clip) |
| `--height-offset` | global z added to every frame (m) |
| `--pad-standing` | add a standing pose + blended transition at each end (off by default) |
| `--pre-static`/`--pre-blend`/`--post-static`/`--post-blend` | pad durations (s); defaults 1.0 / 0.5 |
| `--output-fps` | resample (lerp + slerp) to this rate; default keeps the source fps |
| `--format` | output format (`unitree_rl_mjlab`) |
| `--name` | output file stem (default derived from the source + crop range) |
| `--visualize` / `--save-video` | replay the final clip / render it to an mp4 |

The `unitree_rl_mjlab` writer recreates upstream `csv_to_npz.py`'s output in **pure
MuJoCo + numpy** (no `mjlab`/`torch`/CUDA): per output frame it forward-kinematics the
model's *scene* XML for the full 30-body set (the 23-DOF scene pads with 6 dummy
bodies) and derives velocities by finite difference (`np.gradient` for linear/joint,
an SO3 central difference for angular). The NPZ holds `joint_pos`, `joint_vel`,
`body_pos_w`, `body_quat_w` (wxyz), `body_lin_vel_w`, `body_ang_vel_w`, and `fps`
(the root is body index 0; there is no separate root key).

## Assets

`assets/robots/unitree_g1/` is vendored from `unitree_rl_mjlab`
(`src/assets/robots/unitree_g1/`) to keep clipper self-contained and compatible:

| clipper file | upstream file |
|---|---|
| `xmls/g1_29dof.xml` | `g1.xml` (29-DOF, bare) |
| `xmls/g1_23dof.xml` | `g1_23dof.xml` (23-DOF, bare) |
| `xmls/scene_g1_29dof.xml` | `scene_g1.xml` (29-DOF + floor) |
| `xmls/scene_g1_23dof.xml` | `scene_g1_23dof.xml` (23-DOF + floor) |
| `xmls/assets/*.STL` | same (38 meshes) |

The 29-DOF XMLs are renamed from upstream `g1*` to `g1_29dof*`; the files are
self-contained (no `<include>`), so renaming is safe. Joint orderings and the
standing "home" pose are extracted into `src/clipper/constants.py` (the upstream
`*_constants.py` are `mjlab`-coupled and cannot be imported directly).

To refresh assets after an upstream change, re-copy and re-rename as above.
