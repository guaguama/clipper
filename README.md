# clipper

View, edit, and convert reference trajectories from a variety of mocap sources
into formats suitable for downstream motion-tracking repos.

- **Sources** (read in place from `~/.g1mocap`): `DefaultDatasets` (`.npz`),
  `Lafan1` (`.npz`), `bones_seed` (`.csv`).
- **First target**: [`unitree_rl_mjlab`](../unitree_rl_mjlab) motion NPZ format.

Planned entry points (not yet implemented): `visualize_trajectory.py` (pure-MuJoCo
replay / slider scrubbing) and `crop_trajectory.py` (crop + height-offset +
optional standing-pose padding, with target-specific output writers). Both take a
**required** `--source` and a `--model` (`g1_29dof` | `g1_23dof`); there is no
source autodetection.

## Setup

```bash
conda env create -f environment.yml      # creates env "clipper" (Python 3.12)
conda activate clipper
pip install -e .                          # install the clipper package
```

`requirements.txt` mirrors the pip deps for non-conda installs. The stack is
lean / pure-MuJoCo (no `mjlab` / `torch` / CUDA).

## Visualizing trajectories

Replay a reference motion in MuJoCo. `--source` and `--model` are required (no
autodetection):

```bash
# LocoMuJoCo-format npz (DefaultDatasets / Lafan1) -> 23-DOF G1
python scripts/visualize_trajectory.py \
    --path ~/.g1mocap/DefaultDatasets/walk.npz --source default_datasets --model g1_23dof
python scripts/visualize_trajectory.py \
    --path ~/.g1mocap/Lafan1/dance1_subject2.npz --source lafan1 --model g1_23dof

# bones_seed CSV -> 29-DOF G1 (cm->m, deg->rad, auto-grounded; no fps stored)
python scripts/visualize_trajectory.py \
    --path <bones_seed.csv> --source bones_seed --model g1_29dof --fps 30
```

| flag | meaning |
|---|---|
| `--source` | `default_datasets` \| `lafan1` \| `bones_seed` (required) |
| `--model` | `g1_29dof` \| `g1_23dof` (required) |
| `--mode` | `replay` (default) \| `scrub` (interactive slider + keyboard control) |
| `--fps` | playback rate override; also sets the rate for bones_seed |
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
