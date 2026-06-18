# myoLeg80_OSL_KA (imported)

Self-contained copy of the 80-muscle transfemoral OSL prosthetic model from
[myoassist](https://github.com/amathislab/myoassist), used by the new
`MyoLeg80_OSL_KA` / `MjxMyoLeg80_OSL_KA` environments in
`musclemimic/environments/humanoids/myoleg80_osl_ka.py`.

## Provenance

- Source: `myoassist/models/80muscle/myoLeg80_OSL_KA/` (see `README_upstream.md`).
- Mesh STLs copied from the `myo_sim` repository (`myo_sim/meshes/`).
- Original include/meshdir paths rewritten so this directory loads standalone:
  - `meshdir`/`texturedir` → `meshes/`
  - Includes for `terrain_config80.xml`, `myotorso_rigid_assets.xml`,
    `myotorso_rigid_chain.xml` resolved locally.
  - Per-mesh `<mesh file="...">` paths reduced to bare filenames.

No structural changes to the model (joints, bodies, actuators, equality
constraints, keyframes, sensors are identical to upstream).

## Layout

```
myoLeg80_OSL_KA/
├── myolegs_OSL_KA.xml          # top-level model
├── terrain_config80.xml        # ground/skybox/hfield
├── myotorso_rigid_assets.xml   # rigid passive torso
├── myotorso_rigid_chain.xml
├── assets/
│   ├── myolegs_assets_OSL_KA.xml
│   └── myolegs_chain_OSL_KA.xml
└── meshes/                     # 24 STLs
```

## Quick load test

```bash
uv run python -c "import mujoco; m = mujoco.MjModel.from_xml_path('myolegs_OSL_KA.xml'); print(m.nq, m.nu, m.nbody)"
# expected: 30 56 19
```
