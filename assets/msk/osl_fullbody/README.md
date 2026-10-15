# osl_fullbody (imported)

Self-contained copy of the NeuMoVe Lab full-body + OSL knee-ankle prosthesis model,
exported with myo_sim 0.2.3 and with the prosthetic shank lengthened by hand. It
replaces the earlier hand-grafted `osl_fullbody` and the myo_sim 0.2.2 export
`new_osl_fullbody`, which live on the `old_osl` branch under those names.

## Provenance

- Source: `myoassist/models/osl_fullbody_v3.xml`, generated 2026-10-14 in myoassist
  (`acee39b7`, fork `github.com/guaguama/myoassist`) with

  ```bash
  python -m assist_sim combine myofullbody OpenSourceLeg_KA_L1 \
      -o models/osl_fullbody_v3.xml
  ```

  using assist_sim 1.0.0 and myo_sim 0.2.3 (`93b0ca8`). The comment block at the top
  of `body/osl_fullbody.xml` is the source's own record of what it is and how it
  was edited; it is kept verbatim.
- Against the myo_sim 0.2.2 export behind `new_osl_fullbody` the export differs only by the 0.2.3
  fixes: negated left-knee polycoefs (`knee_angle_translation2_constraint_l`,
  `knee_angle_rotation3_constraint_l`) and wrap-geom edits (`DELT1hh_ellipsoid_DELT1`,
  `EDCL/EDCM_torus_wrap` on both arms, `back_cylinder_l`).
- Hand edit in myoassist after the export: the OSL tibial pylon is 22.5 mm longer
  (`OSL_KA_L1_osl_ankle_assembly` pos y −0.1561 → −0.1786, pylon cylinder and inertial
  grown at the distal end). Chosen with `myoassist/models/osl_shank_extension.py`.
  Total mass 82.289 kg (+10.9 g).
- Meshes: 70 from myo_sim, 6 OSL parts from assist_sim. 76 STLs, byte-identical to
  `new_osl_fullbody`'s set.
- Scene: `musclemimic_scene.xml` and `musclemimic_skybox.png` from musclemimic, as for
  the other full-body models.

## Edits applied when vendoring

The same four mechanical edits musclemimic's `scripts/vendor_new_osl_fullbody_model.py`
applies, rigid socket only:

1. The 108 `file=` mesh attributes reduced to bare basenames, with
   `meshdir="../meshes"` added to `<compiler>`.
2. The export's builtin `<texture name="sky">` deleted; it collides with the scene's
   file-backed texture of the same name.
3. The exporter's `<visual>` block deleted, so the scene's values win.
4. `<include file="../scene/musclemimic_scene.xml"/>` added, and the model renamed to
   `OSLFullBody`.

Everything else, including keyframes, is the source byte-for-byte.

## Loaded by clipper

`load_model("osl_fullbody")` deletes the 40 finger joints (4 of the right-hand ones
lack the `_r` suffix in this export) and injects the 17 `*_mimic` sites:
nq 80 / nv 79 / nu 403 / nbody 106 / nkey 5 (`stand`, `walk_left`, `walk_right`,
`squat`, `lunge`). At qpos0 the lowest collision point of each sole sits within 0.1 mm
of the other (53.2 mm intact, 53.1 mm OSL); in `new_osl_fullbody` the OSL sole sat 22.5 mm higher.
