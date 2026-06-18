"""Load a MuJoCo model (Unitree G1 or musculoskeletal) for visualization.

For the **G1** models this uses the *bare* robot XMLs (clean ``nq = 7 + DOF``),
which avoids the placeholder dummy bodies in ``scene_g1_23dof.xml`` (parked at
z=20); a ground plane is added programmatically via ``MjSpec`` when wanted, so the
qpos layout stays identical across models.

For the **MSK** models (``assets/msk``) the main XML already ``<include>``s its own
scene (floor/lights/skybox/cameras), so it is compiled directly. The ``*_mimic``
tracking sites are injected via ``MjSpec`` (as musclemimic does at build time) so
the round-trip writer can reproduce the cache's site FK; they are group-4 geoms,
hidden in the viewer by default. The ``floor`` argument is a no-op for MSK models.
"""

from __future__ import annotations

import mujoco
import numpy as np

from . import constants


def _remove_joints(spec: "mujoco.MjSpec", joint_names: tuple[str, ...]) -> None:
    """Delete the named joints from the spec (mirrors musclemimic finger disabling).

    Joints absent from the spec are ignored. Bodies are left intact (they weld to
    their parent), so body indexing matches musclemimic's cached models.
    """
    drop = set(joint_names)
    for joint in [j for j in spec.joints if j.name in drop]:
        spec.delete(joint)


def _add_mimic_sites(spec: "mujoco.MjSpec", body2site: dict[str, str]) -> None:
    """Inject ``*_mimic`` tracking sites onto named bodies (mirrors musclemimic).

    Each site is a small invisible (group 4) box at the body origin. Bodies absent
    from the spec are skipped so the same dict tolerates trimmed model variants.
    """
    for body_name, site_name in body2site.items():
        try:
            body = spec.body(body_name)
        except (KeyError, ValueError):
            continue
        if body is None:
            continue
        body.add_site(
            name=site_name,
            group=4,
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[0.075, 0.05, 0.025],
            rgba=[1.0, 0.0, 0.0, 0.5],
            pos=[0.0, 0.0, 0.0],
        )


def load_model(model_id: str, floor: bool = True) -> mujoco.MjModel:
    """Compile and return the MuJoCo model for `model_id`.

    Args:
        model_id: a G1 id (``"g1_29dof"``/``"g1_23dof"``) or an MSK id
            (``"osl_ka"``/``"myofullbody"``/``"myolegs"``).
        floor: If True, add a ground-plane geom and a light to the world. Ignored
            for MSK models (their bundled scene already provides one).

    Returns:
        Compiled ``mujoco.MjModel``. qpos layout is ``[base(7), joints(DOF)]``.
    """
    if model_id not in constants.ROBOT_XML:
        raise ValueError(
            f"Unknown model {model_id!r}; expected one of {constants.MODELS}."
        )
    xml_path = str(constants.ROBOT_XML[model_id])

    if model_id in constants.MSK_MODELS:
        spec = mujoco.MjSpec.from_file(xml_path)
        _remove_joints(spec, constants.MSK_REMOVE_JOINTS.get(model_id, ()))
        _add_mimic_sites(spec, constants.MSK_MIMIC_SITES.get(model_id, {}))
        return spec.compile()

    if not floor:
        return mujoco.MjModel.from_xml_path(xml_path)

    spec = mujoco.MjSpec.from_file(xml_path)

    # Reconstruct scene_g1_*.xml's environment
    spec.stat.center = np.array([1.0, 0.7, 1.0])
    spec.stat.extent = 0.8

    spec.visual.headlight.ambient = [0.1, 0.1, 0.1]
    spec.visual.headlight.diffuse = [0.6, 0.6, 0.6]
    spec.visual.headlight.specular = [0.9, 0.9, 0.9]
    spec.visual.rgba.haze = [0.15, 0.25, 0.35, 1.0]
    spec.visual.global_.azimuth = -140.0
    spec.visual.global_.elevation = -20.0

    sun = spec.worldbody.add_light()
    sun.type = mujoco.mjtLightType.mjLIGHT_DIRECTIONAL
    sun.pos = np.array([1.0, 0.0, 3.5])
    sun.dir = np.array([0.0, 0.0, -1.0])

    # Flat-black skybox + checker ground plane (with edge marks) from the scene.
    spec.add_texture(
        name="skybox",
        type=mujoco.mjtTexture.mjTEXTURE_SKYBOX,
        builtin=mujoco.mjtBuiltin.mjBUILTIN_FLAT,
        rgb1=[0.0, 0.0, 0.0],
        rgb2=[0.0, 0.0, 0.0],
        width=512,
        height=3072,
    )
    groundplane_tex = spec.add_texture(
        name="groundplane",
        type=mujoco.mjtTexture.mjTEXTURE_2D,
        builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
        rgb1=[0.2, 0.3, 0.4],
        rgb2=[0.1, 0.2, 0.3],
        width=300,
        height=300,
    )
    groundplane_tex.mark = mujoco.mjtMark.mjMARK_EDGE
    groundplane_tex.markrgb = [0.8, 0.8, 0.8]
    groundplane_mat = spec.add_material(
        name="groundplane",
        textures=["", "groundplane"],
        texrepeat=[5, 5],
        reflectance=0.2,
    )
    groundplane_mat.texuniform = True
    floor_geom = spec.worldbody.add_geom()
    floor_geom.name = "floor"
    floor_geom.type = mujoco.mjtGeom.mjGEOM_PLANE
    floor_geom.size = np.array([0.0, 0.0, 0.05])
    floor_geom.material = "groundplane"

    return spec.compile()


def home_keyframe_qpos(model_id: str) -> np.ndarray:
    """Return the model's first keyframe qpos (the MSK standing-pose source).

    The G1 home pose is rule-based (``constants.home_qpos``); the MSK models ship
    real ``<key>`` keyframes, so the standing-pose pad seeds from ``key_qpos[0]``.

    Raises:
        ValueError: if the model defines no keyframes.
    """
    model = load_model(model_id)
    if model.nkey < 1:
        raise ValueError(f"Model {model_id!r} has no keyframe to use as a home pose.")
    return np.array(model.key_qpos[0], dtype=np.float64)
