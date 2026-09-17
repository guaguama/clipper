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

import functools

import mujoco
import numpy as np

from . import constants


# qpos width per joint type, for deriving qpos addresses straight from an uncompiled
# spec (``spec.joints`` is in qpos order — verified exact against ``jnt_qposadr`` for
# every MSK model). Avoids a throwaway compile, which is slow for these models.
_JNT_NQ: dict[int, int] = {
    mujoco.mjtJoint.mjJNT_FREE: 7,
    mujoco.mjtJoint.mjJNT_BALL: 4,
    mujoco.mjtJoint.mjJNT_SLIDE: 1,
    mujoco.mjtJoint.mjJNT_HINGE: 1,
}


def _remove_joints(spec: "mujoco.MjSpec", joint_names: tuple[str, ...]) -> None:
    """Delete the named joints from the spec (mirrors musclemimic finger disabling).

    Joints absent from the spec are ignored. Bodies are left intact (they weld to
    their parent), so body indexing matches musclemimic's cached models.

    Any ``<key>`` in the spec is declared against the *pre-deletion* nq, so its qpos
    is trimmed to the surviving columns first — otherwise the compile fails with
    ``keyframe 'stand': invalid qpos size``. Only ``new_osl_fullbody`` currently ships
    real keyframes (5 of them, qpos-only); the hand-built models carry none in the
    spec, so this is a no-op for them.
    """
    drop = set(joint_names)
    victims = [j for j in spec.joints if j.name in drop]
    if not victims:
        return

    if spec.keys:
        adr: dict[str, int] = {}
        nq = 0
        for joint in spec.joints:
            adr[joint.name] = nq
            nq += _JNT_NQ[joint.type]
        gone = {
            a
            for j in victims
            for a in range(adr[j.name], adr[j.name] + _JNT_NQ[j.type])
        }
        keep = [i for i in range(nq) if i not in gone]
        for key in spec.keys:
            if len(key.qpos) == nq:  # leave anything already sized differently alone
                key.qpos = np.asarray(key.qpos)[keep].tolist()

    for joint in victims:
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
        model_id: a G1 id (``"g1_29dof"``/``"g1_23dof"``) or any MSK id in
            ``constants.MSK_MODELS``.
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


@functools.lru_cache(maxsize=None)
def joint_names(model_id: str) -> tuple[str, ...]:
    """Return `model_id`'s non-base joint names in qpos order.

    This is the model-generic counterpart of ``constants.JOINT_NAMES``, which covers
    only the G1 robots. Its length is the ``dof`` of the canonical qpos layout
    ``[base(7), joints(DOF)]`` — i.e. the number in ``qpos_<dof>dof.csv``.

    The G1 tuples are returned verbatim: they are copied from unitree_rl_mjlab's
    ``csv_to_npz.py`` and are the wire contract shared with mj-nlp, so they are never
    re-derived. Every other model (the MSK family) has no stored ordering and is read
    off its compiled model instead.

    Compiling an MSK XML is slow enough to be worth caching, hence ``lru_cache``.

    Raises:
        ValueError: if the model is not a single free base followed by 1-DOF joints.
    """
    if model_id in constants.JOINT_NAMES:
        return constants.JOINT_NAMES[model_id]

    model = load_model(model_id)
    order = sorted(range(model.njnt), key=lambda i: model.jnt_qposadr[i])
    free = [i for i in order if model.jnt_type[i] == mujoco.mjtJoint.mjJNT_FREE]
    if len(free) != 1 or model.jnt_qposadr[free[0]] != 0:
        raise ValueError(
            f"{model_id!r}: expected exactly one free joint at qpos[0:7]; got "
            f"{len(free)} free joint(s). The canonical layout is [base(7), joints(DOF)]."
        )
    return tuple(
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        for i in order
        if i != free[0]
    )


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
