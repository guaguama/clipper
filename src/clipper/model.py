"""Load a MuJoCo G1 model for visualization.

Uses the *bare* robot XMLs (clean ``nq = 7 + DOF`` for both models), which avoids
the placeholder dummy bodies in ``scene_g1_23dof.xml`` (parked at z=20). When a
ground plane is wanted, it is added programmatically via ``MjSpec`` so the qpos
layout stays identical across models.
"""

from __future__ import annotations

import mujoco
import numpy as np

from . import constants


def load_model(model_id: str, floor: bool = True) -> mujoco.MjModel:
    """Compile and return the MuJoCo model for `model_id`.

    Args:
        model_id: ``"g1_29dof"`` or ``"g1_23dof"``.
        floor: If True, add a ground-plane geom and a light to the world.

    Returns:
        Compiled ``mujoco.MjModel``. qpos layout is ``[base(7), joints(DOF)]``.
    """
    if model_id not in constants.ROBOT_XML:
        raise ValueError(
            f"Unknown model {model_id!r}; expected one of {constants.MODELS}."
        )
    xml_path = str(constants.ROBOT_XML[model_id])

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
