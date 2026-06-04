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

    # The bare XML has only a dim COM-tracking light and relies on MuJoCo's
    # default headlight (diffuse 0.4) -> the scene renders dark. Match the
    # known-good scene_g1 lighting: a brighter headlight + a directional light.
    spec.visual.headlight.ambient = [0.4, 0.4, 0.4]
    spec.visual.headlight.diffuse = [0.7, 0.7, 0.7]
    spec.visual.headlight.specular = [0.3, 0.3, 0.3]

    sun = spec.worldbody.add_light()
    sun.type = mujoco.mjtLightType.mjLIGHT_DIRECTIONAL
    sun.pos = np.array([1.0, 0.0, 3.5])
    sun.dir = np.array([0.0, 0.0, -1.0])
    sun.castshadow = True

    # Checker-textured ground plane for a clear sense of height/translation.
    spec.add_texture(
        name="groundplane",
        type=mujoco.mjtTexture.mjTEXTURE_2D,
        builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
        rgb1=[0.2, 0.3, 0.4],
        rgb2=[0.1, 0.2, 0.3],
        width=300,
        height=300,
    )
    spec.add_material(
        name="groundplane",
        textures=["", "groundplane"],  # slot 1 = 2D texture
        texrepeat=[5, 5],
        reflectance=0.2,
    )
    floor_geom = spec.worldbody.add_geom()
    floor_geom.name = "floor"
    floor_geom.type = mujoco.mjtGeom.mjGEOM_PLANE
    floor_geom.size = np.array([0.0, 0.0, 0.05])
    floor_geom.material = "groundplane"

    return spec.compile()
