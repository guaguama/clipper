"""TERRA pairing, coordinate conventions, and visualization regressions.

Run with ``python -m unittest discover -s tests``; no external datasets needed.
"""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import mujoco
import numpy as np

from clipper import edits
from clipper.sources import get_loader
from clipper.terrain import TerrainBox
from clipper.viz.overlays import draw_overlays


class TerraTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # A small OSL-named model with a coupled knee checks the shared remapper
        # without compiling the full vendored muscle model for each unit test.
        xml = """
          <mujoco><worldbody><body><freejoint name="root"/>
            <geom type="sphere" size="0.1"/>
            <body><geom type="sphere" size="0.1"/>
              <joint name="OSL_KA_L1_osl_knee_angle_r"/>
              <joint name="OSL_KA_L1_osl_ankle_angle_r"/>
              <joint name="elbow_flexion_r"/>
              <joint name="knee_angle_l"/>
              <joint name="knee_angle_rotation3_l"/>
              <body><geom type="sphere" size="0.1"/>
                <joint name="flex_extension"/>
                <joint name="L1_L2_FE"/>
                <joint name="sternoclavicular_r2_r"/>
              </body>
            </body>
          </body></worldbody><equality>
            <joint joint1="knee_angle_rotation3_l" joint2="knee_angle_l"
                   polycoef="0 0.5 0 0 0"/>
            <joint joint1="L1_L2_FE" joint2="flex_extension"
                   polycoef="0 0.255 0 0 0"/>
            <joint joint1="sternoclavicular_r2_r" joint2="elbow_flexion_r"
                   polycoef="0 0.1 0 0 0"/>
          </equality></mujoco>
        """
        cls.model = mujoco.MjModel.from_xml_string(xml)
        native_xml = xml.replace("OSL_KA_L1_osl_", "").replace("elbow_flexion_r", "elbow_flex_r")
        cls.native_model = mujoco.MjModel.from_xml_string(native_xml)

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "stairs.npz"
        self.sidecar = self.path.with_name("stairs_terrain.json")
        self.raw = np.array([
            [1, 2, 1.0, 1, 0, 0, 0, 0.2, -0.1, 0.6, 0.4, -0.2, 0.08, -0.31, -0.01],
            [2, 3, 1.2, 1, 0, 0, 0, 0.3, -0.2, 0.7, 0.6, -0.3, 0.34, -0.22, -0.02],
            [3, 4, 1.4, 1, 0, 0, 0, 0.4, -0.3, 0.8, 0.8, -0.4, 0.75, -0.31, -0.03],
        ], dtype=float)
        np.savez(
            self.path, qpos=self.raw, frequency=100.0,
            joint_names=["root", "knee_angle_r", "ankle_angle_r", "elbow_flex_r",
                         "knee_angle_l", "knee_angle_rotation3_l", "flex_extension",
                         "L1_L2_FE", "sternoclavicular_r2_r"],
            jnt_type=[0, 3, 3, 3, 3, 3, 3, 3, 3],
        )
        self.payload = {
            "boxes": [{"pos": [1, 2, 0.2], "size": [0.3, 0.5, 0.2],
                       "yaw": np.pi / 2, "pitch": -np.pi / 6,
                       "name": "terrain_box_0"}],
            "provenance": {"source": "test"},
        }
        self.sidecar.write_text(json.dumps(self.payload))

    def load(self, **kwargs):
        return get_loader("terra")(self.path, self.model, "osl_fullbody", **kwargs)

    def test_joint_aliases_couplers_and_world_frame(self):
        traj = self.load()
        self.assertEqual((traj.source, traj.fps, traj.num_frames), ("terra", 100.0, 3))
        np.testing.assert_array_equal(traj.qpos[:, :7], self.raw[:, :7])
        np.testing.assert_array_equal(traj.qpos[:, 7:11], self.raw[:, 7:11])
        # The cache follower had the wrong sign. The target polynomial wins.
        np.testing.assert_allclose(traj.qpos[:, 11], self.raw[:, 10] * 0.5)
        self.assertEqual(traj.terrain.provenance, self.payload["provenance"])
        self.assertEqual(self.load(fps=25).fps, 25)

    def test_both_sources_preserve_solved_spine_and_shoulders(self):
        traj = self.load()
        np.testing.assert_array_equal(traj.qpos[:, 12:], self.raw[:, 12:])
        cache = get_loader("musclemimic")(self.path, self.model, "osl_fullbody")
        np.testing.assert_array_equal(cache.qpos, traj.qpos)
        self.assertIsNone(cache.terrain)
        # Knee adaptation still runs through either source.
        np.testing.assert_allclose(cache.qpos[:, 11], self.raw[:, 10] * 0.5)

    def test_native_myofullbody_preserves_every_solved_coordinate(self):
        for source in ("terra", "musclemimic"):
            with self.subTest(source=source):
                traj = get_loader(source)(self.path, self.native_model, "myofullbody")
                np.testing.assert_array_equal(traj.qpos, self.raw)

    def test_explicit_equality_override(self):
        from clipper.sources.musclemimic import load

        untouched = load(self.path, self.model, "osl_fullbody", equality_joints=())
        np.testing.assert_array_equal(untouched.qpos, self.raw)
        selected = load(self.path, self.model, "osl_fullbody", equality_joints=("L1_L2_FE",))
        np.testing.assert_allclose(selected.qpos[:, 13], self.raw[:, 12] * 0.255)
        np.testing.assert_array_equal(selected.qpos[:, 11], self.raw[:, 11])

    def test_myolegs_still_fills_missing_knee_followers(self):
        from clipper.sources.musclemimic import load

        # The amputee-leg cache has no biological right-knee follower. Reuse
        # the tiny model's coupled left knee to exercise the same fill policy.
        reduced = self.raw[:, [*range(11), 12, 13, 14]]
        np.savez(self.path, qpos=reduced, frequency=100.0,
                 joint_names=["root", "knee_angle_r", "ankle_angle_r", "elbow_flex_r",
                              "knee_angle_l", "flex_extension", "L1_L2_FE",
                              "sternoclavicular_r2_r"], jnt_type=[0, 3, 3, 3, 3, 3, 3, 3])
        traj = load(self.path, self.native_model, "myolegs")
        np.testing.assert_allclose(traj.qpos[:, 11], self.raw[:, 10] * 0.5)
        np.testing.assert_array_equal(traj.qpos[:, 12:], self.raw[:, 12:])

    def test_missing_sidecar_cannot_silently_drop_stairs(self):
        self.sidecar.unlink()
        with self.assertRaisesRegex(FileNotFoundError, "stairs_terrain.json"):
            self.load()

    def test_explicit_terrain_loads_motion_without_a_local_sidecar(self):
        external = Path(self.tmp.name) / "shared stairs.json"
        external.write_text(json.dumps(self.payload))
        self.sidecar.unlink()
        traj = self.load(terrain=external)
        np.testing.assert_array_equal(traj.qpos, get_loader("musclemimic")(
            self.path, self.model, "osl_fullbody").qpos)
        self.assertEqual(traj.terrain.provenance, self.payload["provenance"])
        self.assertEqual(len(traj.terrain.boxes), 1)

    def test_explicit_terrain_overrides_local_sidecar(self):
        external = Path(self.tmp.name) / "flat.json"
        external.write_text('{"boxes": [], "provenance": {"source": "override"}}')
        traj = self.load(terrain=external)
        self.assertEqual(traj.terrain.boxes, ())
        self.assertEqual(traj.terrain.provenance, {"source": "override"})

    def test_invalid_or_missing_override_does_not_fall_back_to_sidecar(self):
        external = Path(self.tmp.name) / "other_terrain.json"
        with self.assertRaisesRegex(FileNotFoundError, "other_terrain.json"):
            self.load(terrain=external)
        external.write_text('{}')
        with self.assertRaisesRegex(ValueError, "other_terrain.json"):
            self.load(terrain=external)

    def test_explicit_flat_terrain(self):
        self.sidecar.write_text('{"boxes": []}')
        self.assertEqual(self.load().terrain.boxes, ())

    def test_invalid_terrain_reports_path_and_box(self):
        for bad_size in ([1, -1, 1], [1, 1], [1, float("nan"), 1]):
            with self.subTest(size=bad_size):
                self.payload["boxes"][0]["size"] = bad_size
                self.sidecar.write_text(json.dumps(self.payload))
                with self.assertRaisesRegex(ValueError, "stairs_terrain.json.*box 0"):
                    self.load()
        self.sidecar.write_text('{}')
        with self.assertRaisesRegex(ValueError, "boxes.*list"):
            self.load()

    def test_box_rotation_matches_mujoco_composed_quaternion(self):
        box = TerrainBox((1, 2, 3), (0.1, 0.2, 0.3), yaw=0.7, pitch=-0.3)
        qz, qy, quat, mat = np.zeros(4), np.zeros(4), np.zeros(4), np.zeros(9)
        mujoco.mju_axisAngle2Quat(qz, np.array([0., 0., 1.]), box.yaw)
        mujoco.mju_axisAngle2Quat(qy, np.array([0., 1., 0.]), box.pitch)
        mujoco.mju_mulQuat(quat, qz, qy)
        mujoco.mju_quat2Mat(mat, quat)
        np.testing.assert_allclose(box.rotation, mat.reshape(3, 3), atol=1e-15)

    def test_overlays_append_without_moving_terrain(self):
        traj = self.load()
        traj.markers = np.zeros((3, 1, 3))
        scn = mujoco.MjvScene(self.model, maxgeom=8)
        scn.ngeom = 1  # Existing model geom must survive appending overlays.
        scn.geoms[0].pos[:] = [9, 8, 7]
        draw_overlays(scn, traj, 1, z_offset=0.25)
        self.assertEqual(scn.ngeom, 3)
        np.testing.assert_allclose(scn.geoms[0].pos, [9, 8, 7])
        box_geom = scn.geoms[1]
        self.assertEqual(box_geom.type, mujoco.mjtGeom.mjGEOM_BOX)
        np.testing.assert_allclose(box_geom.pos, [1, 2, 0.2])
        # The wire format's size is already a half-extent, never halve again.
        np.testing.assert_allclose(box_geom.size, [0.3, 0.5, 0.2])
        np.testing.assert_allclose(box_geom.mat, traj.terrain.boxes[0].rotation, atol=1e-7)
        np.testing.assert_allclose(scn.geoms[2].pos, [0, 0, 0.25])
        # Passive viewers reset only user_scn, and redraw without accumulating.
        scn.ngeom = 0
        draw_overlays(scn, traj, 2)
        self.assertEqual(scn.ngeom, 2)
        np.testing.assert_allclose(scn.geoms[0].pos, [1, 2, 0.2])

    def test_scene_capacity_error_keeps_obstacles_from_disappearing(self):
        traj = self.load()
        scn = mujoco.MjvScene(self.model, maxgeom=1)
        scn.ngeom = 1
        with self.assertRaisesRegex(ValueError, "terrain boxes"):
            draw_overlays(scn, traj, 0)

    def test_edits_keep_terrain_fixed(self):
        original = self.load()
        traj = edits.crop(original, 1, 2)
        traj = edits.apply_height_offset(traj, 0.12)
        traj = edits.apply_ankle_offset(traj, 8, 5)
        with patch("clipper.edits._home_standing_qpos", return_value=self.model.qpos0):
            traj = edits.pad_standing(traj, 0.01, 0.01, 0.01, 0.01)
        self.assertIs(traj.terrain, original.terrain)
        self.assertEqual(traj.terrain.boxes[0].pos, (1, 2, 0.2))
        np.testing.assert_allclose(original.qpos[:, :7], self.raw[:, :7])


if __name__ == "__main__":
    unittest.main()
