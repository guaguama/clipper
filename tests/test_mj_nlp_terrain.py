"""Motion/terrain round trips for the mj-nlp cropper output."""

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from clipper import edits
from clipper.model import load_model
from clipper.qpos import enforce_joint_equalities, import_equality_joints
from clipper.sources.mj_nlp import load
from clipper.terrain import Terrain, TerrainBox
from clipper.trajectory import Trajectory
from clipper.writers.mj_nlp import write


class MjNlpTerrainTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model_id = "osl_fullbody"
        cls.model = load_model(cls.model_id)

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.out_dir = Path(self.tmp.name)
        self.terrain = Terrain(
            boxes=(
                TerrainBox((1, 2, 0.2), (0.3, 0.5, 0.2), yaw=0.8, name="terrain_box_0"),
                TerrainBox((8, 9, 0.4), (1, 0.5, 0.4), yaw=-0.2, pitch=-0.3,
                           name="terrain_box_1"),
            ),
            provenance={"source": "fit_terrain_from_motion", "fps": 100,
                        "parameters": {"margin": [0.1, 0.05]}},
        )
        qpos = np.tile(self.model.qpos0, (9, 1))
        qpos[:, :3] = np.arange(9)[:, None] * [0.1, 0.2, 0.01] + [1, 2, 0.9]
        # A solved spine that differs from default equalities must survive reload.
        qpos[:, self.model.joint("flex_extension").qposadr[0]] = 0.75
        qpos[:, self.model.joint("L1_L2_FE").qposadr[0]] = -0.31
        # Knees as every clipper export has them: coupled (qpos0's are not).
        qpos = enforce_joint_equalities(
            self.model, qpos, joint_names=import_equality_joints(self.model, self.model_id))
        self.traj = Trajectory(qpos, 100, self.model_id, "terra", "stairs", terrain=self.terrain)

    def test_crop_and_pose_edits_round_trip_with_all_obstacles(self):
        traj = edits.crop(self.traj, 2, 7)
        traj = edits.apply_height_offset(traj, 0.03)
        traj = edits.apply_ankle_offset(traj, self.model.joint(
            "OSL_KA_L1_osl_ankle_angle_r").qposadr[0], 2)
        folder = write(traj, out_dir=self.out_dir)
        # JSON matches TERRA's schema, including the box far outside the crop.
        expected = json.loads(json.dumps(self.terrain.to_dict()))
        self.assertEqual(json.loads((folder / "terrain.json").read_text()), expected)
        for path in (folder, folder / "qpos_73dof.csv"):
            with self.subTest(path=path):
                loaded = load(path, self.model, self.model_id)
                np.testing.assert_array_equal(loaded.qpos, traj.qpos)
                self.assertAlmostEqual(loaded.fps, 100)
                self.assertEqual(loaded.terrain, self.terrain)
        np.testing.assert_allclose(np.loadtxt(folder / "time.csv", delimiter=","),
                                   np.arange(6) / 100)

    def test_resampling_changes_only_motion_and_time(self):
        folder = write(self.traj, out_dir=self.out_dir, output_fps=50)
        loaded = load(folder, self.model, self.model_id)
        np.testing.assert_allclose(loaded.qpos, self.traj.qpos[:8:2], atol=1e-15)
        self.assertAlmostEqual(loaded.fps, 50)
        self.assertEqual(loaded.terrain, self.terrain)

    def test_legacy_absence_and_explicit_flat_are_distinct(self):
        for terrain in (None, Terrain(provenance={"source": "flat"})):
            with self.subTest(terrain=terrain):
                self.traj.terrain = terrain
                folder = write(self.traj, out_dir=self.out_dir)
                self.assertEqual((folder / "terrain.json").exists(), terrain is not None)
                self.assertEqual(load(folder, self.model, self.model_id).terrain, terrain)

    def test_invalid_sidecar_is_not_ignored(self):
        folder = write(self.traj, out_dir=self.out_dir)
        (folder / "terrain.json").write_text('{"boxes": [{"pos": [0, 0, 1], "size": [1, -1, 1]}]}')
        with self.assertRaisesRegex(ValueError, "terrain.json.*box 0"):
            load(folder, self.model, self.model_id)

    def test_overwrite_false_protects_motion_and_sidecar(self):
        folder = write(self.traj, out_dir=self.out_dir)
        before = {path.name: path.read_bytes() for path in folder.iterdir()}
        with self.assertRaises(FileExistsError):
            write(self.traj, out_dir=self.out_dir, overwrite=False)
        self.assertEqual(before, {path.name: path.read_bytes() for path in folder.iterdir()})
        # A leftover sidecar alone also counts as an existing export.
        (folder / "qpos_73dof.csv").unlink()
        (folder / "time.csv").unlink()
        with self.assertRaises(FileExistsError):
            write(self.traj, out_dir=self.out_dir, overwrite=False)
        self.assertFalse((folder / "qpos_73dof.csv").exists())

    def test_motion_only_overwrite_removes_old_obstacles(self):
        folder = write(self.traj, out_dir=self.out_dir)
        self.traj.terrain = None
        write(self.traj, out_dir=self.out_dir)
        self.assertFalse((folder / "terrain.json").exists())
        self.assertIsNone(load(folder, self.model, self.model_id).terrain)


if __name__ == "__main__":
    unittest.main()
