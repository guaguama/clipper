"""Joint couplings on the two mj-nlp sources and the gait-library resampler."""

from pathlib import Path
import tempfile
import unittest

import numpy as np

from clipper import gait_library as gl
from clipper.model import load_model
from clipper.qpos import enforce_joint_equalities, import_equality_joints
from clipper.sources import mj_nlp, mj_nlp_out
from clipper.trajectory import Trajectory
from clipper.writers.mj_nlp import write as write_csv

MODEL_ID = "osl_fullbody"


def coupling_residual(model, qpos, names=None):
    """Largest |follower - poly(driver)| over the selected couplings (None: all)."""
    return float(np.abs(enforce_joint_equalities(model, qpos, joint_names=names) - qpos).max())


class MjNlpCouplingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = load_model(MODEL_ID)
        cls.knees = import_equality_joints(cls.model, MODEL_ID)
        cls.adr = lambda _, name: int(cls.model.joint(name).qposadr[0])

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = Path(self.tmp.name)
        m, n = self.model, 40
        t = np.arange(n) / 200.0
        q = np.tile(m.qpos0, (n, 1))
        q[:, 2] = 0.95
        q[:, self.adr("knee_angle_l")] = -0.6 * np.sin(2 * np.pi * t) ** 2
        q[:, self.adr("flex_extension")] = 0.3
        q = enforce_joint_equalities(m, q)
        # A solver's soft-coupling slack on the knee and the shoulder girdle, and an
        # independently solved spine (as TERRA's is).
        q[:, self.adr("knee_angle_beta_rotation1_l")] += 0.05
        q[:, self.adr("sternoclavicular_r2_l")] += 0.02
        q[:, self.adr("L1_L2_FE")] = -0.31
        self.raw, self.t = q, t

    def test_solver_output_solves_every_coupling(self):
        m, n = self.model, self.raw.shape[0]
        qvel = np.zeros((n, m.nv))
        qvel[:, m.joint("knee_angle_l").dofadr[0]] = 1.0
        path = self.dir / "node.npz"
        np.savez(path, state=np.hstack([self.raw, qvel]), time=self.t)
        traj = mj_nlp_out.load(path, m, MODEL_ID)
        self.assertLess(coupling_residual(m, traj.qpos), 1e-12)
        followers = {self.adr(nm) for nm in ("knee_angle_beta_rotation1_l",
                                             "sternoclavicular_r2_l", "L1_L2_FE")}
        changed = set(np.flatnonzero(np.abs(traj.qpos - self.raw).max(axis=0) > 0))
        self.assertTrue(followers <= changed)
        # Drivers and the root stay as solved.
        for nm in ("knee_angle_l", "flex_extension"):
            np.testing.assert_array_equal(traj.qpos[:, self.adr(nm)], self.raw[:, self.adr(nm)])
        np.testing.assert_array_equal(traj.qpos[:, :7], self.raw[:, :7])
        untouched = mj_nlp_out.load(path, m, MODEL_ID, equality_joints=())
        np.testing.assert_array_equal(untouched.qpos, self.raw)

    def test_couple_velocity_is_the_chain_rule(self):
        m = self.model
        q, _ = mj_nlp_out.couple(m, self.raw)
        knee_v = np.gradient(q[:, self.adr("knee_angle_l")], self.t)
        v = np.zeros((q.shape[0], m.nv))
        v[:, m.joint("knee_angle_l").dofadr[0]] = knee_v
        _, v = mj_nlp_out.couple(m, q, v)
        fd = np.gradient(q, self.t, axis=0)
        j = m.joint("knee_angle_rotation3_l")
        np.testing.assert_allclose(v[2:-2, j.dofadr[0]], fd[2:-2, j.qposadr[0]], atol=2e-3)

    def test_csv_reload_solves_knees_only(self):
        m = self.model
        folder = write_csv(Trajectory(self.raw, 200, MODEL_ID, "mj_nlp", "clip"), out_dir=self.dir)
        loaded = mj_nlp.load(folder, m, MODEL_ID)
        self.assertLess(coupling_residual(m, loaded.qpos, self.knees), 1e-12)
        knee = self.adr("knee_angle_beta_rotation1_l")
        others = [a for a in range(m.nq) if a != knee]
        np.testing.assert_allclose(loaded.qpos[:, others], self.raw[:, others], atol=1e-15)
        self.assertGreater(coupling_residual(m, loaded.qpos), 0.019)  # shoulder kept
        np.testing.assert_allclose(
            mj_nlp.load(folder, m, MODEL_ID, equality_joints=()).qpos, self.raw, atol=1e-15)

    def test_resampled_gait_clip_stays_coupled(self):
        """A lerp between coupled 200 Hz samples is not coupled; the resampler re-solves."""
        m = self.model
        q, _ = mj_nlp_out.couple(m, self.raw)
        n = q.shape[0]
        q = np.vstack([q, q[:1]])  # periodic: closing state = frame 0, zero stride
        dq = np.gradient(q, 1 / 200.0, axis=0)
        v = np.zeros((n + 1, m.nv))
        v[:, 6:] = dq[:, 7:]
        node = gl.Node(path=Path("walk_fwd_vx+00000_vy+00000_wz+00000.npz"), qpos=q, qvel=v,
                       dt=1 / 200.0, twist=np.zeros(3), stride=np.eye(3))
        for velocity in ("solver", "central"):
            clip = gl.resample_node(node, MODEL_ID, 30.0, t0=0.0123, velocity=velocity)
            qadr, _ = gl._joint_columns(m, MODEL_ID)
            full = np.tile(m.qpos0, (clip.n, 1))
            full[:, qadr] = clip.joint_pos
            with self.subTest(velocity=velocity):
                self.assertLess(coupling_residual(m, full), 1e-12)


if __name__ == "__main__":
    unittest.main()
