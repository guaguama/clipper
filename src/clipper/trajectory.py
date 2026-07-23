"""Canonical in-memory trajectory format used across clipper.

Per the clipper design, the canonical format is intentionally minimal: a qpos
array already laid out for the target MuJoCo model, plus frame rate and
provenance. Velocities and body kinematics are NOT stored here — they are
derived lazily by output writers that need them (e.g. the unitree NPZ writer).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Trajectory:
    """A reference motion in MuJoCo qpos form for a specific G1 model.

    Attributes:
        qpos: (N, nq) float array in the target model's qpos layout —
            ``[base_pos(3), base_quat_wxyz(4), joints(DOF)]``, radians, meters.
        fps: Playback frame rate in Hz.
        model: Target model id, ``"g1_29dof"`` or ``"g1_23dof"``.
        source: Provenance source id (e.g. ``"lafan1"``, ``"bones_seed"``).
        name: Short name of the motion (typically the source file stem).
        markers: Optional ``(N, K, 3)`` world-frame overlay points (e.g. the source
            mocap skeleton a retarget tracked). Rendered as spheres in visualization
            for inspection; ``None`` for sources that carry no overlay. Purely a
            visual debug aid — output writers never read it.
    """

    qpos: np.ndarray
    fps: float
    model: str
    source: str
    name: str
    markers: np.ndarray | None = None

    @property
    def num_frames(self) -> int:
        return int(self.qpos.shape[0])

    @property
    def duration(self) -> float:
        """Duration in seconds at the trajectory's frame rate."""
        return self.num_frames / self.fps

    def __post_init__(self) -> None:
        self.qpos = np.asarray(self.qpos, dtype=np.float64)
        if self.qpos.ndim != 2:
            raise ValueError(f"qpos must be (N, nq); got shape {self.qpos.shape}")
        if self.markers is not None:
            self.markers = np.asarray(self.markers, dtype=np.float64)
            if self.markers.ndim != 3 or self.markers.shape[0] != self.qpos.shape[0]:
                raise ValueError(
                    f"markers must be (N, K, 3) aligned with qpos frames "
                    f"(N={self.qpos.shape[0]}); got shape {self.markers.shape}"
                )
