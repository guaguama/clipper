"""Static box terrain in the trajectory's world frame.

Reads TERRA's plain TerrainSpec sidecar without importing TERRA, MuscleMimic or
JAX. Positions are box centres in metres, sizes are MuJoCo half-extents, and
orientation is Rz(yaw) @ Ry(pitch), with both angles in radians.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class TerrainBox:
    pos: tuple[float, float, float]
    size: tuple[float, float, float]
    yaw: float = 0.0
    pitch: float = 0.0
    name: str = ""

    def __post_init__(self) -> None:
        for key in ("pos", "size"):
            value = np.asarray(getattr(self, key), dtype=np.float64)
            if value.shape != (3,) or not np.all(np.isfinite(value)):
                raise ValueError(f"box {key} must contain three finite values")
            object.__setattr__(self, key, tuple(float(v) for v in value))
        if any(v <= 0 for v in self.size):
            raise ValueError("box size must contain positive half-extents")
        for key in ("yaw", "pitch"):
            value = float(getattr(self, key))
            if not np.isfinite(value):
                raise ValueError(f"box {key} must be finite")
            object.__setattr__(self, key, value)

    @property
    def rotation(self) -> np.ndarray:
        """World-from-box matrix, matching TERRA's BoxSpec.rotation."""
        cy, sy = np.cos(self.yaw), np.sin(self.yaw)
        cp, sp = np.cos(self.pitch), np.sin(self.pitch)
        return np.array([
            [cy * cp, -sy, cy * sp],
            [sy * cp, cy, sy * sp],
            [-sp, 0.0, cp],
        ])


@dataclass(frozen=True)
class Terrain:
    """Boxes above the scene's z=0 floor, in TERRA's plain TerrainSpec schema.

    Names, geometry and reconstruction provenance survive export unchanged.
    Friction and contact solver settings belong to the consuming experiment.
    """

    boxes: tuple[TerrainBox, ...] = ()
    provenance: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Return the shared JSON representation (sizes are half-extents)."""
        return {
            "boxes": [asdict(box) for box in self.boxes],
            "provenance": dict(self.provenance),
        }

    def save(self, path: str | Path) -> None:
        """Write geometry and original provenance without changing coordinates."""
        payload = json.dumps(self.to_dict(), indent=2, allow_nan=False) + "\n"
        Path(path).write_text(payload)

    @classmethod
    def load(cls, path: str | Path) -> Terrain:
        path = Path(path)
        try:
            value = json.loads(path.read_text())
            if not isinstance(value, dict) or not isinstance(value.get("boxes"), list):
                raise ValueError("expected a TerrainSpec object with a 'boxes' list")
            boxes = []
            for i, box in enumerate(value["boxes"]):
                try:
                    boxes.append(TerrainBox(**box))
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"box {i}: {exc}") from exc
            return cls(tuple(boxes), dict(value.get("provenance", {})))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{path}: invalid terrain: {exc}") from exc
