"""Trajectory source loaders, dispatched by the clipper `--source` value.

Each loader has the signature
``load(path, model, model_id, source=..., fps=None) -> Trajectory`` and returns a
`Trajectory` whose qpos is already in the target model's layout.
"""

from __future__ import annotations

from typing import Callable

from ..trajectory import Trajectory
from . import bones_seed, locomujoco

# DefaultDatasets and Lafan1 share the LocoMuJoCo npz format/loader.
SOURCE_LOADERS: dict[str, Callable[..., Trajectory]] = {
    "default_datasets": locomujoco.load,
    "lafan1": locomujoco.load,
    "bones_seed": bones_seed.load,
}


def get_loader(source: str) -> Callable[..., Trajectory]:
    """Return the loader function for a `--source` value."""
    try:
        return SOURCE_LOADERS[source]
    except KeyError:
        raise ValueError(
            f"Unknown source {source!r}; expected one of {tuple(SOURCE_LOADERS)}."
        ) from None


__all__ = ["SOURCE_LOADERS", "get_loader"]
