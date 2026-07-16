"""Trajectory source loaders, dispatched by the clipper `--source` value.

Each loader has the signature
``load(path, model, model_id, source=..., fps=None) -> Trajectory`` and returns a
`Trajectory` whose qpos is already in the target model's layout.
"""

from __future__ import annotations

from typing import Callable

from ..trajectory import Trajectory
from . import bones_seed, locomujoco, mj_nlp, mj_nlp_out, musclemimic, srb_kino, unilab

# DefaultDatasets and Lafan1 share the LocoMuJoCo npz format/loader.
SOURCE_LOADERS: dict[str, Callable[..., Trajectory]] = {
    "default_datasets": locomujoco.load,
    "lafan1": locomujoco.load,
    "bones_seed": bones_seed.load,
    # clipper's own unitree_rl_mjlab output, re-loadable for viz/re-crop.
    "unilab": unilab.load,
    # mj-nlp (sim-nlp) qpos/time CSV folder, re-loadable for viz/re-crop.
    "mj_nlp": mj_nlp.load,
    # mj-nlp task-output npz (solved MPC rollout `state`), read-only import.
    "mj_nlp_out": mj_nlp_out.load,
    # headerless qpos CSV (xyzw quat, no timing) fed to unitree_rl_mjlab csv_to_npz.
    "srb_kino": srb_kino.load,
    # musclemimic retargeted caches + clipper's round-trip output (self-describing).
    "musclemimic": musclemimic.load,
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
