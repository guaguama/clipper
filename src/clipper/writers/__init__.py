"""Target-format writers for clipper trajectories.

Mirrors the `sources/` loader registry: each writer is keyed by output format and
exposes ``write(traj, ...) -> Path``.
"""

from __future__ import annotations

from typing import Callable

from . import mj_nlp, unitree_rl_mjlab

WRITERS: dict[str, Callable[..., object]] = {
    "unitree_rl_mjlab": unitree_rl_mjlab.write,
    "mj_nlp": mj_nlp.write,
}


def get_writer(fmt: str) -> Callable[..., object]:
    """Return the writer function for a `--format` value."""
    try:
        return WRITERS[fmt]
    except KeyError:
        raise ValueError(
            f"Unknown format {fmt!r}; expected one of {tuple(WRITERS)}."
        ) from None


__all__ = ["WRITERS", "get_writer"]
