#!/usr/bin/env python
"""Visualize a reference trajectory from any clipper source in MuJoCo.

Examples:
    python scripts/visualize_trajectory.py \
        --path ~/.g1mocap/DefaultDatasets/walk.npz \
        --source default_datasets --model g1_23dof

    python scripts/visualize_trajectory.py \
        --path <bones_seed.csv> --source bones_seed --model g1_29dof
"""

from __future__ import annotations

import argparse
from pathlib import Path

from clipper import constants
from clipper.model import load_model
from clipper.sources import get_loader
from clipper.viz import replay, scrub


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Visualize a reference trajectory in MuJoCo.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--path", required=True, type=Path, help="Trajectory file.")
    p.add_argument(
        "--source",
        required=True,
        choices=constants.SOURCES,
        help="Source format (no autodetection).",
    )
    p.add_argument(
        "--model",
        required=True,
        choices=constants.MODELS,
        help="Target G1 model.",
    )
    p.add_argument(
        "--mode",
        default="replay",
        choices=("replay", "scrub"),
        help="replay = autoplay; scrub = interactive slider + keyboard control.",
    )
    p.add_argument(
        "--fps",
        type=float,
        default=None,
        help="Playback rate override; defaults to the source's own fps "
        "(npz frequency, or 120 Hz for bones_seed).",
    )
    p.add_argument("--speed", type=float, default=1.0, help="Real-time multiplier.")
    p.add_argument("--no-loop", action="store_true", help="Play once, then stop.")
    p.add_argument("--no-floor", action="store_true", help="Hide the ground plane.")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    if not args.path.exists():
        raise SystemExit(f"Trajectory file not found: {args.path}")

    model = load_model(args.model, floor=not args.no_floor)
    loader = get_loader(args.source)
    traj = loader(
        args.path, model, args.model, source=args.source, fps=args.fps
    )
    if args.mode == "scrub":
        scrub(model, traj)
    else:
        replay(model, traj, fps=args.fps, loop=not args.no_loop, speed=args.speed)


if __name__ == "__main__":
    main()
