#!/usr/bin/env python
"""Play a looping gait library live, driven by a twist command (sliders / joystick).

The command [vx, vy, wz] snaps to the nearest library node, with each axis weighted by
1/span^2. The clip loops continuously, with each stride chained onto the last, and
switches at the same normalised phase with a crossfade. This mirrors what sesteban's
`LibraryMotionCommand` feeds a policy. The slider and stick ranges are the library's
own extremes (widened to include 0). See `clipper/viz/library_player.py` and
`clipper/viz/library_view.py`.

Examples:
    # The converted forward-walk library (convert_gait_library.py output)
    python scripts/visualize_gait_library.py \
        outputs/mjlab/osl_fullbody/gait_library_walk_fwd_multi \
        --source mjlab --model osl_fullbody --joystick

    # Raw mj-nlp libraries, converted and phase-aligned in memory (forward + arcs)
    python scripts/visualize_gait_library.py \
        ../mj-nlp/examples/osl_mimic/clip_search/gait_library_walk_fwd_multi \
        ../mj-nlp/examples/osl_mimic/turn_search/gait_library_walk_arc_multi \
        --source mj_nlp_out --model osl_fullbody --idle
"""

from __future__ import annotations

import argparse
from pathlib import Path

from clipper import constants
from clipper.model import load_model
from clipper.viz.joystick import Joystick
from clipper.viz.library_player import LibraryPlayer, load_library
from clipper.viz.library_view import view_library

SUPPORTED_MODELS = sorted(set(constants.FWD_AXIS_BODY) & set(constants.FOOT_CONTACT_BODIES))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "library", type=Path, nargs="+",
        help="Library folder(s). Several are combined; a repeated twist keeps the first.",
    )
    p.add_argument(
        "--source", required=True, choices=("mjlab", "mj_nlp_out"),
        help="mjlab: converted folders (convert_gait_library.py). mj_nlp_out: raw "
        "mj-nlp folders, converted + phase-aligned in memory.",
    )
    p.add_argument("--model", required=True, choices=SUPPORTED_MODELS)
    p.add_argument(
        "--fps", type=float, default=constants.MJLAB_MOTION_FPS,
        help="Clip frame rate (mjlab clips are 50 Hz; mj_nlp_out resamples to it).",
    )
    p.add_argument("--speed", type=float, default=1.0, help="Playback speed factor.")
    p.add_argument(
        "--blend", type=float, default=0.4,
        help="Crossfade on a clip switch (s); 0 = hard switch.",
    )
    p.add_argument(
        "--treadmill", action="store_true",
        help="Pin the root xy at the origin instead of travelling (camera fixed).",
    )
    p.add_argument(
        "--idle", action="store_true",
        help="mj_nlp_out only: add the model's standing clip as the zero-twist node.",
    )
    p.add_argument(
        "--glob", default="*_vx*.npz", help="mj_nlp_out only: node files per folder.",
    )
    p.add_argument(
        "--joystick", nargs="?", const="/dev/input/js0", default=None, metavar="DEV",
        help="Drive with a joystick (Linux js device; bare flag = /dev/input/js0).",
    )
    p.add_argument("--deadzone", type=float, default=0.12, help="Joystick stick deadzone.")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    print(f"loading {args.source} library for {args.model}:")
    clips = load_library(
        args.library, args.source, args.model, fps=args.fps, idle=args.idle, glob=args.glob,
    )
    player = LibraryPlayer(clips, args.model, fps=args.fps, blend_s=args.blend)
    js = Joystick.open(args.joystick) if args.joystick else None
    view_library(
        load_model(args.model), player, args.model,
        speed=args.speed, treadmill=args.treadmill, joystick=js, dz=args.deadzone,
        title=" + ".join(p.name for p in args.library),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
