#!/usr/bin/env python
"""Convert an mj-nlp gait library into a looping, phase-aligned mjlab library.

Each node (one periodic stride) becomes one mjlab motion NPZ carrying its ``twist``
label, written into a directory of its own: the consuming loader globs ``*.npz``.
See ``clipper/gait_library.py`` for why this is not ``crop_trajectory.py --format
mjlab`` (exact-period frame count, solver velocities, normalised-phase alignment).

Examples:
    # The 21-node forward-walk library + a standing clip
    python scripts/convert_gait_library.py \
        ../mj-nlp/examples/osl_mimic/clip_search/gait_library_walk_fwd_multi \
        --model osl_fullbody --idle

    # A later library (e.g. arcs), aligned to the same phase 0 as the walk library
    python scripts/convert_gait_library.py <arc library dir> --model osl_fullbody \
        --align-to outputs/mjlab/osl_fullbody/gait_library_walk_fwd_multi/walk_fwd_vx+01000_vy+00000_wz+00000.npz
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from clipper import constants
from clipper import gait_library as gl

SUPPORTED_MODELS = sorted(set(constants.FWD_AXIS_BODY) & set(constants.FOOT_CONTACT_BODIES))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("library", type=Path, help="mj-nlp gait-library folder.")
    p.add_argument(
        "--model", required=True, choices=SUPPORTED_MODELS,
        help="Models with a forward axis (FWD_AXIS_BODY) and foot bodies "
        "(FOOT_CONTACT_BODIES), both needed for alignment, re-base and --idle.",
    )
    p.add_argument(
        "--glob", default="*_vx*.npz",
        help="Node files inside the library (seed_*.npz and index.json are skipped).",
    )
    p.add_argument(
        "--out-dir", type=Path, default=None,
        help="Output folder. Default: outputs/mjlab/<model>/<library name>.",
    )
    p.add_argument(
        "--fps", type=float, default=constants.MJLAB_MOTION_FPS,
        help="Output rate; must equal the env's 1/step_dt.",
    )
    p.add_argument(
        "--align-to", type=Path, default=None,
        help="Converted mjlab NPZ whose phase is phase 0. Default: this library's "
        "straight node nearest its mid forward speed.",
    )
    p.add_argument("--no-align", action="store_true", help="Skip phase alignment.")
    p.add_argument("--idle", action="store_true", help="Also write a standing clip.")
    p.add_argument("--idle-frames", type=int, default=50)
    p.add_argument("--idle-keyframe", default="stand")
    p.add_argument(
        "--overwrite", action="store_true",
        help="Delete *.npz already in the output folder first.",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    files = sorted(
        f for f in args.library.glob(args.glob) if not f.name.startswith("seed_")
    )
    if not files:
        print(f"no nodes matching {args.glob!r} in {args.library}", file=sys.stderr)
        return 1
    out_dir = args.out_dir or (
        constants.REPO_ROOT / "outputs" / "mjlab" / args.model / args.library.name
    )
    old = sorted(out_dir.glob("*.npz")) if out_dir.exists() else []
    if old and not args.overwrite:
        print(
            f"{out_dir} already holds {len(old)} npz; pass --overwrite to replace them "
            "(the loader globs every *.npz there).",
            file=sys.stderr,
        )
        return 1

    clips = [gl.load_node(f, args.model, args.fps) for f in files]
    print(f"loaded {len(clips)} nodes from {args.library}")

    if not args.no_align:
        ref = gl.profile_from_npz(args.align_to, args.model) if args.align_to else None
        if ref is None:
            print(f"phase reference: {clips[gl.reference_index(clips)].name}")
        else:
            print(f"phase reference: {args.align_to}")
        report = gl.align_library(clips, args.model, ref)
    else:
        report = [{"name": c.name, "lag": np.nan, "shift": 0, "residual": np.nan} for c in clips]

    for c in clips:
        gl.rebase(c, args.model)

    print(
        f"{'node':40s} {'N':>3s} {'T[s]':>5s} {'kappa':>6s} {'lag%':>6s} {'shift':>5s} "
        f"{'resid%':>6s}  seam/interior-max: jpos jvel root linv angv"
    )
    bad = []
    for c, r in zip(clips, report):
        seam = gl.seam_report(c)
        ratios = {k: s / max(i, 1e-9) for k, (s, i) in seam.items()}
        if any(v > gl.SEAM_TOL for v in ratios.values()):
            bad.append(c.name)
        print(
            f"{c.name:40s} {c.n:3d} {c.stride_T:5.3f} {c.kappa:6.4f} "
            f"{100 * r['lag']:6.1f} {r['shift']:5d} {100 * r['residual']:6.2f}  "
            + " ".join(f"{ratios[k]:4.2f}" for k in ratios)
        )
    if bad:
        print(f"SEAM CHECK FAILED (> {gl.SEAM_TOL}x interior): {bad}", file=sys.stderr)
        return 2

    out_dir.mkdir(parents=True, exist_ok=True)
    for f in old:
        f.unlink()
    for c in clips:
        gl.write_clip(c, out_dir, args.fps)
    if args.idle:
        idle, info = gl.idle_clip(args.model, args.idle_frames, args.idle_keyframe)
        gl.write_clip(idle, out_dir, args.fps)
        gaps = ", ".join(f"{k} {1000 * v:.1f} mm" for k, v in info["foot_gap"].items())
        print(f"idle: keyframe {args.idle_keyframe!r}, dz {info['dz']:+.4f} m, foot gap {gaps}")
    print(f"wrote {len(clips) + int(args.idle)} clips to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
