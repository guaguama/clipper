#!/usr/bin/env python
"""Convert an mj-nlp gait-library release into a looping mjlab release.

Follows the gait library scheme: the input is one flat release folder
(``<name>_vNNN/`` with ``library.json``, ``idle.npz`` and one
``<family>_vx±NNNNN_vy±NNNNN_wz±NNNNN.npz`` node per twist), and the output is a folder
with the same name and the same file names in mjlab's motion format, plus the
``library.json`` copied over with a ``clipper`` block added. mjlab trains on it as is (it
globs ``*.npz``). Per node, in order:

  a. check that the loop closes, and stop, naming every node that fails;
  b./c. resample to exactly round(50 T) frames starting at the OSL touchdown;
  d. re-base (root at the origin, stride-mean heading along +x), then FK;
  e. write it under the same name;
  f. add the clipper block to library.json.

``idle.npz`` becomes 50 identical frames of the solved stance with zero velocities.
Folders without a ``library.json`` (mj-nlp working folders from before the scheme) still
convert, with a warning, and a manifest is started for them. See
``clipper/gait_library.py`` for why this is not ``crop_trajectory.py --format mjlab``.

Examples:
    # A release -> outputs/mjlab/osl_fullbody/kit_walk_fwd_v001/
    python scripts/convert_gait_library.py \
        ../mj-nlp/examples/osl_mimic/libraries/kit_walk_fwd_v001 --model osl_fullbody

    # A legacy working folder, with the model's `stand` keyframe as the idle
    python scripts/convert_gait_library.py \
        ../mj-nlp/examples/osl_mimic/clip_search/gait_library_walk_fwd_multi \
        --model osl_fullbody --idle
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from clipper import constants
from clipper import gait_library as gl

SUPPORTED_MODELS = sorted(
    set(constants.FWD_AXIS_BODY)
    & set(constants.FOOT_CONTACT_BODIES)
    & set(constants.TOUCHDOWN_FOOT)
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("library", type=Path, help="mj-nlp release (or legacy library) folder.")
    p.add_argument(
        "--model", required=True, choices=SUPPORTED_MODELS,
        help="Models with a forward axis (FWD_AXIS_BODY), foot bodies "
        "(FOOT_CONTACT_BODIES) and a touchdown foot (TOUCHDOWN_FOOT).",
    )
    p.add_argument(
        "--out-dir", type=Path, default=None,
        help="Output folder. Default: outputs/mjlab/<model>/<release name>.",
    )
    p.add_argument(
        "--fps", type=float, default=constants.MJLAB_MOTION_FPS,
        help="Output rate; must equal the env's 1/step_dt.",
    )
    p.add_argument(
        "--phase", choices=gl.PHASES, default="touchdown",
        help="Phase 0 of every clip. touchdown: the OSL foot touchdown (the scheme). "
        "xcorr: foot-height cross-correlation against a reference node (pre-scheme). "
        "none: the node's own frame 0.",
    )
    p.add_argument(
        "--align-to", type=Path, default=None,
        help="--phase xcorr only: converted mjlab NPZ whose phase is phase 0. Default: "
        "the straight node nearest the library's mid forward speed.",
    )
    p.add_argument(
        "--velocity", choices=gl.VELOCITIES, default="solver",
        help="qvel source (body velocities follow by FK). solver: mj-nlp's qvel x kappa. "
        "central: musclemimic's convention (its loader recomputes central differences). "
        "forward: d(q[k] -> q[k+1]). Both differences at --fps, wrapped through the loop.",
    )
    p.add_argument(
        "--glob", default=None,
        help="Legacy folders only (no library.json): node files. Default '*_vx*.npz'.",
    )
    p.add_argument(
        "--idle", action="store_true",
        help="Legacy folders without idle.npz: write the --idle-keyframe pose as idle.",
    )
    p.add_argument(
        "--idle-node", type=Path, default=None,
        help="Legacy folders without idle.npz: an mj-nlp idle node file to hold as idle "
        "(e.g. bones_search/seeds/idle__idle_median_lev.npz).",
    )
    p.add_argument("--idle-frames", type=int, default=50)
    p.add_argument("--idle-keyframe", default="stand")
    p.add_argument(
        "--overwrite", action="store_true",
        help="Replace an existing output folder's npz + library.json. A published "
        "release is never edited: bump its _vNNN instead.",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.align_to is not None and args.phase != "xcorr":
        print("--align-to needs --phase xcorr", file=sys.stderr)
        return 1
    try:
        rel = gl.read_release(args.library, args.glob)
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    if not rel.nodes and rel.idle is None:
        print(f"no nodes in {args.library}", file=sys.stderr)
        return 1
    out_dir = args.out_dir or (constants.REPO_ROOT / "outputs" / "mjlab" / args.model / rel.name)
    old = sorted(out_dir.glob("*.npz")) + sorted(out_dir.glob(gl.MANIFEST)) if out_dir.exists() else []
    if old and not args.overwrite:
        print(
            f"{out_dir} already holds {len(old)} files. A published release is never "
            "edited: publish the next _vNNN instead, or pass --overwrite while iterating.",
            file=sys.stderr,
        )
        return 1

    kind = "release" if rel.manifest is not None else "legacy folder"
    print(f"{kind} {rel.name}: {len(rel.nodes)} nodes" + (" + idle.npz" if rel.idle else ""))
    try:
        conv = gl.convert_release(
            rel, args.model, args.fps, phase=args.phase, align_to=args.align_to,
            idle_frames=args.idle_frames,
            idle_keyframe=args.idle_keyframe if args.idle else None,
            velocity=args.velocity, idle_node=args.idle_node,
        )
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2
    for w in conv.warnings:
        print(f"  warning: {w}")
    if args.idle and rel.idle is not None:
        print("  note: --idle ignored, the folder has its own idle.npz")

    print(
        f"{'node':40s} {'N':>3s} {'T[s]':>5s} {'kappa':>6s} {'ph0%':>5s}"
        + (f" {'resid%':>6s}" if args.phase == "xcorr" else "")
        + "  seam/interior-max: jpos jvel root linv angv"
    )
    bad = []
    for r in conv.rows:
        if any(v > gl.SEAM_TOL for v in r["seam"].values()):
            bad.append(r["name"])
        print(
            f"{r['name']:40s} {r['n_frames']:3d} {r['period_s']:5.3f} {r['kappa']:6.4f} "
            f"{100 * r['phase_shift']:5.1f}"
            + (f" {100 * r['xcorr_residual']:6.2f}" if args.phase == "xcorr" else "")
            + "  " + " ".join(f"{v:4.2f}" for v in r["seam"].values())
        )
    if bad:
        print(f"SEAM CHECK FAILED (> {gl.SEAM_TOL}x interior): {bad}", file=sys.stderr)
        return 2

    out_dir.mkdir(parents=True, exist_ok=True)
    for f in old:
        f.unlink()
    for c in conv.clips:
        gl.write_clip(c, out_dir, args.fps)
    if conv.idle is not None:
        gl.write_clip(conv.idle, out_dir, args.fps)
        print(f"idle from {conv.idle_info['from']}: " + ", ".join(
            f"{k} {v:.4g}" if isinstance(v, float) else f"{k} {v}"
            for k, v in conv.idle_info.items() if k != "from"
        ))
    block = {
        "model": args.model,
        "fps": args.fps,
        "phase": args.phase,
        "velocity": args.velocity,
        "touchdown_tol_m": gl.TOUCHDOWN_TOL if args.phase == "touchdown" else None,
        "nodes": {
            f"{r['name']}.npz": {
                "n_frames": r["n_frames"],
                "period_s": r["period_s"],
                "kappa": r["kappa"],
                "phase_shift": r["phase_shift"],
                **({"lifts_m": r["lifts_m"]} if r["lifts_m"] else {}),
            }
            for r in conv.rows
        },
        "idle": conv.idle_info or None,
    }
    gl.write_manifest(out_dir, rel, block)
    n = len(conv.clips) + int(conv.idle is not None)
    print(f"wrote {n} clips + {gl.MANIFEST} to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
