#!/usr/bin/env python
"""Crop / height-adjust / pad a reference trajectory and write a target NPZ.

Pipeline: load -> crop -> height-offset -> (optional) standing-pose pad -> write.
Frame indices are 0-based inclusive, matching the scrub viewer's `f` marker.

Examples:
    # Crop Lafan1 frames 100..400 and write an mjlab motion NPZ (50 Hz)
    python scripts/crop_trajectory.py \
        --path ~/.g1mocap/Lafan1/dance1_subject1.npz --source lafan1 --model g1_29dof \
        --start 100 --stop 400 --format mjlab

    # Crop, raise 3 cm, pad standing at both ends, resample to 50 Hz, preview
    python scripts/crop_trajectory.py \
        --path <bones_seed.csv> --source bones_seed --model g1_29dof \
        --start 50 --stop 600 --height-offset 0.03 --pad-standing --output-fps 50 \
        --format mj_nlp --visualize
"""

from __future__ import annotations

import argparse
from pathlib import Path

from clipper import constants, edits
from clipper.model import load_model
from clipper.qpos import joint_qpos_address
from clipper.sources import get_loader
from clipper.writers import WRITERS, get_writer


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Crop, height-adjust, pad, and convert a reference trajectory.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--path", required=True, type=Path, help="Trajectory file.")
    p.add_argument(
        "--source", required=True, choices=constants.SOURCES,
        help="Source format (no autodetection).",
    )
    p.add_argument(
        "--model", required=True, choices=constants.MODELS, help="Target model.",
    )
    p.add_argument("--start", type=int, default=None, help="First frame (0-based, inclusive).")
    p.add_argument("--stop", type=int, default=None, help="Last frame (0-based, inclusive).")
    p.add_argument(
        "--height-offset", type=float, default=0.0,
        help="Global z offset (m) added to every frame's base height.",
    )
    p.add_argument(
        "--ankle-offset", type=float, default=0.0,
        help="Global offset (deg) added to the right-ankle joint of every frame "
        "(flattens the OSL prosthetic foot). Matches scrub's R-ankle slider.",
    )
    p.add_argument(
        "--pad-standing", action="store_true",
        help="Prepend/append a standing pose with a blended transition.",
    )
    p.add_argument("--pre-static", type=float, default=1.0, help="Standing hold (s) before motion.")
    p.add_argument("--pre-blend", type=float, default=0.5, help="Blend (s) standing -> motion.")
    p.add_argument("--post-static", type=float, default=1.0, help="Standing hold (s) after motion.")
    p.add_argument("--post-blend", type=float, default=0.5, help="Blend (s) motion -> standing.")
    p.add_argument(
        "--output-fps", type=float, default=None,
        help="Resample the written NPZ to this fps (lerp/slerp). "
        "Default: keep the source fps.",
    )
    p.add_argument(
        "--format", required=True, choices=tuple(WRITERS),
        help="Output format.",
    )
    p.add_argument("--name", type=str, default=None, help="Output file stem (default derived).")
    p.add_argument("--visualize", action="store_true", help="Replay the final clip first.")
    p.add_argument("--save-video", action="store_true", help="Render the final clip to an mp4.")
    p.add_argument(
        "--skybox", type=Path, default=None,
        help="PNG skybox for --visualize / --save-video only (single images must be "
        "square). Does not affect the written NPZ.",
    )
    p.add_argument(
        "--floor-texture", type=Path, default=None,
        help="Seamless PNG tiled over the ground plane, for --visualize / "
        "--save-video only. Does not affect the written NPZ.",
    )
    p.add_argument(
        "--clones", type=int, default=1,
        help="Model copies on an evenly spaced grid, in sync, for --visualize / "
        "--save-video only. Does not affect the written NPZ.",
    )
    p.add_argument(
        "--clone-spacing", type=float, default=1.5,
        help="Distance (m) between neighbouring clones.",
    )
    return p.parse_args(argv)


def _save_video(
    model_id: str, traj, path: Path, skybox: Path | None = None, clones: int = 1,
    clone_spacing: float = 1.5, floor_texture: Path | None = None,
) -> None:
    """Offscreen-render `traj` to an mp4 beside the NPZ (best effort)."""
    try:
        import imageio.v2 as imageio
        import mujoco

        from clipper.viz.camera import center_on_pelvis
        from clipper.viz.replay import (
            draw_clone_markers, grid_offsets, grid_radius, set_pose,
        )
    except Exception as exc:  # pragma: no cover - optional dep
        raise SystemExit(f"--save-video unavailable: {exc}")

    model = load_model(
        model_id, floor=True, skybox=skybox, clones=clones, floor_texture=floor_texture
    )
    offsets = grid_offsets(clones, clone_spacing) if clones > 1 else None
    data = mujoco.MjData(model)
    try:
        # The 10000-geom default fits one MSK model (~1.4k geoms) but not a grid.
        renderer = mujoco.Renderer(model, height=480, width=640, max_geom=10000 * clones)
    except Exception as exc:
        raise SystemExit(
            f"--save-video: could not create an offscreen GL context ({exc}). "
            "Set MUJOCO_GL=egl or osmesa, or drop --save-video."
        )
    # Match the interactive viewers: no shadows / reflections in the output video.
    renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0
    renderer.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = 0
    # Free camera, re-centered on the pelvis each frame: starts framed like the
    # interactive viewers, then tracks the robot at a fixed world viewing angle.
    cam = mujoco.MjvCamera()
    frames = []
    for k in range(traj.num_frames):
        set_pose(model, data, traj, k, clone_offsets=offsets)
        center_on_pelvis(cam, traj.qpos[k], traj.model, extra_distance=grid_radius(offsets))
        renderer.update_scene(data, cam)
        if traj.markers is not None:
            draw_clone_markers(renderer.scene, traj.markers[k], offsets)
        frames.append(renderer.render())
    renderer.close()
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(path, frames, fps=int(round(traj.fps)))
    print(f"Wrote video {path}")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if not args.path.exists():
        raise SystemExit(f"Trajectory file not found: {args.path}")
    if args.clones < 1:
        raise SystemExit(f"--clones must be >= 1, got {args.clones}.")

    model = load_model(args.model, floor=False)
    loader = get_loader(args.source)
    traj = loader(args.path, model, args.model, source=args.source)

    traj = edits.crop(traj, args.start, args.stop)
    traj = edits.apply_height_offset(traj, args.height_offset)
    if args.ankle_offset != 0.0:
        ankle_name = constants.RIGHT_ANKLE_JOINT.get(args.model)
        ankle_adr = joint_qpos_address(model, ankle_name) if ankle_name else None
        if ankle_adr is None:
            raise SystemExit(
                f"--ankle-offset is not supported for model {args.model!r} "
                "(no mapped right-ankle joint)."
            )
        traj = edits.apply_ankle_offset(traj, ankle_adr, args.ankle_offset)
    if args.pad_standing:
        traj = edits.pad_standing(
            traj, args.pre_static, args.pre_blend, args.post_static, args.post_blend
        )

    print(
        f"Final clip '{traj.name}': {traj.num_frames} frames @ {traj.fps:g} Hz "
        f"({traj.duration:.2f} s)."
    )

    if args.visualize:
        from clipper.viz import replay
        from clipper.viz.replay import grid_offsets

        offsets = (
            grid_offsets(args.clones, args.clone_spacing) if args.clones > 1 else None
        )
        view_model = load_model(
            args.model, skybox=args.skybox, clones=args.clones,
            floor_texture=args.floor_texture,
        )
        replay(view_model, traj, clone_offsets=offsets)

    writer = get_writer(args.format)
    out = writer(traj, output_fps=args.output_fps, name=args.name)
    print(f"Wrote {args.format} -> {out}")

    if args.save_video:
        _save_video(
            args.model, traj, out.with_suffix(".mp4"), skybox=args.skybox,
            clones=args.clones, clone_spacing=args.clone_spacing,
            floor_texture=args.floor_texture,
        )


if __name__ == "__main__":
    main()
