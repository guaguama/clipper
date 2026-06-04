"""Interactive scrubbing of a clipper `Trajectory` (pure MuJoCo + matplotlib).

A small **matplotlib slider window** owns all control; the MuJoCo passive viewer
beside it is display + mouse-camera only. Focus the slider window and use:

* the **Frame** slider (readout ``frame k / N  t=..s``) or ``a``/``d`` (∓1 frame),
  ``w``/``s`` (±``coarse`` frames) to move through the trajectory;
* the **Z-height** slider (readout signed meters) or ``q``/``e`` to raise/lower a
  global z-offset (ground-clipping fix), ``r`` to reset it to 0;
* ``f`` prints a marker line (frame + time + z) to the console — press it at a
  clip's start and end to note crop bounds.

Everything runs on matplotlib's main thread, so the keyboard and the sliders share
one code path (a key just calls ``slider.set_val`` and the slider's ``on_changed``
re-poses). matplotlib's own default shortcuts (``q``=quit, ``s``=save, ``r``=home,
``f``=fullscreen, ...) are disabled *for this figure only* so our keys win without
touching global config or other windows. The passive viewer renders in its own
thread; we only ever call ``viewer.sync()`` from here.
"""

from __future__ import annotations

import mujoco
import mujoco.viewer

from ..trajectory import Trajectory
from .replay import set_pose


def _clamp_frame(f: float, n: int) -> int:
    if n <= 1:
        return 0
    return max(0, min(int(round(f)), n - 1))


def _clamp_z(z: float, r: float) -> float:
    return max(-r, min(float(z), r))


def scrub(
    model: mujoco.MjModel,
    traj: Trajectory,
    z_range: float = 0.2,
    z_fine_step: float = 0.005,
    coarse: int = 10,
    poll_ms: int = 30,
) -> None:
    """Interactively scrub `traj` with a frame slider, z-height slider, and keys.

    Args:
        model: Compiled model whose qpos layout matches `traj.qpos`.
        traj: Trajectory to scrub.
        z_range: Symmetric bound (meters) for the global z-height offset slider.
        z_fine_step: Z-offset change per `q`/`e` keypress (meters).
        coarse: Frame jump per `w`/`s` keypress.
        poll_ms: Interval (ms) of the timer that closes the window when the
            MuJoCo viewer is closed.
    """
    n = traj.num_frames
    fps = float(traj.fps)
    if n == 0:
        raise SystemExit(f"Cannot scrub empty trajectory '{traj.name}'.")

    # matplotlib drives the controls; bail clearly on a non-interactive backend.
    import matplotlib

    backend = matplotlib.get_backend()
    if backend.lower() in ("agg", "template"):
        raise SystemExit(
            f"scrub mode needs an interactive matplotlib backend (got {backend!r}). "
            "Run on a desktop session, or use --mode replay."
        )
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Slider

    data = mujoco.MjData(model)

    # Assigned once the viewer / figure exist; the closures below read them lazily.
    # The two sliders are the single source of truth (everything is main-thread).
    viewer = None
    fig = None
    frame_slider = None
    z_slider = None
    timer = None

    def _fmt_frame(f: float) -> str:
        f = int(f)
        return f"frame {f} / {n}   t={f / fps:.2f}s"

    def _fmt_z(z: float) -> str:
        return f"{z:+.3f} m"

    def _repose() -> None:
        """The sole pose-writer; reads the current slider values."""
        f = _clamp_frame(frame_slider.val, n)
        z = _clamp_z(z_slider.val, z_range)
        set_pose(model, data, traj, f, z_offset=z)
        if viewer is not None:
            try:
                viewer.sync()
            except Exception:
                pass  # viewer may have closed mid-update; timer will tear down

    def _on_change(_val=None) -> None:
        frame_slider.valtext.set_text(_fmt_frame(frame_slider.val))
        z_slider.valtext.set_text(_fmt_z(z_slider.val))
        _repose()

    def _on_key(event) -> None:
        key = (event.key or "").lower()
        if key == "f":  # marker: read-only, no re-pose
            f = _clamp_frame(frame_slider.val, n)
            z = _clamp_z(z_slider.val, z_range)
            print(f"  MARK  frame {f}/{n}  t={f / fps:.2f}s  z={z:+.3f}m")
            return
        cur = _clamp_frame(frame_slider.val, n)
        if key == "a":
            frame_slider.set_val(_clamp_frame(cur - 1, n))
        elif key == "d":
            frame_slider.set_val(_clamp_frame(cur + 1, n))
        elif key == "w":
            frame_slider.set_val(_clamp_frame(cur - coarse, n))
        elif key == "s":
            frame_slider.set_val(_clamp_frame(cur + coarse, n))
        elif key == "q":
            z_slider.set_val(_clamp_z(z_slider.val + z_fine_step, z_range))
        elif key == "e":
            z_slider.set_val(_clamp_z(z_slider.val - z_fine_step, z_range))
        elif key == "r":
            z_slider.set_val(0.0)
        # set_val triggers on_changed -> _on_change -> readout update + re-pose.

    def _tick() -> None:
        if viewer is None or not viewer.is_running():
            timer.stop()
            plt.close(fig)

    print(
        f"Scrubbing '{traj.name}' [{traj.source} -> {traj.model}]: "
        f"{n} frames @ {fps:g} Hz ({traj.duration:.2f} s)."
    )
    print(
        f"  Keys (focus the slider window): a/d +-1 frame  w/s +-{coarse} frames  "
        "q/e z up/down  r reset-z  f mark."
    )
    print("  Or drag the Frame / Z-height sliders. Close either window to quit.")

    with mujoco.viewer.launch_passive(
        model, data, show_left_ui=False, show_right_ui=False
    ) as v:
        viewer = v

        fig = plt.figure(figsize=(6.8, 1.9))
        try:
            fig.canvas.manager.set_window_title(f"clipper scrub - {traj.name}")
        except Exception:
            pass
        # Shifted left with a wide right gutter so the readout never clips.
        ax_frame = fig.add_axes([0.16, 0.56, 0.48, 0.22])
        ax_z = fig.add_axes([0.16, 0.18, 0.48, 0.22])
        frame_slider = Slider(ax_frame, "Frame", 0, max(n - 1, 1), valinit=0, valstep=1)
        z_slider = Slider(ax_z, "Z-height", -z_range, z_range, valinit=0.0)
        frame_slider.valtext.set_text(_fmt_frame(0))
        z_slider.valtext.set_text(_fmt_z(0.0))
        frame_slider.on_changed(_on_change)
        z_slider.on_changed(_on_change)

        # Disable matplotlib's default key shortcuts for THIS figure only (so our
        # q/s/r/f/etc. don't quit/save/home/fullscreen). Scoped to this canvas —
        # global rcParams and other windows are untouched.
        cid = getattr(fig.canvas.manager, "key_press_handler_id", None)
        if cid is not None:
            fig.canvas.mpl_disconnect(cid)
        fig.canvas.mpl_connect("key_press_event", _on_key)

        _repose()  # show frame 0

        timer = fig.canvas.new_timer(interval=poll_ms)
        timer.add_callback(_tick)
        timer.start()
        plt.show(block=True)
    print()
