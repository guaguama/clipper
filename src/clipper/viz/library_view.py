"""Interactive twist-command viewer for a gait library (MuJoCo + matplotlib).

Same split as ``scrub.py``: a small **matplotlib window** owns all control on the main
thread, and the MuJoCo passive viewer only displays (mouse camera). A matplotlib timer
is the playback clock. Each tick it:

1. measures wall time;
2. reads the command from the sliders, or from the joystick while a stick is out of
   its deadzone;
3. calls ``LibraryPlayer.command`` and ``LibraryPlayer.step``;
4. writes qpos, runs ``mj_forward`` and ``viewer.sync()``.

Controls (focus the slider window for keys):

* sliders ``vx`` / ``vy`` / ``wz``, one per axis the library actually varies, spanning
  ``[min(lib, 0), max(lib, 0)]``. The command snaps to the nearest node; the status
  line shows which one.
* **Stop** button or ``space``: command 0 (the idle clip if the library has one).
* ``up``/``down``: vx ±0.05. ``left``/``right``: wz ±0.1 (left = +wz = CCW turn).
* joystick (``--joystick``), twin-stick:
  - left stick Y = vx, right stick X = wz, left stick X = vy (when it varies);
  - stick ends map to the library extremes, centre to 0;
  - A = stop, B = hand control back to the sliders;
  - while a stick drives, the sliders follow it.
"""

from __future__ import annotations

import time

import mujoco
import mujoco.viewer
import numpy as np

from .camera import center_on_pelvis
from .joystick import AXIS_LX, AXIS_LY, AXIS_RX, BUTTON_A, BUTTON_B, Joystick, deadzone
from .library_player import AXES, LibraryPlayer
from .replay import _default_render_flags_off

TICK_MS = 16
STATUS_PERIOD = 0.2  # s between status-text redraws
KEY_STEP = {"vx": 0.05, "vy": 0.05, "wz": 0.1}


def view_library(
    model: mujoco.MjModel,
    player: LibraryPlayer,
    model_id: str,
    speed: float = 1.0,
    treadmill: bool = False,
    joystick: Joystick | None = None,
    dz: float = 0.12,
    title: str = "gait library",
) -> None:
    import matplotlib

    if matplotlib.get_backend().lower() in ("agg", "template"):
        raise SystemExit(
            f"the gait-library viewer needs an interactive matplotlib backend "
            f"(got {matplotlib.get_backend()!r}). Run on a desktop session."
        )
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Button, Slider

    sel = player.selector
    axes_on = [i for i in range(3) if sel.varies[i]]
    data = mujoco.MjData(model)
    state = {"joy": False, "from_joy": False, "t": None, "status_t": 0.0}
    sliders: dict[int, Slider] = {}
    viewer = fig = timer = status = None

    def _slider_cmd() -> np.ndarray:
        cmd = np.zeros(3)
        for i, s in sliders.items():
            cmd[i] = s.val
        return cmd

    def _set_sliders(cmd) -> None:
        state["from_joy"] = True
        try:
            for i, s in sliders.items():
                if abs(s.val - cmd[i]) > 1e-9:
                    s.set_val(float(cmd[i]))
        finally:
            state["from_joy"] = False

    def _on_slider(_val) -> None:
        if not state["from_joy"]:
            state["joy"] = False  # a hand on the sliders takes control back

    def _stop(_event=None) -> None:
        state["joy"] = False
        _set_sliders(np.zeros(3))

    def _on_key(event) -> None:
        key = event.key or ""
        if key == " ":
            _stop()
            return
        delta = {
            "up": (0, +1), "down": (0, -1), "left": (2, +1), "right": (2, -1),
        }.get(key)
        if delta is None or delta[0] not in sliders:
            return
        i, sign = delta
        state["joy"] = False
        cmd = _slider_cmd()
        cmd[i] += sign * KEY_STEP[AXES[i]]
        _set_sliders(sel.clamp(cmd))

    def _joystick_cmd() -> np.ndarray | None:
        """Command from the sticks while they drive, else None (sliders drive)."""
        if joystick is None:
            return None
        axes, pressed = joystick.poll()
        if BUTTON_A in pressed:
            _stop()
            return None
        if BUTTON_B in pressed:
            state["joy"] = False
            return None
        stick = np.array([
            -deadzone(axes[AXIS_LY], dz),  # up = forward
            -deadzone(axes[AXIS_LX], dz),  # left = +vy
            -deadzone(axes[AXIS_RX], dz),  # left = +wz (CCW)
        ])
        stick[~sel.varies] = 0.0
        if np.any(stick != 0.0):
            state["joy"] = True
        if not state["joy"]:
            return None
        cmd = sel.from_stick(stick)
        _set_sliders(cmd)
        return cmd

    def _status_lines() -> tuple[str, str]:
        c = player.clip
        tw = ", ".join(f"{AXES[i]} {c.twist[i]:+.2f}" for i in range(3))
        cm = ", ".join(f"{AXES[i]} {player.cmd[i]:+.2f}" for i in range(3))
        blend = f"   blend {player.alpha:.0%}" if player.prev is not None else ""
        src = "joystick" if state["joy"] else "sliders"
        left = "command\nclip\nnode twist\nphase"
        right = (
            f"{cm}   [{src}]\n{c.name}\n{tw}   (eff. vx {c.twist_effective[0]:+.3f})\n"
            f"{player.phase:.2f} of {c.n} frames   loop {player.loops}{blend}"
        )
        return left, right

    def _tick() -> None:
        if viewer is None or not viewer.is_running():
            timer.stop()
            plt.close(fig)
            return
        now = time.perf_counter()
        dt = 0.0 if state["t"] is None else min(now - state["t"], 0.1)
        state["t"] = now
        cmd = _joystick_cmd()
        if cmd is None:
            cmd = _slider_cmd()
        player.command(cmd)
        q = player.step(dt * speed)
        if treadmill:
            q = q.copy()
            q[:2] = 0.0
        with viewer.lock():
            data.qpos[:] = q
            mujoco.mj_forward(model, data)
            if not treadmill:
                viewer.cam.lookat[:2] = q[:2]  # follow; angle and zoom stay the user's
        if now - state["status_t"] > STATUS_PERIOD:
            state["status_t"] = now
            left, right = _status_lines()
            viewer.set_texts((None, mujoco.mjtGridPos.mjGRID_TOPLEFT, left, right))
            status.set_text(right)
            fig.canvas.draw_idle()
        viewer.sync()

    lo, hi = sel.lo, sel.hi
    print(f"{len(player.clips)} clips. Command ranges (library extremes, widened to 0):")
    for i in range(3):
        rng = f"[{lo[i]:+.2f}, {hi[i]:+.2f}]" if sel.varies[i] else "fixed 0"
        print(f"  {AXES[i]}: {rng}   weight {sel.weights[i]:.3f}")
    idle = [c.name for c in player.clips if c.is_idle]
    print(f"  idle clip: {idle[0] if idle else 'none (zero command snaps to the nearest node)'}")
    print(
        "  Keys (slider window): up/down vx, left/right wz, space stop."
        + ("  Joystick: LY vx, RX wz, A stop, B sliders." if joystick else "")
        + "  Close either window to quit."
    )

    with mujoco.viewer.launch_passive(
        model, data, show_left_ui=False, show_right_ui=False
    ) as v:
        viewer = v
        _default_render_flags_off(viewer)
        q0 = player.qpos()
        center_on_pelvis(viewer.cam, q0, model_id)

        n_rows = max(len(axes_on), 1)
        fig = plt.figure(figsize=(7.2, 1.0 + 0.55 * n_rows + 1.3))
        try:
            fig.canvas.manager.set_window_title(f"clipper gait library - {title}")
        except Exception:
            pass
        h = fig.get_figheight()
        y = 1.0 - 0.45 / h
        for i in axes_on:
            y -= 0.55 / h
            ax = fig.add_axes([0.12, y, 0.62, 0.32 / h])
            s = Slider(ax, AXES[i], lo[i], hi[i], valinit=0.0)
            s.on_changed(_on_slider)
            sliders[i] = s
        fixed = [AXES[i] for i in range(3) if not sel.varies[i]]
        if fixed:
            fig.text(0.02, 1.0 - 0.3 / h, f"fixed at 0 (library never varies them): {', '.join(fixed)}",
                     fontsize=8, color="0.4")
        ax_stop = fig.add_axes([0.82, y, 0.14, 0.32 / h])
        stop_btn = Button(ax_stop, "Stop")
        stop_btn.on_clicked(_stop)
        status = fig.text(0.02, 0.05, "", fontsize=8, family="monospace", va="bottom")

        cid = getattr(fig.canvas.manager, "key_press_handler_id", None)
        if cid is not None:
            fig.canvas.mpl_disconnect(cid)
        fig.canvas.mpl_connect("key_press_event", _on_key)

        timer = fig.canvas.new_timer(interval=TICK_MS)
        timer.add_callback(_tick)
        timer.start()
        plt.show(block=True)
    if joystick is not None:
        joystick.close()
