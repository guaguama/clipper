"""Minimal Linux joystick reader (standard library only, ``/dev/input/js*``).

Reads the kernel joystick API's 8-byte ``js_event`` records (``u32 time, s16 value,
u8 type, u8 number``) on a daemon thread and keeps the latest axis values plus a queue
of button presses. No pygame/evdev dependency.

Xbox controllers on the ``xpad`` driver (checked on a Series S|X pad, 8 axes, 11
buttons): axes 0 = left X, 1 = left Y (up is negative), 2 = LT, 3 = right X,
4 = right Y, 5 = RT, 6/7 = d-pad; buttons 0 = A, 1 = B, 2 = X, 3 = Y.

Open the device with ``os.open``/``os.read``. A plain buffered ``open(..., "rb")``
read hung without delivering even the init events on this machine.
"""

from __future__ import annotations

import array
import fcntl
import os
import select
import struct
import threading

_EVENT = struct.Struct("IhBB")
_JS_BUTTON, _JS_AXIS, _JS_INIT = 0x01, 0x02, 0x80
_JSIOCGAXES, _JSIOCGBUTTONS = 0x80016A11, 0x80016A12
_JSIOCGNAME_64 = 0x80006A13 + (64 << 16)

AXIS_LX, AXIS_LY, AXIS_RX = 0, 1, 3
BUTTON_A, BUTTON_B = 0, 1


class Joystick:
    """Background reader for one ``/dev/input/js*`` device."""

    def __init__(self, path: str = "/dev/input/js0"):
        self.path = path
        self._fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK)
        buf = array.array("B", [0])
        fcntl.ioctl(self._fd, _JSIOCGAXES, buf)
        n_axes = buf[0]
        fcntl.ioctl(self._fd, _JSIOCGBUTTONS, buf)
        n_buttons = buf[0]
        name = array.array("B", [0] * 64)
        fcntl.ioctl(self._fd, _JSIOCGNAME_64, name)
        self.name = name.tobytes().split(b"\0")[0].decode(errors="replace")
        self._axes = [0.0] * max(n_axes, 6)
        self._buttons = [False] * max(n_buttons, 2)
        self._pressed: list[int] = []
        self._lock = threading.Lock()
        self._stop = False
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    @classmethod
    def open(cls, path: str) -> "Joystick | None":
        """Open `path`, or print why not and return None (the caller runs without it)."""
        try:
            js = cls(path)
        except OSError as e:
            print(f"joystick: cannot open {path} ({e}); sliders only.")
            return None
        print(f"joystick: {js.name} on {path}")
        return js

    def _run(self) -> None:
        while not self._stop:
            r, _, _ = select.select([self._fd], [], [], 0.1)
            if not r:
                continue
            try:
                data = os.read(self._fd, _EVENT.size * 64)
            except BlockingIOError:
                continue
            except OSError:
                print(f"joystick: {self.path} disconnected.")
                return
            with self._lock:
                for off in range(0, len(data) - _EVENT.size + 1, _EVENT.size):
                    _, value, typ, num = _EVENT.unpack_from(data, off)
                    if typ & _JS_AXIS and num < len(self._axes):
                        self._axes[num] = value / 32767.0
                    elif typ & _JS_BUTTON and num < len(self._buttons):
                        down = bool(value)
                        if down and not (typ & _JS_INIT) and not self._buttons[num]:
                            self._pressed.append(num)
                        self._buttons[num] = down

    def poll(self) -> tuple[list[float], list[int]]:
        """(latest axes in [-1, 1], buttons pressed since the last poll)."""
        with self._lock:
            pressed, self._pressed = self._pressed, []
            return list(self._axes), pressed

    def close(self) -> None:
        self._stop = True
        self._thread.join(timeout=0.5)
        try:
            os.close(self._fd)
        except OSError:
            pass


def deadzone(v: float, dz: float) -> float:
    """Zero inside |v| < dz, rescaled to reach ±1 at full deflection outside it."""
    if abs(v) < dz:
        return 0.0
    return (abs(v) - dz) / (1.0 - dz) * (1.0 if v > 0 else -1.0)
