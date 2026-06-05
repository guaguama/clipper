"""Quaternion and interpolation helpers (numpy, MuJoCo wxyz convention).

Shared by ``edits.pad_standing`` (slerp / yaw extraction) and the
``unitree_rl_mjlab`` writer (slerp resampling + SO3 angular-velocity derivative).
All functions operate on float64 arrays with the quaternion in the last axis as
``(w, x, y, z)`` and vectorize over any leading frame/body axes.

The quaternion math mirrors mjlab's ``utils/lab_api/math.py`` (``axis_angle_from_quat``,
``quat_mul``, ``quat_conjugate``, ``quat_slerp``) so the produced motion NPZ matches
what upstream ``csv_to_npz.py`` would emit.
"""

from __future__ import annotations

import numpy as np


def quat_conj(q: np.ndarray) -> np.ndarray:
    """Conjugate of a wxyz quaternion: (w, -x, -y, -z)."""
    q = np.asarray(q, dtype=np.float64)
    out = q.copy()
    out[..., 1:] *= -1.0
    return out


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product of two wxyz quaternion arrays (broadcasting)."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    aw, ax, ay, az = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bw, bx, by, bz = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return np.stack(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        axis=-1,
    )


def quat_normalize(q: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Return unit-norm quaternions; zero-norm inputs map to identity."""
    q = np.asarray(q, dtype=np.float64)
    norm = np.linalg.norm(q, axis=-1, keepdims=True)
    out = np.where(norm > eps, q / np.where(norm > eps, norm, 1.0), 0.0)
    # zero-norm -> identity
    bad = (norm <= eps)[..., 0]
    if np.any(bad):
        out[bad] = np.array([1.0, 0.0, 0.0, 0.0])
    return out


def axis_angle_from_quat(quat: np.ndarray, eps: float = 1.0e-6) -> np.ndarray:
    """Convert wxyz quaternions to axis-angle (rotation vector); shape (..., 3).

    Mirrors mjlab ``axis_angle_from_quat``: sign-folds so ``w >= 0`` (shortest arc)
    and uses a Taylor expansion of ``sin(theta/2)/theta`` near zero angle.
    """
    quat = np.asarray(quat, dtype=np.float64)
    quat = quat * (1.0 - 2.0 * (quat[..., 0:1] < 0.0))  # fold sign so w >= 0
    mag = np.linalg.norm(quat[..., 1:], axis=-1)
    half_angle = np.arctan2(mag, quat[..., 0])
    angle = 2.0 * half_angle
    sin_half_over_angle = np.where(
        np.abs(angle) > eps,
        np.sin(half_angle) / np.where(angle != 0.0, angle, 1.0),
        0.5 - angle * angle / 48.0,
    )
    return quat[..., 1:4] / sin_half_over_angle[..., None]


def lerp(a: np.ndarray, b: np.ndarray, t: np.ndarray | float) -> np.ndarray:
    """Linear interpolation a*(1-t) + b*t."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return a * (1.0 - t) + b * t


def slerp(q0: np.ndarray, q1: np.ndarray, t: np.ndarray | float) -> np.ndarray:
    """Spherical linear interpolation between wxyz quaternions (vectorized).

    Handles the antipodal sign flip (shortest path) and falls back to a
    normalized lerp when the quaternions are nearly parallel. ``t`` broadcasts
    over the leading axis.
    """
    q0 = np.asarray(q0, dtype=np.float64)
    q1 = np.asarray(q1, dtype=np.float64).copy()
    t = np.asarray(t, dtype=np.float64)

    dot = np.sum(q0 * q1, axis=-1)
    # shortest path: flip q1 where the dot product is negative
    flip = dot < 0.0
    q1[flip] *= -1.0
    dot = np.abs(dot)

    t_b = t[..., None] if t.ndim == q0.ndim - 1 else t
    near = dot > (1.0 - 1e-7)

    # general slerp
    dot_c = np.clip(dot, -1.0, 1.0)
    theta = np.arccos(dot_c)
    sin_theta = np.sin(theta)
    sin_theta_safe = np.where(sin_theta > 1e-12, sin_theta, 1.0)
    w0 = np.sin((1.0 - t) * theta) / sin_theta_safe
    w1 = np.sin(t * theta) / sin_theta_safe
    slerped = w0[..., None] * q0 + w1[..., None] * q1

    # near-parallel fallback: normalized lerp
    lerped = quat_normalize(lerp(q0, q1, t_b))

    out = np.where(near[..., None], lerped, slerped)
    return quat_normalize(out)


def yaw_only_quat(q: np.ndarray) -> np.ndarray:
    """Return the upright (roll=pitch=0) wxyz quaternion with `q`'s yaw only."""
    q = np.asarray(q, dtype=np.float64)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    half = 0.5 * yaw
    zeros = np.zeros_like(yaw)
    return np.stack([np.cos(half), zeros, zeros, np.sin(half)], axis=-1)


def frame_blend(
    n_in: int, in_fps: float, out_fps: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Resampling index map, reproducing upstream csv_to_npz interpolation.

    Returns ``(index_0, index_1, blend, n_out)`` such that output frame k is
    ``lerp(x[index_0[k]], x[index_1[k]], blend[k])`` (slerp for quaternions),
    with output times uniformly spaced on ``[0, duration]`` at ``1/out_fps``.

    Args:
        n_in: Number of input frames.
        in_fps: Input frame rate (Hz).
        out_fps: Target output frame rate (Hz).
    """
    if n_in < 2:
        idx = np.zeros(max(n_in, 1), dtype=np.intp)
        return idx, idx.copy(), np.zeros(idx.shape[0]), idx.shape[0]
    in_dt = 1.0 / in_fps
    out_dt = 1.0 / out_fps
    duration = (n_in - 1) * in_dt
    times = np.arange(0.0, duration, out_dt)
    phase = times / duration
    scaled = phase * (n_in - 1)
    index_0 = np.floor(scaled).astype(np.intp)
    index_1 = np.minimum(index_0 + 1, n_in - 1)
    blend = scaled - index_0
    return index_0, index_1, blend, times.shape[0]
