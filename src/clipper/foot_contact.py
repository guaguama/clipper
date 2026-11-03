"""Estimate ankle + height offsets that put the feet flat on the ground in stance.

Backs crop's ``--auto-foot-contact``. Retargeted walking clips played on
``osl_fullbody`` tend to stand toe-up: only the heel touches in stance. This module
finds the same three global corrections a user dials in by hand with the scrub
viewer (``--ankle-offset``, ``--bio-ankle-offset``, ``--height-offset``), per clip:

1. **Contact surface.** The non-mesh collision geoms on each foot's bodies
   (``constants.FOOT_CONTACT_BODIES``), sampled as surface points. Meshes are skipped,
   so the OSL foot is its 3 capsules, not its foot mesh.
2. **Heel / forefoot.** In the model's ``stand`` keyframe each foot's points are split
   along the foot's horizontal long axis: the rear 30 % (the side the ankle joint sits
   over) is the heel, the front 30 % the forefoot. Points ride rigidly on their geoms,
   so the labels hold in every frame.
3. **Planted frames**, detected on the input clip: the foot's lowest point within
   20 mm of its own 5th-percentile height, and its horizontal speed under 0.15 m/s
   (standing and mid-stance; not swing, mostly not heel strike / toe off).
4. **Ankle offset**, per foot: the constant joint delta that makes the median over
   planted frames of (lowest forefoot z - lowest heel z) zero — a flat foot. Root-found
   by bisection, rounded to 0.1 deg.
5. **Height offset**, after the ankle offsets: minus the median lowest-point height
   over every planted frame of both feet (median contact), rounded to 0.1 mm.

Only kinematics is used (``mj_kinematics``); no contact forces. Validated against the
hand-tuned KIT/7 ``WalkingStraightForwards03`` clip (OSL -7 deg, bio -5 deg): this
estimates -6.7 / -4.7.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import mujoco
import numpy as np

from . import constants
from .qpos import joint_qpos_address
from .trajectory import Trajectory

# Which ankle table drives which foot (tags match `edits.apply_ankle_offset`).
_ANKLE_TABLES: dict[str, dict[str, str]] = {
    "ankle": constants.RIGHT_ANKLE_JOINT,
    "bioankle": constants.BIO_ANKLE_JOINT,
}

HEEL_TOE_FRACTION = 0.30  # rear / front share of the foot length labelled heel / forefoot
PLANTED_HEIGHT = 0.020  # m above the foot's 5th-percentile lowest point
PLANTED_SPEED = 0.15  # m/s horizontal speed of the foot's contact points
SEARCH_DEG = 20.0  # ankle offset bracket, +/- degrees
SWING_WARN = 0.005  # m: warn when a non-planted frame clips deeper than this


@dataclass
class FootFit:
    """Result for one foot."""

    tag: str  # "ankle" (OSL) / "bioankle" — the clip suffix tag
    joint: str
    qpos_adr: int
    offset_deg: float
    n_planted: int
    flat_before: float  # m, median(forefoot - heel) over planted frames, input clip
    flat_after: float  # m, same with the (rounded) offset applied
    clamped: bool = False  # no sign change within +/- SEARCH_DEG
    stance_gap: float = 0.0  # m, median lowest point over planted frames after dz


@dataclass
class FootContactFit:
    """All offsets plus the post-correction contact diagnostics (meters)."""

    feet: list[FootFit]
    dz: float
    n_frames: int
    stance_gap_median: float  # lowest planted point after dz (~0 by construction)
    stance_pen_p95: float  # 95th percentile stance penetration depth (>= 0)
    stance_pen_max: float
    stance_float_p95: float  # 95th percentile stance float height (>= 0)
    swing_pen_max: float  # deepest penetration on a non-planted frame
    swing_pen_frames: int  # non-planted (foot, frame) pairs below the ground
    warnings: list[str] = field(default_factory=list)


def _sphere_dirs(n: int) -> np.ndarray:
    """`n` roughly uniform unit vectors (Fibonacci sphere), deterministic."""
    i = np.arange(n) + 0.5
    phi = np.arccos(1.0 - 2.0 * i / n)
    theta = np.pi * (1.0 + 5.0**0.5) * i
    return np.stack(
        [np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)], axis=1
    )


def _geom_points(model: mujoco.MjModel, g: int) -> np.ndarray:
    """Surface sample points of a primitive collision geom, in its local frame."""
    size = model.geom_size[g]
    typ = model.geom_type[g]
    if typ == mujoco.mjtGeom.mjGEOM_SPHERE:
        return _sphere_dirs(200) * size[0]
    if typ == mujoco.mjtGeom.mjGEOM_CAPSULE:
        along = np.linspace(-size[1], size[1], 25)
        shell = _sphere_dirs(80) * size[0]
        pts = shell[None, :, :] + np.stack([0 * along, 0 * along, along], 1)[:, None, :]
        return pts.reshape(-1, 3)
    if typ == mujoco.mjtGeom.mjGEOM_ELLIPSOID:
        return _sphere_dirs(400) * size[:3]
    if typ == mujoco.mjtGeom.mjGEOM_BOX:
        corners = np.array(np.meshgrid([-1, 1], [-1, 1], [-1, 1])).reshape(3, -1).T
        return corners * size[:3]
    raise ValueError(f"unsupported foot contact geom type {mujoco.mjtGeom(typ).name}")


class _Foot:
    """Sampled contact surface of one foot plus its heel / forefoot labels."""

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, bodies, joint: str):
        body_ids = {model.body(b).id for b in bodies}
        self.geoms = [
            g
            for g in range(model.ngeom)
            if model.geom_bodyid[g] in body_ids
            and model.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH
            and (model.geom_contype[g] or model.geom_conaffinity[g])
        ]
        if not self.geoms:
            raise ValueError(f"no non-mesh collision geoms on foot bodies {bodies}")
        self.local = [_geom_points(model, g) for g in self.geoms]

        # Labels from the stand pose (data must already hold it, kinematics done).
        w = self.world(data)
        xy = w[:, :2] - w[:, :2].mean(axis=0)
        axis = np.linalg.svd(xy, full_matrices=False)[2][0]  # horizontal long axis
        anchor = data.xanchor[model.joint(joint).id][:2] - w[:, :2].mean(axis=0)
        if anchor @ axis > 0:  # point the axis heel -> toe (ankle sits over the heel)
            axis = -axis
        s = xy @ axis
        lo, hi = s.min(), s.max()
        cut = HEEL_TOE_FRACTION * (hi - lo)
        self.heel = s < lo + cut
        self.toe = s > hi - cut

    def world(self, data: mujoco.MjData) -> np.ndarray:
        return np.concatenate(
            [
                data.geom_xpos[g] + p @ data.geom_xmat[g].reshape(3, 3).T
                for g, p in zip(self.geoms, self.local)
            ]
        )


def _features(model, data, qpos: np.ndarray, foot: _Foot) -> np.ndarray:
    """Per frame: [lowest z, lowest heel z, lowest forefoot z, centroid x, centroid y]."""
    out = np.empty((len(qpos), 5))
    for k, q in enumerate(qpos):
        data.qpos[:] = q
        mujoco.mj_kinematics(model, data)
        w = foot.world(data)
        out[k] = (w[:, 2].min(), w[foot.heel, 2].min(), w[foot.toe, 2].min(),
                  *w[:, :2].mean(axis=0))
    return out


def _planted(feat: np.ndarray, fps: float) -> np.ndarray:
    ground = np.percentile(feat[:, 0], 5)
    if len(feat) > 1:
        speed = np.linalg.norm(np.gradient(feat[:, 3:5], 1.0 / fps, axis=0), axis=1)
    else:
        speed = np.zeros(len(feat))
    return (feat[:, 0] < ground + PLANTED_HEIGHT) & (speed < PLANTED_SPEED)


def fit_foot_contact(model: mujoco.MjModel, traj: Trajectory) -> FootContactFit:
    """Estimate the ankle offsets (deg) and height offset (m) for `traj`.

    Raises:
        ValueError: if `traj.model` has no foot / ankle table entries, or a foot is
            never planted.
    """
    model_id = traj.model
    table = constants.FOOT_CONTACT_BODIES.get(model_id)
    if table is None:
        raise ValueError(f"no FOOT_CONTACT_BODIES entry for model {model_id!r}")

    data = mujoco.MjData(model)
    stand = next((k for k in range(model.nkey) if model.key(k).name == "stand"), 0)
    data.qpos[:] = model.key_qpos[stand] if model.nkey else model.qpos0
    mujoco.mj_kinematics(model, data)

    q = traj.qpos
    feet, fits, warnings = [], [], []
    for tag, bodies in table.items():
        joint = _ANKLE_TABLES[tag].get(model_id)
        adr = joint_qpos_address(model, joint) if joint else None
        if adr is None:
            raise ValueError(f"model {model_id!r} has no {tag} joint for its foot {bodies}")
        foot = _Foot(model, data, bodies, joint)
        feat = _features(model, data, q, foot)
        planted = _planted(feat, traj.fps)
        if not planted.any():
            raise ValueError(f"{tag} foot ({joint}) is never planted in {traj.name}")
        qp = q[planted]

        def flat(deg: float) -> float:
            qq = qp.copy()
            qq[:, adr] += np.deg2rad(deg)
            f = _features(model, data, qq, foot)
            return float(np.median(f[:, 2] - f[:, 1]))

        lo, hi = -SEARCH_DEG, SEARCH_DEG
        f_lo, f_hi = flat(lo), flat(hi)
        clamped = np.sign(f_lo) == np.sign(f_hi)
        if clamped:
            deg = lo if abs(f_lo) < abs(f_hi) else hi
            warnings.append(
                f"{tag}: no flat-foot offset within +/-{SEARCH_DEG:g} deg; clamped to {deg:+g}"
            )
        else:
            while hi - lo > 0.05:
                mid = 0.5 * (lo + hi)
                f_mid = flat(mid)
                if np.sign(f_mid) == np.sign(f_lo):
                    lo, f_lo = mid, f_mid
                else:
                    hi = mid
            deg = 0.5 * (lo + hi)
        deg = round(deg, 1) + 0.0  # + 0.0 turns -0.0 into 0.0
        fits.append(
            FootFit(
                tag=tag, joint=joint, qpos_adr=adr, offset_deg=deg,
                n_planted=int(planted.sum()), flat_before=float(np.median(feat[planted, 2] - feat[planted, 1])),
                flat_after=flat(deg), clamped=bool(clamped),
            )
        )
        feet.append((foot, planted))

    # Height: median contact over every planted frame of both feet, ankles applied.
    q_fixed = q.copy()
    for fit in fits:
        q_fixed[:, fit.qpos_adr] += np.deg2rad(fit.offset_deg)
    low_stance, low_swing = [], []
    for fit, (foot, planted) in zip(fits, feet):
        low = _features(model, data, q_fixed, foot)[:, 0]
        fit.stance_gap = float(np.median(low[planted]))  # dz added below
        low_stance.append(low[planted])
        low_swing.append(low[~planted])
    stance = np.concatenate(low_stance)
    swing = np.concatenate(low_swing)
    dz = round(-float(np.median(stance)), 4) + 0.0
    stance = stance + dz
    swing = swing + dz
    for fit in fits:
        fit.stance_gap += dz

    swing_pen = np.clip(-swing, 0.0, None)
    swing_pen_max = float(swing_pen.max()) if swing.size else 0.0
    if swing_pen_max > SWING_WARN:
        warnings.append(
            f"non-planted frames clip the ground by up to {swing_pen_max * 1e3:.1f} mm "
            f"({int((swing_pen > 0).sum())} foot-frames); z not adjusted for it"
        )
    pen = np.clip(-stance, 0.0, None)
    flo = np.clip(stance, 0.0, None)
    return FootContactFit(
        feet=fits,
        dz=dz,
        n_frames=len(q),
        stance_gap_median=float(np.median(stance)),
        stance_pen_p95=float(np.percentile(pen, 95)),
        stance_pen_max=float(pen.max()),
        stance_float_p95=float(np.percentile(flo, 95)),
        swing_pen_max=swing_pen_max,
        swing_pen_frames=int((swing_pen > 0).sum()),
        warnings=warnings,
    )


def format_report(fit: FootContactFit) -> str:
    """Human-readable summary of a fit (mm / deg)."""
    lines = ["Auto foot contact:"]
    for f in fit.feet:
        lines.append(
            f"  {f.tag:<9} {f.joint:<30} {f.offset_deg:+5.1f} deg   planted "
            f"{f.n_planted}/{fit.n_frames}   forefoot-heel {f.flat_before * 1e3:+6.1f} "
            f"-> {f.flat_after * 1e3:+5.1f} mm   stance gap {f.stance_gap * 1e3:+.1f} mm"
            + ("   (CLAMPED)" if f.clamped else "")
        )
    lines.append(
        f"  height    {fit.dz:+.4f} m   stance: median gap {fit.stance_gap_median * 1e3:+.1f} mm, "
        f"penetration p95 {fit.stance_pen_p95 * 1e3:.1f} / max {fit.stance_pen_max * 1e3:.1f} mm, "
        f"float p95 {fit.stance_float_p95 * 1e3:.1f} mm"
    )
    lines.append(
        f"            swing: max penetration {fit.swing_pen_max * 1e3:.1f} mm "
        f"({fit.swing_pen_frames} foot-frames)"
    )
    lines += [f"  WARNING: {w}" for w in fit.warnings]
    return "\n".join(lines)
