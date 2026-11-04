"""5 universal spatial primitives (per spec section 4.3). Numpy + scipy only.

Conventions: positions are (T, 2) float arrays in pixels, y grows downward.
alpha is meters-per-pixel: float or (T,) array. Missing detections = NaN rows.
"""
from __future__ import annotations

import numpy as np

try:
    from scipy.signal import savgol_filter
except ImportError:  # shortcut: savgol missing, fall back to raw gradient (noisy on real video)
    savgol_filter = None


def _as_alpha(alpha, T: int) -> np.ndarray:
    if np.isscalar(alpha):
        return np.full(T, float(alpha))
    a = np.asarray(alpha, dtype=float)
    if a.shape != (T,):
        raise ValueError(f"alpha shape {a.shape} != ({T},)")
    return a


def _dist_m(p: np.ndarray, q: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    return np.linalg.norm(p - q, axis=1) * alpha


def proximity_timer(
    subj_foot_px: np.ndarray,
    ball_center_px: np.ndarray,
    alpha,
    contact_radius_m: float = 0.45,
    fps: float = 30.0,
    hysteresis: int = 4,
) -> list[dict]:
    """Contiguous possession windows where foot-ball distance <= radius.

    A gap of <= hysteresis non-contact frames is bridged (rapid-touch dropout).
    Returns [{start_frame, end_frame, duration_s}, ...].
    """
    p = np.asarray(subj_foot_px, dtype=float)
    b = np.asarray(ball_center_px, dtype=float)
    T = len(p)
    a = _as_alpha(alpha, T)
    finite = np.isfinite(p).all(axis=1) & np.isfinite(b).all(axis=1)
    contact = np.zeros(T, dtype=bool)
    contact[finite] = _dist_m(p[finite], b[finite], a[finite]) <= contact_radius_m

    # Raw True-runs, then merge gaps <= hysteresis.
    runs, s = [], None
    for t in range(T):
        if contact[t] and s is None:
            s = t
        elif not contact[t] and s is not None:
            runs.append((s, t - 1))
            s = None
    if s is not None:
        runs.append((s, T - 1))
    merged = []
    for s, e in runs:
        if merged and s - merged[-1][1] - 1 <= hysteresis:
            merged[-1][1] = e
        else:
            merged.append([s, e])
    return [
        {"start_frame": int(s), "end_frame": int(e), "duration_s": (e - s + 1) / fps}
        for s, e in merged
    ]


def tether_distance_series(
    foot_a_px: np.ndarray, foot_b_px: np.ndarray, alpha
) -> np.ndarray:
    """Per-frame Euclidean distance in meters between two tracks."""
    a = np.asarray(foot_a_px, dtype=float)
    b = np.asarray(foot_b_px, dtype=float)
    return _dist_m(a, b, _as_alpha(alpha, len(a)))


def nearest_tether_series(
    group_a: dict[int, np.ndarray], group_b: dict[int, np.ndarray], alpha
) -> np.ndarray:
    """Per-frame min distance (m) between any track in A and any in B."""
    ids_a, ids_b = list(group_a), list(group_b)
    T = len(group_a[ids_a[0]])
    best = np.full(T, np.inf)
    for i in ids_a:
        for j in ids_b:
            d = tether_distance_series(group_a[i], group_b[j], alpha)
            best = np.minimum(best, np.where(np.isfinite(d), d, np.inf))
    return best


def _seg_intersect(
    p1: np.ndarray, p2: np.ndarray, p3: np.ndarray, p4: np.ndarray
) -> tuple[bool, float]:
    """Segment p1p2 vs p3p4 intersection. Returns (hit, t_on_p3p4 in [0,1])."""
    d1, d2 = p2 - p1, p4 - p3
    denom = d1[0] * d2[1] - d1[1] * d2[0]
    if abs(denom) < 1e-12:
        return False, float("nan")
    s = ((p3[0] - p1[0]) * d2[1] - (p3[1] - p1[1]) * d2[0]) / denom  # on ball path
    t = ((p3[0] - p1[0]) * d1[1] - (p3[1] - p1[1]) * d1[0]) / denom  # on anchor seg
    # NOTE: s param is along ball vector, t along anchor; we need both in [0,1]
    # but return t (anchor projection) for the interior [0.15, 0.85] gate.
    if 0.0 <= s <= 1.0 and 0.0 <= t <= 1.0:
        return True, float(t)
    return False, float("nan")


def line_intersection(
    ball_px: np.ndarray,
    anchor_a_px: np.ndarray,
    anchor_b_px: np.ndarray,
    alpha,
    max_tether_distance_m: float = 3.0,
) -> list[dict]:
    """Split-pass events: ball path segment crosses anchor segment S_t.

    Gates: crossing point interior t_proj in [0.15, 0.85], anchor gap <= max gap.
    shortcut: spec example uses 0.25m which is far too tight for defenders;
    pass a realistic max (e.g. 3-5m) for rondo use.
    """
    ball = np.asarray(ball_px, dtype=float)
    A = np.asarray(anchor_a_px, dtype=float)
    B = np.asarray(anchor_b_px, dtype=float)
    T = len(ball)
    a = _as_alpha(alpha, T)
    events = []
    for t in range(1, T):
        if not all(np.isfinite([*ball[t - 1], *ball[t], *A[t], *B[t]])):
            continue
        gap_m = float(np.linalg.norm(A[t] - B[t]) * a[t])
        if gap_m > max_tether_distance_m:
            continue
        hit, t_proj = _seg_intersect(ball[t - 1], ball[t], A[t], B[t])
        # Collapse a multi-frame crossing into one event (keep first frame).
        if hit and 0.15 <= t_proj <= 0.85 and (not events or t - events[-1]["frame"] > 1):
            events.append({"frame": t, "split_gap_m": gap_m, "t_proj": t_proj})
    return events


def mask_kinematics(
    foot_px: np.ndarray, alpha, fps: float = 30.0, feature: str = "speed"
) -> np.ndarray:
    """Smoothed speed (m/s) or acceleration (m/s^2) from foot trajectory.

    Savitzky-Golay (window=5, polyorder=2) per spec; falls back to gradient if T<5.
    """
    p = np.asarray(foot_px, dtype=float)
    T = len(p)
    a = _as_alpha(alpha, T)
    dt = 1.0 / fps
    if T >= 5 and savgol_filter is not None:
        sm = savgol_filter(p, window_length=5, polyorder=2, axis=0, mode="interp")
        vel = np.gradient(sm, dt, axis=0) * a[:, None]
    else:
        vel = np.gradient(np.where(np.isfinite(p), p, 0.0), dt, axis=0) * a[:, None]
        vel[~np.isfinite(p).all(axis=1)] = np.nan
    speed = np.linalg.norm(vel, axis=1)
    if feature == "speed":
        return speed
    if feature in ("acceleration", "accel", "decel"):
        acc = np.gradient(np.where(np.isfinite(speed), speed, 0.0), dt)
        acc[~np.isfinite(speed)] = np.nan
        return acc
    raise ValueError(f"unknown kinematic_feature {feature!r}")


def strike_frame(ball_px: np.ndarray, alpha, fps: float = 30.0) -> int:
    """Impact frame t* = argmax ball acceleration magnitude."""
    return int(np.nanargmax(np.abs(mask_kinematics(ball_px, alpha, fps, "acceleration"))))


def mask_pose_axis(
    torso_lean_deg: np.ndarray,
    subj_foot_px: np.ndarray,
    ball_px: np.ndarray,
    alpha,
    at_frame: int | None = None,
) -> dict:
    """Torso lean + plant-foot distance at ball-strike frame."""
    lean = np.asarray(torso_lean_deg, dtype=float)
    p = np.asarray(subj_foot_px, dtype=float)
    b = np.asarray(ball_px, dtype=float)
    t = strike_frame(b, alpha) if at_frame is None else int(at_frame)
    a = _as_alpha(alpha, len(p))
    return {
        "frame": t,
        "torso_lean_deg": float(lean[t]),
        "plant_dist_m": float(np.linalg.norm(p[t] - b[t]) * a[t]),
    }
