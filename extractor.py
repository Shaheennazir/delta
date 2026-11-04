"""Mask feature extractor (per spec section 4.2). Numpy-only, one mask = 2D bool array."""
from __future__ import annotations

import numpy as np


def _coords(mask: np.ndarray) -> np.ndarray:
    m = np.asarray(mask, dtype=bool)
    if not m.any():
        return np.zeros((0, 2))
    ys, xs = np.nonzero(m)
    return np.column_stack([xs.astype(float), ys.astype(float)])  # (x, y)


def foot_contact_point(mask: np.ndarray) -> tuple[float, float]:
    """Median (x, y) of bottom 5% of foreground pixels (y grows downward)."""
    pts = _coords(mask)
    if len(pts) == 0:
        return (float("nan"), float("nan"))
    k = max(1, int(len(pts) * 0.05))
    bottom = pts[np.argsort(pts[:, 1])[::-1][:k]]
    return (float(np.median(bottom[:, 0])), float(np.median(bottom[:, 1])))


def centroid(mask: np.ndarray) -> tuple[float, float]:
    pts = _coords(mask)
    if len(pts) == 0:
        return (float("nan"), float("nan"))
    return (float(pts[:, 0].mean()), float(pts[:, 1].mean()))


def stance_width_m(mask: np.ndarray, alpha: float = 1.0) -> float:
    """Horizontal span of bottom 20% of mask, scaled to meters."""
    pts = _coords(mask)
    if len(pts) == 0:
        return float("nan")
    k = max(1, int(len(pts) * 0.20))
    bottom = pts[np.argsort(pts[:, 1])[::-1][:k]]
    return float((bottom[:, 0].max() - bottom[:, 0].min()) * alpha)


def torso_axis_deg(mask: np.ndarray) -> float:
    """Torso lean from central moments of upper 70% of mask (spec formula)."""
    pts = _coords(mask)
    if len(pts) < 3:
        return float("nan")
    cutoff = np.quantile(pts[:, 1], 0.70)  # upper 70% = smaller y
    upper = pts[pts[:, 1] <= cutoff]
    if len(upper) < 3:
        upper = pts
    cx, cy = upper[:, 0].mean(), upper[:, 1].mean()
    dx, dy = upper[:, 0] - cx, upper[:, 1] - cy
    mu20 = float((dx * dx).mean())
    mu02 = float((dy * dy).mean())
    mu11 = float((dx * dy).mean())
    return float(np.degrees(0.5 * np.arctan2(2 * mu11, mu20 - mu02)))


def extract_features(mask: np.ndarray, alpha: float = 1.0) -> dict:
    foot = foot_contact_point(mask)
    center = centroid(mask)
    return {
        "foot_px": foot,
        "center_px": center,
        "stance_width_m": stance_width_m(mask, alpha),
        "torso_lean_deg": torso_axis_deg(mask),
    }
