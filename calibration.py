"""Pixel-to-meter scale calibration for single-camera footage.

Two modes (per spec section 4.1):
 1. ball_diameter (default): alpha = R / sqrt(A / pi), regressed vs image y.
 2. homography (optional): 4-cone planar projection via DLT (numpy, no cv2).
"""
from __future__ import annotations

import numpy as np

BALL_RADIUS_M = 0.11


def ball_alphas(areas_px: np.ndarray, ball_radius_m: float = BALL_RADIUS_M) -> np.ndarray:
    """Per-frame scale from ball mask areas: R / sqrt(A / pi)."""
    areas = np.asarray(areas_px, dtype=float)
    return ball_radius_m / np.sqrt(areas / np.pi)


def fit_alpha_vs_y(alphas: np.ndarray, ys: np.ndarray) -> tuple[float, float]:
    """Least-squares fit alpha(y) = slope * y + intercept (perspective compensation).

    Returns (slope, intercept). Needs >= 2 samples; single sample -> flat line.
    """
    a = np.asarray(alphas, dtype=float)
    y = np.asarray(ys, dtype=float)
    mask = np.isfinite(a) & np.isfinite(y) & (a > 0)
    a, y = a[mask], y[mask]
    if a.size == 0:
        raise ValueError("no valid ball samples for scale calibration")
    if a.size == 1:
        return 0.0, float(a[0])
    A = np.column_stack([y, np.ones_like(y)])
    (slope, intercept), *_ = np.linalg.lstsq(A, a, rcond=None)
    return float(slope), float(intercept)


def scale_at_y(y: float, slope: float, intercept: float) -> float:
    return float(slope * y + intercept)


def calibrate_from_ball(
    areas_px: np.ndarray, ys: np.ndarray, ball_radius_m: float = BALL_RADIUS_M
) -> tuple[float, float]:
    """One-call ball-diameter calibration -> (slope, intercept)."""
    return fit_alpha_vs_y(ball_alphas(areas_px, ball_radius_m), ys)


def find_homography(src_px: np.ndarray, dst_m: np.ndarray) -> np.ndarray:
    """DLT homography H (3x3) mapping src_px -> dst_m. Each input is 4x2 (or Nx2, N>=4)."""
    src = np.asarray(src_px, dtype=float)
    dst = np.asarray(dst_m, dtype=float)
    if src.shape[0] < 4 or dst.shape[0] < 4 or src.shape != dst.shape:
        raise ValueError("need >=4 matching src/dst point pairs")
    rows = []
    for (x, y), (X, Y) in zip(src, dst):
        rows.append([-x, -y, -1, 0, 0, 0, X * x, X * y, X])
        rows.append([0, 0, 0, -x, -y, -1, Y * x, Y * y, Y])
    _, _, vt = np.linalg.svd(np.asarray(rows, dtype=float))
    return (vt[-1] / vt[-1, -1]).reshape(3, 3)


def apply_homography(H: np.ndarray, pts_px: np.ndarray) -> np.ndarray:
    """Project (N,2) pixel points to meters via H."""
    pts = np.asarray(pts_px, dtype=float)
    homo = np.column_stack([pts, np.ones(len(pts))]) @ np.asarray(H).T
    return homo[:, :2] / homo[:, 2:3]
