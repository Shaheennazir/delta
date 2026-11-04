import numpy as np

from calibration import apply_homography, calibrate_from_ball, find_homography
from extractor import extract_features
from primitives import (
    line_intersection,
    mask_kinematics,
    proximity_timer,
    tether_distance_series,
)


def test_proximity_timer_bridges_dropout():
    T = 30
    foot = np.full((T, 2), [0.0, 0.0])
    ball = np.full((T, 2), [0.0, 0.0])
    ball[10:12] = [500.0, 500.0]  # 2-frame dropout
    w = proximity_timer(foot, ball, alpha=0.02, contact_radius_m=0.45, fps=30.0)
    assert len(w) == 1 and w[0]["start_frame"] == 0 and w[0]["end_frame"] == T - 1


def test_proximity_timer_split_on_long_gap():
    T = 30
    foot = np.full((T, 2), [0.0, 0.0])
    ball = np.full((T, 2), [0.0, 0.0])
    ball[10:20] = [500.0, 500.0]  # 10-frame gap > hysteresis
    w = proximity_timer(foot, ball, alpha=0.02, contact_radius_m=0.45, fps=30.0)
    assert len(w) == 2


def test_line_intersection_and_tether():
    ball = np.array([[0.0, 0.0], [5.0, 0.0], [10.0, 0.0]])
    a = np.array([[5.0, -5.0]] * 3)
    b = np.array([[5.0, 5.0]] * 3)
    evts = line_intersection(ball, a, b, alpha=0.1, max_tether_distance_m=5.0)
    assert len(evts) == 1 and evts[0]["frame"] == 1
    d = tether_distance_series(a, b, alpha=0.1)
    assert abs(d[0] - 1.0) < 1e-9


def test_mask_kinematics_speed():
    T = 20
    foot = np.column_stack([np.arange(T, dtype=float), np.zeros(T)])  # 1 px/frame
    speed = mask_kinematics(foot, alpha=0.5, fps=10.0, feature="speed")
    assert abs(np.nanmedian(speed) - 5.0) < 0.5  # 1px/frame * 0.5m/px * 10fps


def test_calibration_and_extractor():
    slope, intercept = calibrate_from_ball(
        np.array([314.0, 314.0]), np.array([100.0, 500.0])
    )
    assert abs(slope) < 1e-9 and intercept > 0
    H = find_homography(
        np.array([[0, 0], [4, 0], [4, 4], [0, 4]], float),
        np.array([[0, 0], [2, 0], [2, 2], [0, 2]], float),
    )
    assert np.allclose(apply_homography(H, [[2.0, 2.0]]), [[1.0, 1.0]])
    mask = np.zeros((30, 30), bool)
    mask[5:25, 10:20] = True
    f = extract_features(mask, alpha=0.02)
    assert f["stance_width_m"] > 0 and np.isfinite(f["torso_lean_deg"])
