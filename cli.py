"""Phase 1 CLI: run the deterministic kinematics engine on synthetic tracks.

Usage:
  python cli.py --self-check   # no video / API keys needed; exercises all 5 primitives
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from calibration import calibrate_from_ball, find_homography, apply_homography
from extractor import extract_features
from primitives import (
    line_intersection,
    mask_kinematics,
    mask_pose_axis,
    nearest_tether_series,
    proximity_timer,
    tether_distance_series,
)

FPS = 30.0
ALPHA = 0.02  # 2 cm/px synthetic scale


def synthetic_rondo() -> dict:
    rng = np.random.default_rng(0)
    T = 90
    # Ball: held by attacker 0 for 45 frames (1.5s dwell), then passed through defenders.
    ball = np.zeros((T, 2))
    ball[:45] = [100.0, 200.0] + rng.normal(0, 0.6, (45, 2))
    ball[45:] = np.column_stack([
        np.linspace(100, 160, 45),
        np.linspace(200, 200, 45),
    ])
    att0 = np.full((T, 2), [100.0, 201.0]) + rng.normal(0, 0.5, (T, 2))
    def1 = np.full((T, 2), [130.0, 190.0])
    def2 = np.full((T, 2), [130.0, 210.0])
    return {"ball": ball, "att0": att0, "def1": def1, "def2": def2}


def self_check() -> dict:
    d = synthetic_rondo()
    dwell = proximity_timer(d["att0"], d["ball"], ALPHA, contact_radius_m=0.45, fps=FPS)
    splits = line_intersection(d["ball"], d["def1"], d["def2"], ALPHA, max_tether_distance_m=3.0)
    tether = tether_distance_series(d["def1"], d["def2"], ALPHA)
    nearest = nearest_tether_series({"a": d["att0"]}, {"d1": d["def1"], "d2": d["def2"]}, ALPHA)
    speed = mask_kinematics(d["ball"], ALPHA, FPS, "speed")
    lean = np.full(len(d["ball"]), 12.0)
    pose = mask_pose_axis(lean, d["att0"], d["ball"], ALPHA)

    # Calibration + homography + extractor smoke checks on synthetic inputs.
    slope, intercept = calibrate_from_ball(
        np.array([300.0, 310.0, 290.0]), np.array([200.0, 400.0, 600.0])
    )
    H = find_homography(
        np.array([[0, 0], [100, 0], [100, 100], [0, 100]], float),
        np.array([[0, 0], [10, 0], [10, 10], [0, 10]], float),
    )
    proj = apply_homography(H, np.array([[50.0, 50.0]]))
    mask = np.zeros((40, 20), bool)
    mask[10:35, 5:15] = True
    feats = extract_features(mask, ALPHA)

    assert dwell and dwell[0]["duration_s"] > 1.0, "dwell should exceed 1.0s threshold"
    assert len(splits) >= 1, "pass should split the defenders"
    assert abs(float(proj[0, 0]) - 5.0) < 1e-6, "homography center maps to (5,5)"
    assert feats["stance_width_m"] > 0 and np.isfinite(feats["torso_lean_deg"])

    return {
        "dwell_windows": dwell,
        "avg_dwell_s": float(np.mean([w["duration_s"] for w in dwell])),
        "splits": splits,
        "defender_gap_m": float(np.median(tether)),
        "min_attacker_pressure_m": float(np.min(nearest)),
        "peak_ball_speed_m_s": float(np.nanmax(speed)),
        "strike_pose": pose,
        "scale_fit": {"slope": slope, "intercept": intercept},
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="delta Phase-1 kinematics CLI")
    ap.add_argument("--self-check", action="store_true")
    args = ap.parse_args()
    if not args.self_check:
        ap.error("only --self-check is supported in Phase 1 (no video ingest yet)")
    print(json.dumps(self_check(), indent=2))


if __name__ == "__main__":
    main()
