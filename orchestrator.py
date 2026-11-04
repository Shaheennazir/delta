"""Pipeline: plan (Spark) -> perceive (SAM 3.1) -> kinematics -> review (Spark).

Two paths:
- Mock (no API key): deterministic demo tracks + mock plan/review. Offline-safe;
  every test runs here.
- Real (META_API_KEY set): upload/resolve video -> SAM streams per role ->
  real mask geometry (foot/center/lean, ball-scale calibration, Kalman ball
  fill) -> rule evaluation -> keyframe crops -> Spark reviewer.
"""
from __future__ import annotations

import asyncio
import itertools
import logging
import os
import uuid

import numpy as np

from calibration import calibrate_from_ball
from planner import mock_plan, plan_with_spark
from primitives import (
    kalman_fill_2d,
    line_intersection,
    mask_kinematics,
    mask_pose_axis,
    nearest_tether_series,
    proximity_timer,
    strike_frame,
    tether_distance_series,
)
from reviewer import mock_feedback, review_with_spark
from schemas import (
    CoachSessionInput,
    CoachingEvent,
    DrillAnalysisPlan,
    HudOverlay,
    SessionReport,
    VideoMeta,
)

log = logging.getLogger("delta.orchestrator")

ALPHA = 0.02
FPS_DEFAULT = 30.0
SEVERITY_COLOR = {"critical": "#EF4444", "warning": "#F59E0B", "info": "#38BDF8"}

TRACK_IDS = {"ball": [1], "attacker": [3, 4], "defender": [10, 11], "marker": [20]}

_store: dict[str, SessionReport] = {}
_lanes: dict[str, dict] = {}


def demo_tracks(seed: int = 0) -> dict[str, np.ndarray]:
    """Canonical synthetic Rondo tracks (single source; cli.py reuses this)."""
    rng = np.random.default_rng(seed)
    T = 90
    ball = np.zeros((T, 2))
    ball[:45] = [100.0, 200.0] + rng.normal(0, 0.6, (45, 2))
    ball[45:] = np.column_stack([np.linspace(100, 160, 45), np.full(45, 200.0)])
    return {
        "ball": ball,
        "attacker": np.full((T, 2), [100.0, 201.0]) + rng.normal(0, 0.5, (T, 2)),
        "defender_a": np.full((T, 2), [130.0, 190.0]),
        "defender_b": np.full((T, 2), [130.0, 210.0]),
    }


def _fires(measured: float, operator: str, threshold: float) -> bool:
    if operator == "gt":
        return measured > threshold
    if operator == "lt":
        return measured < threshold
    return abs(measured - threshold) < 1e-9  # eq


def _violation_windows(series: np.ndarray, operator: str, threshold: float) -> list[tuple[int, int]]:
    mask = np.array([_fires(float(v), operator, threshold) for v in series])
    mask = np.where(np.isfinite(series), mask, False)
    runs, s = [], None
    for t, on in enumerate(mask):
        if on and s is None:
            s = t
        elif not on and s is not None:
            runs.append((s, t - 1))
            s = None
    if s is not None:
        runs.append((s, len(mask) - 1))
    return runs


def _alpha_for(plan_tracks: dict, ball_scale, fps_len: int):
    """Per-frame m/px from ball-median-area regression; constant fallback."""
    if not ball_scale:
        return ALPHA
    areas, ys = ball_scale
    try:
        slope, intercept = calibrate_from_ball(np.asarray(areas), np.asarray(ys))
        ball = plan_tracks["ball"]
        T = len(ball)
        alpha = slope * np.nan_to_num(ball[:, 1], nan=np.nanmean(ys)) + intercept
        return np.where(np.isfinite(alpha) & (alpha > 0), alpha, ALPHA)
    except ValueError:
        return ALPHA


def evaluate_plan(plan: DrillAnalysisPlan, tracks: dict[str, np.ndarray], fps: float,
                  alpha=ALPHA, media: tuple | None = None,
                  lean: dict[str, np.ndarray] | None = None,
                  ball_scale=None) -> tuple[list[dict], dict[str, float]]:
    """Run metric_rules against tracks. Returns (raw_violations, summary)."""
    alpha = _alpha_for(tracks, ball_scale, len(tracks["ball"])) if ball_scale else alpha
    ball = tracks["ball"]
    lean = lean or {}
    violations: list[dict] = []
    dwell_all: list[dict] = []
    splits = 0
    for rule in plan.metric_rules:
        if rule.primitive == "proximity_timer":
            subj = tracks.get(rule.subject_role or "attacker", tracks["attacker"])
            tgt = tracks.get(rule.target_role or "ball", ball)
            dwell_all = proximity_timer(subj, kalman_fill_2d(tgt), alpha,
                                        rule.contact_radius_m, fps)
            for w in dwell_all:
                if _fires(w["duration_s"], rule.operator, rule.threshold):
                    violations.append({"rule": rule, "measured": w["duration_s"],
                                       "s": w["start_frame"], "e": w["end_frame"],
                                       "rkey": rule.subject_role or "attacker",
                                       "track": TRACK_IDS.get(rule.subject_role or "attacker", [3])[0]})
        elif rule.primitive == "line_intersection":
            if "defender_a" not in tracks or "defender_b" not in tracks:
                continue
            evts = line_intersection(kalman_fill_2d(ball), tracks["defender_a"],
                                     tracks["defender_b"], alpha, rule.max_tether_distance_m)
            splits = len(evts)
            for e in evts:
                if _fires(1.0, rule.operator, rule.threshold):
                    violations.append({"rule": rule, "measured": 1.0,
                                       "s": e["frame"], "e": e["frame"], "rkey": "ball",
                                       "track": 1, "extra_ids": TRACK_IDS["defender"]})
        elif rule.primitive == "tether_distance":
            a = tracks.get(rule.role_a or "attacker", tracks["attacker"])
            b = tracks.get(rule.role_b or "defender_a", tracks.get("defender_a", tracks["attacker"]))
            series = tether_distance_series(a, b, alpha)
            for s, e in _violation_windows(series, rule.operator, rule.threshold):
                peak = float(np.nanmax(series[s:e + 1]))
                violations.append({"rule": rule, "measured": peak, "s": s, "e": e,
                                   "rkey": rule.role_a or "attacker", "track": 3})
        elif rule.primitive == "mask_kinematics":
            tgt = tracks.get(rule.target_role or "attacker", tracks["attacker"])
            series = mask_kinematics(tgt, alpha, fps, rule.kinematic_feature)
            for s, e in _violation_windows(series, rule.operator, rule.threshold):
                peak = float(np.nanmax(np.abs(series[s:e + 1])))
                violations.append({"rule": rule, "measured": peak, "s": s, "e": e,
                                   "rkey": rule.target_role or "attacker", "track": 3})
        elif rule.primitive == "mask_pose_axis":
            subj = tracks.get(rule.subject_role or "attacker", tracks["attacker"])
            lean_arr = lean.get(rule.subject_role or "attacker",
                                np.full(len(ball), np.nan))
            filled = kalman_fill_2d(ball)
            try:
                pose = mask_pose_axis(lean_arr, subj, filled, alpha)
            except (ValueError, IndexError):
                continue
            m = pose["torso_lean_deg"] if (rule.measure or "") != "plant_dist_m" else pose["plant_dist_m"]
            if np.isfinite(m) and _fires(m, rule.operator, rule.threshold):
                violations.append({"rule": rule, "measured": m, "s": pose["frame"],
                                   "e": pose["frame"], "rkey": rule.subject_role or "attacker",
                                   "track": 3})
    total = len(dwell_all) if dwell_all else len(violations)
    summary = {
        "total_touches": float(len(dwell_all)),
        "avg_dwell_s": float(np.mean([w["duration_s"] for w in dwell_all])) if dwell_all else 0.0,
        "defensive_splits": float(splits),
        "compliance_score": float(max(0.0, min(100.0, 100.0 * (1 - len(violations) / max(total, 1))))),
    }
    return violations, summary


def _anchor(tracks: dict, rkey: str, frame: int, media: tuple | None) -> tuple[float | None, float | None]:
    if not media or rkey not in tracks:
        return None, None
    w, h = media
    if not w or not h:
        return None, None
    p = tracks[rkey][min(frame, len(tracks[rkey]) - 1)]
    if not np.isfinite(p).all():
        return None, None
    return round(float(p[0] / w), 4), round(float(p[1] / h), 4)


def _build_events(violations: list[dict], tracks: dict, fps: float,
                  media: tuple | None, feedback: dict[str, str] | None = None) -> list[CoachingEvent]:
    counter = itertools.count(1)
    events = []
    for v in violations:
        rule = v["rule"]
        n = next(counter)
        t_s, t_e = v["s"] / fps, v["e"] / fps
        ax, ay = _anchor(tracks, v.get("rkey", ""), v["s"], media)
        key = f"evt_{n:02d}"
        fb = (feedback or {}).get(key) or mock_feedback(
            rule.rule_id, float(v["measured"]), rule.threshold, v["track"], t_s)
        involved = [v["track"], TRACK_IDS["ball"][0]] if v["track"] != 1 else \
            [1, *v.get("extra_ids", [])][:3]
        events.append(CoachingEvent(
            event_id=key, rule_id=rule.rule_id,
            timestamp_start_s=round(t_s, 2), timestamp_end_s=round(t_e, 2),
            measured_value=round(float(v["measured"]), 3),
            involved_track_ids=involved,
            hud_overlay=HudOverlay(
                target_track_id=v["track"],
                color=SEVERITY_COLOR[rule.severity],
                label=f"{rule.name}: {v['measured']:.2f} (limit {rule.threshold})",
                anchor_x=ax, anchor_y=ay,
            ),
            coach_feedback=fb,
        ))
    return events


def run_pipeline(inp: CoachSessionInput, tracks: dict[str, np.ndarray] | None = None,
                 plan: DrillAnalysisPlan | None = None) -> SessionReport:
    """Deterministic sync pipeline (mock feedback). API background-task core."""
    tracks = tracks or demo_tracks()
    plan = plan or mock_plan(inp)
    fps = inp.fps or FPS_DEFAULT
    violations, summary = evaluate_plan(plan, tracks, fps)
    session_id = f"ses_{uuid.uuid4().hex[:8]}"
    return SessionReport(session_id=session_id, status="completed",
                         summary_metrics=summary,
                         events=_build_events(violations, tracks, fps, None),
                         plan=plan)


# ---- Real perception (SAM 3.1) ----

async def track_concept(file_id: str, role: str, prompt: str, n_frames: int,
                        progress=None) -> dict:
    """One SAM stream for a concept prompt. Returns parsed result + tracks."""
    from sam_client import stream_segmentation, tracks_from_result

    result = await stream_segmentation(file_id, prompt)
    if progress:
        progress(role, result)
    return {"role": role, "prompt": prompt, "result": result,
            "parsed": tracks_from_result(result, n_frames)}


async def execute_perception_pass(file_id: str, concept_prompts: list,
                                  n_frames: int, progress=None) -> dict[str, dict]:
    """Concurrent SAM streams, one per unique role (spec section 6)."""
    seen: dict[str, str] = {}
    for c in concept_prompts:
        role = c["role"] if isinstance(c, dict) else c.role
        prompt = c["prompt"] if isinstance(c, dict) else c.prompt
        seen.setdefault(role, prompt)
    results = await asyncio.gather(*[
        track_concept(file_id, role, prompt, n_frames, progress)
        for role, prompt in seen.items()
    ])
    return {r["role"]: r for r in results}


def _assemble_tracks(perceived: dict[str, dict], n_frames: int) -> tuple[dict, dict, tuple | None]:
    """Role streams -> primitive-ready tracks + lean map + ball scale samples."""
    tracks: dict[str, np.ndarray] = {}
    lean: dict[str, np.ndarray] = {}
    by_role: dict[str, list] = {}
    for role, p in perceived.items():
        objs = p["parsed"]["objects"]
        by_role[role] = sorted(objs.values(), key=lambda o: o["n_frames"], reverse=True)
    T = n_frames

    def pad(arr: np.ndarray) -> np.ndarray:
        if len(arr) == T:
            return arr
        out = np.full((T, 2), np.nan)
        out[:min(T, len(arr))] = arr[:T]
        return out

    def pad1(arr: np.ndarray) -> np.ndarray:
        if len(arr) == T:
            return arr
        out = np.full(T, np.nan)
        out[:min(T, len(arr))] = arr[:T]
        return out

    ball_objs = by_role.get("ball", [])
    if ball_objs:
        b = ball_objs[0]
        tracks["ball"] = kalman_fill_2d(pad(b["center_px"]))
        areas = b["area_px"]
        lean["ball"] = pad1(b["lean_deg"])
    else:
        tracks["ball"] = np.full((T, 2), np.nan)
        areas = np.full(T, np.nan)
    ball_scale = (areas, tracks["ball"][:, 1])

    att = by_role.get("attacker", [])
    tracks["attacker"] = pad(att[0]["foot_px"]) if att else np.full((T, 2), np.nan)
    lean["attacker"] = pad1(att[0]["lean_deg"]) if att else np.full(T, np.nan)
    defs = by_role.get("defender", [])
    tracks["defender_a"] = pad(defs[0]["foot_px"]) if len(defs) > 0 else np.full((T, 2), np.nan)
    tracks["defender_b"] = pad(defs[1]["foot_px"]) if len(defs) > 1 else np.full((T, 2), np.nan)
    return tracks, lean, ball_scale


async def run_pipeline_async(inp: CoachSessionInput, progress_cb=None,
                             session_id: str | None = None) -> SessionReport:
    """Full async pipeline; mock path when no key, real SAM+Spark otherwise."""
    from sam_client import NoApiKey, api_key, resolve_video

    session_id = session_id or f"ses_{uuid.uuid4().hex[:8]}"
    save_lanes(session_id, {"width": 0, "height": 0, "fps": inp.fps or FPS_DEFAULT, "lanes": {}})

    def progress(role, result):
        entry = _lanes.get(session_id, {"lanes": {}})
        entry["lanes"][role] = result.raw_output
        save_lanes(session_id, entry)
        if progress_cb:
            progress_cb({"role": role, "revision": result.revision,
                         "records": len(result.records)})

    try:
        api_key()
    except NoApiKey:
        plan = await plan_with_spark(inp)
        report = run_pipeline(inp, tracks=demo_tracks(), plan=plan)
        violations = [{"key": e.event_id, "rule_id": e.rule_id, "measured": e.measured_value,
                       "threshold": next(r.threshold for r in plan.metric_rules if r.rule_id == e.rule_id),
                       "track_id": e.involved_track_ids[0], "t": e.timestamp_start_s}
                      for e in report.events]
        if violations:
            fb = await review_with_spark(inp.drill_setup, violations)
            for e in report.events:
                if e.event_id in fb:
                    e.coach_feedback = fb[e.event_id]
        report.session_id = session_id
        return report

    from media import crop_around, frame_to_jpeg, probe, read_frame

    file_id, video_bytes = await resolve_video(inp.video_url)
    meta = probe(video_bytes) if video_bytes else {}
    fps = meta.get("fps") or inp.fps or FPS_DEFAULT
    n_frames = meta.get("n_frames") or int((meta.get("duration_s") or 30) * fps)
    W, H = meta.get("width", 0), meta.get("height", 0)

    frame0 = frame_to_jpeg(read_frame(video_bytes, 0)) if video_bytes else None
    plan = await plan_with_spark(inp, frame0)

    lanes: dict[str, str] = {}

    perceived = await execute_perception_pass(
        file_id, [{"role": p.role, "prompt": p.prompt} for p in plan.sam_concept_prompts],
        n_frames, progress)
    lanes = dict(_lanes.get(session_id, {}).get("lanes", {}))
    for role, p in perceived.items():
        lanes[role] = p["result"].raw_output

    tracks, lean, ball_scale = _assemble_tracks(perceived, n_frames)
    violations, summary = evaluate_plan(plan, tracks, fps, media=(W, H),
                                        lean=lean, ball_scale=ball_scale)

    crops: dict[str, str] = {}
    frames = {}
    if video_bytes:
        for v in violations:
            key = f"evt_{violations.index(v) + 1:02d}"
            fidx = min(v["s"], n_frames - 1)
            if fidx not in frames:
                frames[fidx] = read_frame(video_bytes, fidx)
            foot = tracks.get(v.get("rkey", ""), np.full((1, 2), np.nan))[min(v["s"], n_frames - 1)]
            cx, cy = (float(foot[0]), float(foot[1])) if np.isfinite(foot).all() else (W / 2, H / 2)
            crops[key] = frame_to_jpeg(crop_around(frames[fidx], cx, cy))
    fb_requests = [{"key": f"evt_{i + 1:02d}", "rule_id": v["rule"].rule_id,
                    "measured": float(v["measured"]), "threshold": v["rule"].threshold,
                    "track_id": v["track"], "t": v["s"] / fps} for i, v in enumerate(violations)]
    feedback = await review_with_spark(inp.drill_setup, fb_requests, crops) if fb_requests else {}

    report = SessionReport(
        session_id=session_id, status="completed", summary_metrics=summary,
        events=_build_events(violations, tracks, fps, (W, H), feedback),
        plan=plan, video=VideoMeta(width=W, height=H, fps=fps,
                                   duration_s=meta.get("duration_s", n_frames / fps)),
    )
    save_lanes(session_id, {"width": W, "height": H, "fps": fps, "lanes": lanes})
    return report


def save_report(report: SessionReport) -> None:
    _store[report.session_id] = report


def get_report(session_id: str) -> SessionReport | None:
    return _store.get(session_id)


def save_lanes(session_id: str, payload: dict) -> None:
    _lanes[session_id] = payload


def get_lanes(session_id: str) -> dict | None:
    return _lanes.get(session_id)
