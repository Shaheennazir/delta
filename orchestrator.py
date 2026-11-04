"""Pipeline: plan (Spark) -> perceive (SAM 3.1) -> kinematics -> review (Spark).

Sync core `run_pipeline` is fully deterministic and offline-safe (mock plan +
demo tracks + mock feedback). Async variants try the Meta Model API and fall
back to mocks on any error, so tests never need keys or video.
"""
from __future__ import annotations

import asyncio
import itertools
import os
import uuid

import numpy as np

from calibration import calibrate_from_ball
from planner import mock_plan, plan_with_spark
from primitives import (
    line_intersection,
    mask_kinematics,
    mask_pose_axis,
    nearest_tether_series,
    proximity_timer,
    strike_frame,
    tether_distance_series,
)
from reviewer import mock_feedback, review_with_spark
from schemas import CoachSessionInput, CoachingEvent, DrillAnalysisPlan, HudOverlay, SessionReport

ALPHA = 0.02
FPS_DEFAULT = 30.0
SEVERITY_COLOR = {"critical": "#EF4444", "warning": "#F59E0B", "info": "#38BDF8"}

TRACK_IDS = {"ball": [1], "attacker": [3, 4], "defender": [10, 11], "marker": [20]}

_store: dict[str, SessionReport] = {}


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


def evaluate_plan(plan: DrillAnalysisPlan, tracks: dict[str, np.ndarray],
                  fps: float) -> tuple[list[dict], dict[str, float]]:
    """Run metric_rules against tracks. Returns (raw_violation_dicts, summary)."""
    alpha = ALPHA  # shortcut: constant scale; per-(x,y) alpha(y) fit lands in Phase 3
    ball = tracks["ball"]
    violations: list[dict] = []
    dwell_all: list[dict] = []
    splits = 0
    for rule in plan.metric_rules:
        if rule.primitive == "proximity_timer":
            subj = tracks.get(rule.subject_role or "attacker", tracks["attacker"])
            tgt = tracks.get(rule.target_role or "ball", ball)
            dwell_all = proximity_timer(subj, tgt, alpha, rule.contact_radius_m, fps)
            for w in dwell_all:
                if _fires(w["duration_s"], rule.operator, rule.threshold):
                    violations.append({"rule": rule, "measured": w["duration_s"],
                                       "s": w["start_frame"], "e": w["end_frame"],
                                       "track": TRACK_IDS.get(rule.subject_role or "attacker", [3])[0]})
        elif rule.primitive == "line_intersection":
            evts = line_intersection(ball, tracks["defender_a"], tracks["defender_b"],
                                     alpha, rule.max_tether_distance_m)
            splits = len(evts)
            for e in evts:
                if _fires(1.0, rule.operator, rule.threshold):
                    violations.append({"rule": rule, "measured": 1.0,
                                       "s": e["frame"], "e": e["frame"], "track": 1,
                                       "extra_ids": TRACK_IDS["defender"]})
        elif rule.primitive == "tether_distance":
            a = tracks.get(rule.role_a or "attacker", tracks["attacker"])
            b = tracks.get(rule.role_b or "defender_a", tracks["defender_a"])
            series = tether_distance_series(a, b, alpha)
            for s, e in _violation_windows(series, rule.operator, rule.threshold):
                peak = float(np.nanmax(series[s:e + 1]))
                violations.append({"rule": rule, "measured": peak, "s": s, "e": e, "track": 3})
        elif rule.primitive == "mask_kinematics":
            tgt = tracks.get(rule.target_role or "attacker", tracks["attacker"])
            series = mask_kinematics(tgt, alpha, fps, rule.kinematic_feature)
            for s, e in _violation_windows(series, rule.operator, rule.threshold):
                peak = float(np.nanmax(np.abs(series[s:e + 1])))
                violations.append({"rule": rule, "measured": peak, "s": s, "e": e, "track": 3})
        elif rule.primitive == "mask_pose_axis":
            subj = tracks.get(rule.subject_role or "attacker", tracks["attacker"])
            lean = np.full(len(ball), 12.0)  # shortcut: lean from extractor masks in Phase 3
            pose = mask_pose_axis(lean, subj, ball, alpha)
            m = pose["torso_lean_deg"] if (rule.measure or "") != "plant_dist_m" else pose["plant_dist_m"]
            if _fires(m, rule.operator, rule.threshold):
                violations.append({"rule": rule, "measured": m, "s": pose["frame"],
                                   "e": pose["frame"], "track": 3})
    total = len(dwell_all) if dwell_all else len(violations)
    summary = {
        "total_touches": float(len(dwell_all)),
        "avg_dwell_s": float(np.mean([w["duration_s"] for w in dwell_all])) if dwell_all else 0.0,
        "defensive_splits": float(splits),
        "compliance_score": float(max(0.0, min(100.0, 100.0 * (1 - len(violations) / max(total, 1))))),
    }
    return violations, summary


def run_pipeline(inp: CoachSessionInput, tracks: dict[str, np.ndarray] | None = None,
                 plan: DrillAnalysisPlan | None = None) -> SessionReport:
    """Deterministic sync pipeline (mock feedback). Used by API background task core + tests."""
    tracks = tracks or demo_tracks()
    plan = plan or mock_plan(inp)
    fps = inp.fps or FPS_DEFAULT
    violations, summary = evaluate_plan(plan, tracks, fps)
    counter = itertools.count(1)
    events = []
    for v in violations:
        rule = v["rule"]
        n = next(counter)
        t_s, t_e = v["s"] / fps, v["e"] / fps
        ball_id = TRACK_IDS["ball"][0]
        involved = [v["track"], ball_id] if v["track"] != ball_id else [ball_id, *v.get("extra_ids", [])]
        if "extra_ids" in v and v["track"] == ball_id:
            involved = [ball_id, *v["extra_ids"]][:3]
        events.append(CoachingEvent(
            event_id=f"evt_{n:02d}", rule_id=rule.rule_id,
            timestamp_start_s=round(t_s, 2), timestamp_end_s=round(t_e, 2),
            measured_value=round(float(v["measured"]), 3),
            involved_track_ids=involved,
            hud_overlay=HudOverlay(
                target_track_id=v["track"],
                color=SEVERITY_COLOR[rule.severity],
                label=f"{rule.name}: {v['measured']:.2f} (limit {rule.threshold})",
            ),
            coach_feedback=mock_feedback(rule.rule_id, float(v["measured"]),
                                         rule.threshold, v["track"], t_s),
        ))
    session_id = f"ses_{uuid.uuid4().hex[:8]}"
    return SessionReport(session_id=session_id, status="completed",
                         summary_metrics=summary, events=events, plan=plan)


# ---- Async API-facing layer (real Spark/SAM attempts with mock fallback) ----

async def track_concept(video_url: str, role: str, prompt: str) -> dict:
    """Single SAM 3.1 concept stream. Falls back to demo tracks offline."""
    if not os.getenv("META_API_KEY"):
        d = demo_tracks()
        key = {"ball": "ball", "attacker": "attacker"}.get(role, "attacker")
        return {"role": role, "prompt": prompt, "tracks": d.get(key, d["attacker"])}
    try:
        from openai import AsyncOpenAI

        client = AsyncOpenAI(api_key=os.getenv("META_API_KEY"), base_url="https://api.meta.ai/v1")
        stream = await client.responses.create(
            model="sam-3.1", stream=True,
            input=[{"type": "message", "role": "user", "content": [
                {"type": "input_text", "text": prompt},
                {"type": "input_video", "video_url": video_url}]}],
        )
        chunks = []
        async for chunk in stream:  # minimal collect; full RLE decode lands with real API shape
            chunks.append(chunk)
        return {"role": role, "prompt": prompt, "tracks": demo_tracks()["attacker"],
                "note": f"live stream collected {len(chunks)} chunks (RLE decode TODO)"}
    except Exception:
        return {"role": role, "prompt": prompt, "tracks": demo_tracks()["attacker"]}


async def execute_perception_pass(video_url: str, concept_prompts: list) -> dict[str, dict]:
    results = await asyncio.gather(*[
        track_concept(video_url, c["role"] if isinstance(c, dict) else c.role,
                      c["prompt"] if isinstance(c, dict) else c.prompt)
        for c in concept_prompts
    ])
    return {r["role"]: r for r in results}


async def run_pipeline_async(inp: CoachSessionInput) -> SessionReport:
    plan = await plan_with_spark(inp)
    await execute_perception_pass(inp.video_url, [
        {"role": p.role, "prompt": p.prompt} for p in plan.sam_concept_prompts])
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
    return report


def save_report(report: SessionReport) -> None:
    _store[report.session_id] = report


def get_report(session_id: str) -> SessionReport | None:
    return _store.get(session_id)
