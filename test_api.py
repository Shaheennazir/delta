import time

from fastapi.testclient import TestClient

from app import app
from orchestrator import demo_tracks, evaluate_plan, run_pipeline
from planner import mock_plan
from schemas import CoachSessionInput

client = TestClient(app)

PAYLOAD = {
    "drill_title": "4v2 High-Tempo Rondo",
    "drill_setup": "4 attackers on the perimeter, 2 defenders pressing inside a 10x10m grid. Two-touch maximum.",
    "analysis_goals": "Flag whenever an attacker holds the ball longer than 1.0 second, and detect when a pass splits the two defenders.",
    "video_url": "https://storage.pitchmind.io/clips/session_9821.mp4",
    "fps": 30.0,
}


def test_mock_plan_capped_to_known_primitives():
    plan = mock_plan(CoachSessionInput(**PAYLOAD))
    assert 1 <= len(plan.metric_rules) <= 4
    assert {p.prompt for p in plan.sam_concept_prompts} >= {"soccer ball"}


def test_pipeline_produces_coaching_events():
    report = run_pipeline(CoachSessionInput(**PAYLOAD), tracks=demo_tracks())
    assert report.status == "completed"
    assert report.summary_metrics["defensive_splits"] >= 1
    dwell = [e for e in report.events if e.rule_id == "dwell_time_check"]
    assert dwell and dwell[0].measured_value > 1.0
    assert all("Cue:" in e.coach_feedback for e in report.events)


def test_api_create_and_read():
    r = client.post("/api/v1/sessions", json=PAYLOAD)
    assert r.status_code == 202
    sid = r.json()["session_id"]
    for _ in range(50):  # background task completes async
        g = client.get(f"/api/v1/sessions/{sid}")
        assert g.status_code == 200
        if g.json()["status"] == "completed":
            break
        time.sleep(0.05)
    body = client.get(f"/api/v1/sessions/{sid}").json()
    assert body["status"] == "completed"
    assert len(body["events"]) >= 2


def test_api_rejects_bad_payload():
    r = client.post("/api/v1/sessions", json={"drill_title": "x"})
    assert r.status_code == 422
    r = client.post("/api/v1/sessions", json={**PAYLOAD, "evil": "print(1)"})
    assert r.status_code == 422  # schema sandboxing: no extras


def test_evaluate_all_five_primitives():
    import numpy as np

    plan = mock_plan(CoachSessionInput(**{**PAYLOAD, "analysis_goals": (
        "hold ball dwell, split pass line, sprint speed, strike lean posture")}))
    assert len(plan.metric_rules) == 4
    primitives = {r.primitive for r in plan.metric_rules}
    assert "proximity_timer" in primitives and "line_intersection" in primitives
    violations, summary = evaluate_plan(plan, demo_tracks(), 30.0)
    assert summary["compliance_score"] < 100.0 and violations
