"""Offline engine tests: temporal algebra, evidence gate, observer, audio, store.

No API key needed — all deterministic local computation.
"""
import io

import imageio.v3 as iio
import numpy as np


def _mp4(frames: int = 12, w: int = 160, h: int = 128, motion_at: int = 6) -> bytes:
    arr = np.zeros((frames, h, w, 3), dtype=np.uint8)
    arr[motion_at:] = 200  # hard scene change halfway
    buf = io.BytesIO()
    iio.imwrite(buf, arr, extension=".mp4", fps=10)
    return buf.getvalue()


def test_allen_relations():
    import temporal
    assert temporal.relation(0, 1, 2, 3) == "before"
    assert temporal.relation(2, 3, 0, 1) == "after"
    assert temporal.relation(0, 3, 1, 2) == "contains"
    assert temporal.relation(1, 2, 0, 3) == "during"
    assert temporal.relation(0, 2, 1, 3) == "overlaps"
    assert temporal.relation(1, 1, 1, 1) == "equals"
    assert temporal.before(0, 1, 2, 3) and not temporal.before(0, 3, 1, 2)
    assert temporal.overlaps(0, 3, 1, 2) and not temporal.overlaps(0, 1, 2, 3)
    assert temporal.gap(1.0, 2.5) == 1.5
    assert temporal.duration(2.0, 5.0) == 3.0


def test_temporal_order_and_contradictions():
    import temporal
    evs = [{"event_id": "E02", "start_t": 5.0, "end_t": 6.0},
           {"event_id": "E01", "start_t": 1.0, "end_t": 2.0}]
    ordered = temporal.order_events(evs)
    assert [e["event_id"] for e in ordered] == ["E01", "E02"]
    assert temporal.find_before(evs, 3.0) and temporal.find_after(evs, 3.0)
    assert temporal.find_between(evs, 2.0, 5.0) == []
    assert temporal.find_containing(evs, 1.5)[0]["event_id"] == "E01"
    bad = [{"event_id": "E01", "start_t": 5.0, "end_t": 3.0},
           {"event_id": "E01", "start_t": 1.0, "end_t": 2.0}]
    assert len(temporal.contradictions(bad)) == 2  # end<=start + duplicate
    clamped = temporal.validate_clamp(
        [{"event_id": "E01", "start_t": -2.0, "end_t": 99.0}], 10.0)
    assert clamped[0]["start_t"] == 0.0 and clamped[0]["end_t"] == 10.0


def test_evidence_gate_and_abstain():
    import evidence
    ok, _ = evidence.verify([{"start": 0.0, "end": 1.0, "frame_ids": [3]}], 10.0, {3, 4})
    assert ok
    bad, reason = evidence.verify([{"start": 0.0, "end": 1.0, "frame_ids": [99]}], 10.0, {3, 4})
    assert not bad and "not in frame store" in reason
    bad2, _ = evidence.verify([{"start": 0.0, "end": 99.0, "frame_ids": []}], 10.0, set())
    assert not bad2
    assert evidence.calibrate(0.9, 0, 0) == 0.0
    assert 0.0 < evidence.calibrate(0.9, 1, 2) <= 1.0
    rec = evidence.abstain("Was there a helicopter?")
    assert rec["abstained"] and rec["verification"] == "unknown"
    assert rec["confidence"] >= 0.5  # status-certainty, not factual probability


def test_observer_adaptive_plan():
    import observer
    data = _mp4()
    track, meta = observer.probe_motion(data)
    assert len(track) >= 2
    assert any(p["scene_cut"] for p in track), "hard cut must be detected"
    plan = observer.plan_sampling(track, 1.2, base_fps=10.0, max_frames=60)
    assert plan and all(p["t"] <= 1.2 for p in plan)
    tiers = {p["tier"] for p in plan}
    assert "transition" in tiers, f"cut neighbourhood must densify, got {tiers}"
    rows = observer.materialise_frames(data, plan[:4], "test_ses")
    assert rows and all("path" in r for r in rows)
    assert [r["frame_index"] for r in rows] == [p["frame_index"] for p in plan[:4]]


def test_audio_events_deterministic():
    import audio_events
    assert audio_events.detect(None) == []
    assert audio_events.detect(b"RIFFxxxx") == []
    # Synthetic WAV: quiet bed with one loud burst.
    import struct
    sr, secs = 24000, 3
    n = sr * secs
    import math
    samples = [int(2000 * math.sin(2 * math.pi * 440 * i / sr)) for i in range(n)]
    for i in range(sr, sr + sr // 10):  # 100ms burst
        samples[i] = 25000
    raw = struct.pack(f"<{n}h", *samples)
    fmt_body = struct.pack("<HHIIHH", 1, 1, sr, sr * 2, 2, 16)
    wav = (b"RIFF" + struct.pack("<I", 36 + len(raw)) + b"WAVE"
           + b"fmt " + struct.pack("<I", 16) + fmt_body
           + b"data" + struct.pack("<I", len(raw)) + raw)
    evs = audio_events.detect(wav)
    assert any(e["kind"] in ("impact", "loud-event") for e in evs), evs


def test_store_roundtrip(tmp_path=None):
    import store
    store.save_frames("t1", [{"frame_index": 0, "t": 0.0, "path": "p", "motion": 0.1, "keyframe": 1}])
    assert store.get_frames("t1")[0]["frame_index"] == 0
    store.save_entities("t1", [{"entity_id": "person_001", "label": "p", "kind": "person", "confidence": 0.9}],
                        [{"entity_id": "person_001", "t": 0.0, "frame_index": 0,
                          "bbox": [0, 0, 10, 10], "track_conf": 0.9}])
    ents, obs = store.get_entities("t1")
    assert ents[0]["entity_id"] == "person_001" and obs[0]["bbox"] is not None
    store.save_events("t1", [{"event_id": "E01", "action": "walk", "entity_ids": ["person_001"],
                              "start_t": 0.0, "end_t": 1.0, "evidence": [],
                              "confidence": 0.8, "state_before": {}, "state_after": {},
                              "detail": "walks"}])
    assert store.get_events("t1")[0]["action"] == "walk"
    out = __import__("benchmark").run_all("t1")
    assert out["total"] == 11 and out["failed"] == 0, out
