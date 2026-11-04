"""End-to-end capability checks (offline, deterministic, no always-pass tests)."""
from __future__ import annotations

import math
import struct
import uuid

import numpy as np
from PIL import Image


def _wav(burst_times: list[float], seconds: float = 6.0, sr: int = 24000) -> bytes:
    n = int(seconds * sr)
    x = np.full(n, 0.01, dtype=np.float32)
    for t in burst_times:
        a = max(0, int((t - 0.1) * sr))
        b = min(n, int((t + 0.2) * sr))
        x[a:b] = 0.8
    raw = struct.pack(f"<{n}h", *(np.clip(x, -1, 1) * 32767).astype(np.int16))
    fmt = struct.pack("<HHIIHH", 1, 1, sr, sr * 2, 2, 16)
    return (b"RIFF" + struct.pack("<I", 36 + len(raw)) + b"WAVE"
            + b"fmt " + struct.pack("<I", 16) + fmt
            + b"data" + struct.pack("<I", len(raw)) + raw)


def _img(path, color) -> str:
    arr = np.full((200, 200, 3), 245, dtype=np.uint8)
    arr[40:160, 40:160] = color
    Image.fromarray(arr).save(path, format="JPEG")
    return str(path)


def test_identity_is_instance_level_and_cross_session(tmp_path):
    import identity
    import store
    p = _img(tmp_path / "red.jpg", (220, 30, 30))
    bbox = [40, 40, 160, 160]
    kind = f"testperson_{uuid.uuid4().hex[:6]}"  # isolate this gallery slice
    rows = [{"frame_index": 0, "t": 0.0, "path": p, "motion": 0.1}]
    cands = [
        {"candidate_id": "a", "label": "red person", "kind": kind, "confidence": 0.8,
         "observations": [{"entity_id": "", "t": 0.0, "frame_index": 0, "bbox": bbox,
                           "track_conf": 0.9}]},
        {"candidate_id": "b", "label": "red person", "kind": kind, "confidence": 0.7,
         "observations": [{"entity_id": "", "t": 0.2, "frame_index": 1, "bbox": bbox,
                           "track_conf": 0.9}]},
    ]
    first = identity.assign([dict(cands[0])], rows, (200, 200), "sess_a")
    second = identity.assign([dict(cands[1])], rows + [{"frame_index": 1, "t": 0.2, "path": p}],
                             (200, 200), "sess_b")
    assert first[0]["global_id"] == second[0]["global_id"], "same appearance must keep identity"
    assert first[0]["identity_score"] == 0.0 or second[0]["identity_score"] > 0.8
    # Different colour is not force-matched.
    q = _img(tmp_path / "blue.jpg", (20, 40, 220))
    diff = identity.assign([{"candidate_id": "c", "label": "blue", "kind": kind,
                             "confidence": 0.8,
                             "observations": [{"entity_id": "", "t": 0.0, "frame_index": 0,
                                               "bbox": bbox, "track_conf": 0.9}]}],
                           [{"frame_index": 0, "t": 0.0, "path": q}], (200, 200), "sess_c")
    assert diff[0]["global_id"] != first[0]["global_id"]
    # Persisted registry is real, not process memory.
    assert any(g["global_id"] == first[0]["global_id"] for g in store.get_gallery(kind))


def test_av_alignment_measures_lag():
    import av_align
    base = [1.0, 2.0, 3.0, 4.0]
    wav = _wav([t + 0.5 for t in base])          # audio is 500 ms late
    rows = [{"t": i / 10.0, "motion": 0.02, "frame_index": i}
            for i in range(61)]
    for t in base:
        for r in rows:
            if abs(r["t"] - t) <= 0.05:
                r["motion"] = 1.0
    out = av_align.estimate_offset(wav, rows, 6.0)
    assert abs(out["offset_s"] - 0.5) <= 0.1, out
    assert out["confidence"] > 0.2, out
    shifted = av_align.events_offset(
        [{"start_t": t + 0.5, "end_t": t + 0.7} for t in base], rows, 6.0)
    assert abs(shifted["offset_s"] - 0.5) <= 0.1, shifted


def test_events_are_signal_derived_with_typed_state():
    import events
    ents = [{"entity_id": "g1", "label": "person", "kind": "person", "confidence": 0.9}]
    obs = [
        {"entity_id": "g1", "t": 0.0, "frame_index": 0, "bbox": [10, 10, 30, 50], "track_conf": 0.9},
        {"entity_id": "g1", "t": 0.5, "frame_index": 5, "bbox": [80, 10, 100, 50], "track_conf": 0.9},
        {"entity_id": "g1", "t": 1.0, "frame_index": 10, "bbox": [150, 10, 170, 50], "track_conf": 0.9},
        {"entity_id": "g1", "t": 4.0, "frame_index": 40, "bbox": [10, 10, 30, 50], "track_conf": 0.9},
    ]
    frames = [{"frame_index": i * 5, "t": i * 0.5} for i in range(0, 9)]
    cands = events.build_candidates(
        ents, obs, [{"startMs": 1000, "endMs": 1800, "transcript": "go", "speaker": "1"}],
        [{"t": 2.0, "frame_index": 20, "text": "GOAL", "confidence": 0.9}],
        [{"event_id": "au_1", "kind": "impact", "start_t": 3.0, "end_t": 3.2,
          "confidence": 0.8, "detail": "hit"}], 6.0, frames)
    assert cands, "signals must produce candidates"
    assert all(c["end_t"] > c["start_t"] for c in cands)
    # Timestamps come from observations/signals, never a model guess.
    assert any(c["start_t"] == 0.0 and c["end_t"] == 1.0 for c in cands)
    assert any(c["state_after"].get("presence") == "present" for c in cands)
    assert any(c["state_after"].get("direction") == "right" for c in cands)
    assert any(c["source"] == "speech" and c["start_t"] == 1.0 for c in cands)
    assert any(c["source"] == "ocr" and c["state_after"].get("on_screen_text") == "GOAL" for c in cands)
    timeline = events.state_timeline([
        {"entity_ids": ["g1"], "start_t": 0, "end_t": 1, "state_after": {"presence": "present"}},
        {"entity_ids": ["g1"], "start_t": 1, "end_t": 2, "state_after": {"presence": "present"}},
        {"entity_ids": ["g1"], "start_t": 2, "end_t": 3, "state_after": {"presence": "absent"}},
    ])
    assert timeline[0]["start"] == 0 and timeline[0]["end"] == 2


def test_state_contradictions_are_detected():
    import temporal
    evs = [
        {"event_id": "E1", "entity_ids": ["g1"], "start_t": 1.0, "end_t": 3.0,
         "state_after": {"direction": "left"}},
        {"event_id": "E2", "entity_ids": ["g1"], "start_t": 2.0, "end_t": 4.0,
         "state_after": {"direction": "right"}},
        {"event_id": "E3", "entity_ids": ["g1"], "start_t": 9.0, "end_t": 10.0,
         "state_after": {"direction": "left"}},
    ]
    problems = temporal.contradictions(evs, duration=8.0)
    assert any("both 'left'" in p and "'right'" in p for p in problems), problems
    assert any("outside" in p for p in problems), problems


def test_retrieval_uses_exact_index_spans():
    import retrieval
    frames = [{"frame_index": i, "t": float(i)} for i in range(10)]
    events = [
        {"event_id": "E01", "action": "player moves right", "detail": "crosses",
         "entity_ids": ["g1"], "start_t": 4.5, "end_t": 6.0, "confidence": 0.9,
         "evidence": [{"frame_ids": [5]}]},
        {"event_id": "E02", "action": "ball appears", "detail": "throw-in",
         "entity_ids": [], "start_t": 1.0, "end_t": 2.0, "confidence": 0.9,
         "evidence": [{"frame_ids": [1]}]},
    ]
    ents = [{"entity_id": "g1", "label": "player", "kind": "person", "confidence": 0.9}]
    obs = [{"entity_id": "g1", "t": 4.6, "frame_index": 5, "bbox": [0, 0, 1, 1]}]
    items = retrieval.retrieve("q", "what did the player do at 5s?", events, ents, obs,
                               [], [], [], frames, 10.0)
    assert items[0]["id"] == "E01", items
    assert items[0]["start"] == 4.5 and items[0]["end"] == 6.0
    assert retrieval.time_hint("between 2 and 4 seconds", 10.0) == (2.0, 4.0)


def test_evidence_checks_frame_time_and_abstains():
    import evidence
    frames = {3: 3.0, 4: 4.0}
    ok, _ = evidence.verify([{"start": 2.5, "end": 4.5, "frame_ids": [3, 4]}],
                            10.0, {3, 4}, frames)
    assert ok
    bad, reason = evidence.verify([{"start": 7.0, "end": 8.0, "frame_ids": [3]}],
                                  10.0, {3, 4}, frames)
    assert not bad and "outside span" in reason, reason
    assert evidence.decision("unknown", 0.99, True)["abstain"]
    assert evidence.decision("supported", 0.9, True)["abstain"] is False
    assert evidence.decision("supported", 0.01, True)["abstain"] is True
    assert evidence.abstain("x")["abstained"] is True


def test_calibration_fits_isotonic_and_selects_threshold():
    import confidence
    import store
    pairs = [(0.1, 0), (0.2, 0), (0.3, 0), (0.6, 1), (0.8, 1), (0.9, 1)]
    points = confidence.fit_isotonic(pairs)
    xs = [p["x"] for p in points]
    ys = [p["y"] for p in points]
    assert xs == sorted(xs) and ys == sorted(ys)
    assert 0.0 <= confidence.apply_curve(points, 0.6) <= 1.0
    assert confidence.expected_calibration_error(pairs) >= 0.0
    sel = confidence.selective_threshold(pairs, 0.1)
    assert sel["risk"] <= 0.1 and sel["coverage"] > 0
    # Persist through the same path the feedback endpoint uses.
    import uuid as _uuid
    for i, (x, y) in enumerate(pairs):
        store.save_outcome(f"cal_{_uuid.uuid4().hex[:6]}", f"i{i}", x, bool(y))
    out = confidence.fit_from_outcomes()
    assert out["n"] >= len(pairs) and out["points"]


def test_benchmark_has_no_unconditional_passes():
    import benchmark
    import store
    sid = f"bench_{uuid.uuid4().hex[:6]}"
    store.upsert_session(sid, status="completed", has_audio=1, transcript="hi",
                         video_json='{"duration_s": 6.0}')
    store.save_frames(sid, [
        {"frame_index": i, "t": float(i), "path": "", "motion": 0.1, "keyframe": 1}
        for i in range(7)])
    store.save_entities(sid, [
        {"entity_id": "g1", "label": "person", "kind": "person", "confidence": 0.9}], [
        {"entity_id": "g1", "t": 0.0, "frame_index": 0, "bbox": [0, 0, 10, 10], "track_conf": 0.9},
        {"entity_id": "g1", "t": 2.0, "frame_index": 2, "bbox": [2, 0, 12, 10], "track_conf": 0.9},
        {"entity_id": "g1", "t": 4.0, "frame_index": 4, "bbox": [4, 0, 14, 10], "track_conf": 0.9},
    ])
    store.save_events(sid, [
        {"event_id": "E01", "action": "walk", "entity_ids": ["g1"], "start_t": 0.0,
         "end_t": 2.0, "evidence": [], "confidence": 0.8, "state_before": {},
         "state_after": {"presence": "present"}, "detail": ""},
        {"event_id": "E02", "action": "stop", "entity_ids": ["g1"], "start_t": 3.0,
         "end_t": 4.0, "evidence": [], "confidence": 0.8, "state_before": {},
         "state_after": {"motion": "still"}, "detail": ""},
    ])
    store.save_audio_events(sid, [
        {"event_id": "au_1", "kind": "impact", "start_t": 1.0, "end_t": 1.2,
         "confidence": 0.8, "detail": ""}])
    out = benchmark.run_all(sid)
    assert out["total"] == 10
    assert {t["result"] for t in out["tests"]} <= {"pass", "fail", "skip"}
    assert out["failed"] == 0, out
    by = {t["test"]: t for t in out["tests"]}
    assert by["frame_shuffle"]["result"] == "pass"
    assert by["frame_deletion"]["result"] == "pass"
    assert by["counterfactual"]["result"] == "pass"
    assert by["impossible_question"]["result"] == "pass"
    assert all(t["detail"] for t in out["tests"])
    # Empty session must not pretend capabilities it cannot evaluate.
    empty = benchmark.run_all(f"empty_{uuid.uuid4().hex[:6]}")
    assert empty["total"] == 10 and empty["failed"] == 0
    assert sum(1 for t in empty["tests"] if t["result"] == "skip") >= 7
