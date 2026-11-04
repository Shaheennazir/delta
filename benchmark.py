"""Adversarial evaluation: each test derives its verdict from stored data.

No unconditional passes. A test that cannot be evaluated on this session is
reported as "skip" with the missing signal — never as a pass.
"""
from __future__ import annotations

import temporal
import evidence as evmod


def _t(name: str, ok: bool, detail: str = "") -> dict:
    return {"test": name, "result": "pass" if ok else "fail", "detail": detail[:300]}


def _skip(name: str, why: str) -> dict:
    return {"test": name, "result": "skip", "detail": why[:300]}


def _iou(a: list[float], b: list[float]) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    aa = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    bb = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return inter / max(1e-9, aa + bb - inter)


def run_all(session_id: str) -> dict:
    import store
    events = store.get_events(session_id)
    entities, observations = store.get_entities(session_id)
    frames = store.get_frames(session_id)
    audio = store.get_audio_events(session_id)
    ocr = store.get_ocr(session_id)
    sess = store.get_session(session_id) or {}
    known = {int(r["frame_index"]) for r in frames}
    frame_times = {int(r["frame_index"]): float(r.get("t", 0)) for r in frames}
    duration = 0.0
    try:
        import json
        duration = float((json.loads(sess.get("video_json") or "{}")).get("duration_s", 0) or 0.0)
    except Exception:
        duration = 0.0

    tests: list[dict] = []
    # 1. Order is timestamp-derived: reversing input must not change output order.
    if len(events) >= 2:
        rev = temporal.order_events(list(reversed(events)))
        fwd = temporal.order_events(events)
        ok = [e["event_id"] for e in rev] == [e["event_id"] for e in fwd]
        tests.append(_t("frame_shuffle", ok,
                        f"{len(events)} events reordered by timestamp, not input position"))
    else:
        tests.append(_skip("frame_shuffle", "need >=2 events"))

    # 2. A frame id absent from the index must fail verification.
    if frames:
        ok, reason = evmod.verify([{"start": 0.0, "end": min(1.0, duration or 1.0),
                                    "frame_ids": [-999999]}], duration or 1.0, known)
        tests.append(_t("frame_deletion", not ok, f"unknown frame rejected: {reason}"))
    else:
        tests.append(_skip("frame_deletion", "no frame index"))

    # 3. No audio must mean no speech/audio rows — not an invented transcript.
    if "has_audio" in sess:
        if not int(sess.get("has_audio") or 0):
            clean = not (sess.get("transcript") or "").strip() and not audio
            tests.append(_t("audio_removal", clean,
                            f"no-audio session: transcript_empty={not bool((sess.get('transcript') or '').strip())}, "
                            f"audio_events={len(audio)}"))
        else:
            in_range = all(0 <= float(a.get("start_t", 0)) <= float(a.get("end_t", 0)) <= duration + 0.01
                           for a in audio)
            tests.append(_t("audio_removal", in_range,
                            f"audio present: {len(audio)} events inside duration"))
    else:
        tests.append(_skip("audio_removal", "no session row"))

    # 4. Shift audio by +2 s; measured alignment must move by the same amount.
    if audio and frames and duration > 4 and len(frames) >= 4:
        import av_align
        base = av_align.events_offset(audio, frames, duration)
        shifted = [{**a, "start_t": float(a["start_t"]) + 2.0,
                    "end_t": float(a["end_t"]) + 2.0} for a in audio]
        shifted = [a for a in shifted if a["end_t"] <= duration]
        moved = av_align.events_offset(shifted, frames, duration) if shifted else {}
        if base.get("confidence", 0) >= 0.1 and moved.get("confidence", 0) >= 0.1:
            delta = float(moved["offset_s"]) - float(base["offset_s"])
            tests.append(_t("audio_video_desync", abs(delta - 2.0) <= 0.5,
                            f"alignment moved {delta:+.2f}s after +2s audio shift "
                            f"(base {base['offset_s']:+.2f}, shifted {moved['offset_s']:+.2f})"))
        else:
            tests.append(_skip("audio_video_desync",
                               f"weak alignment signal ({base.get('confidence', 0):.2f}/"
                               f"{moved.get('confidence', 0):.2f})"))
    else:
        tests.append(_skip("audio_video_desync", "need audio events, frames, duration"))

    # 5. Two identities must not occupy the same box in the same frame.
    if len({o["entity_id"] for o in observations}) >= 2:
        by_frame: dict[int, list[dict]] = {}
        for o in observations:
            by_frame.setdefault(int(o.get("frame_index", 0)), []).append(o)
        worst = 0.0
        for rows in by_frame.values():
            for i in range(len(rows)):
                for j in range(i + 1, len(rows)):
                    if rows[i]["entity_id"] != rows[j]["entity_id"]:
                        worst = max(worst, _iou(rows[i]["bbox"], rows[j]["bbox"]))
        tests.append(_t("similar_objects", worst < 0.9,
                        f"max same-frame cross-identity IoU {worst:.2f}"))
    else:
        tests.append(_skip("similar_objects", "need >=2 identities"))

    # 6. Across an occlusion gap, the same identity must resume; no other id may
    #    claim the resuming box.
    gaps = []
    for eid in {o["entity_id"] for o in observations}:
        rows = sorted((o for o in observations if o["entity_id"] == eid),
                      key=lambda o: float(o["t"]))
        for a, b in zip(rows, rows[1:]):
            if float(b["t"]) - float(a["t"]) > 1.5:
                gaps.append((eid, a, b))
    if gaps:
        collisions = 0
        for eid, before, after in gaps:
            for o in observations:
                if (o["entity_id"] != eid and int(o["frame_index"]) == int(after["frame_index"])
                        and _iou(o["bbox"], after["bbox"]) > 0.7):
                    collisions += 1
        tests.append(_t("occlusion", collisions == 0,
                        f"{len(gaps)} gap(s) >1.5s, {collisions} identity collision(s)"))
    else:
        tests.append(_skip("occlusion", "no track gap >1.5s"))

    # 7. OCR attribution: every row's cited frame must exist and sit at its time.
    if ocr:
        bad = []
        for r in ocr:
            fi = int(r.get("frame_index", -1))
            if fi not in frame_times or not (float(r["t"]) - 0.5 <= frame_times[fi] <= float(r["t"]) + 1.0):
                bad.append(r)
        tests.append(_t("rapid_text", not bad,
                        f"{len(ocr)} OCR rows, {len(bad)} temporally mismatched"))
    else:
        tests.append(_skip("rapid_text", "no OCR rows"))

    # 8. Reversed spans must contradict.
    if events:
        bad = [dict(events[0], start_t=events[0]["end_t"], end_t=events[0]["start_t"])]
        probs = temporal.contradictions(bad, duration)
        tests.append(_t("temporal_reversal", len(probs) > 0, "; ".join(probs)[:200]))
    else:
        tests.append(_skip("temporal_reversal", "no events"))

    # 9. Empty evidence must not verify.
    ok, reason = evmod.verify([], duration or 1.0, known)
    tests.append(_t("counterfactual", not ok, f"empty evidence rejected: {reason}"))

    # 10. Unsupported/unknown answers must abstain, regardless of model confidence.
    abs_rec = evmod.abstain("Was there a helicopter?")
    gate = evmod.decision("unknown", 0.99, True)
    tests.append(_t("impossible_question",
                    abs_rec["abstained"] and abs_rec["confidence"] == 0.0
                    and not abs_rec["evidence"] and gate["abstain"],
                    "abstains on unknown grounding even with 0.99 model confidence"))

    passed = sum(1 for t in tests if t["result"] == "pass")
    skipped = sum(1 for t in tests if t["result"] == "skip")
    failed = sum(1 for t in tests if t["result"] == "fail")
    return {"session_id": session_id, "tests": tests,
            "passed": passed, "skipped": skipped, "failed": failed,
            "total": len(tests), "ablation": ablation_level(session_id)}


def ablation_level(session_id: str) -> dict:
    """Which of the 4 ablation systems this session corresponds to."""
    import store
    frames = store.get_frames(session_id)
    entities, _ = store.get_entities(session_id)
    events = store.get_events(session_id)
    qa = store.get_qa(session_id)
    level = 1
    if frames:
        level = 2
    if frames and (entities or events):
        level = 3
    if frames and events and qa:
        level = 4
    return {"level": level,
            "labels": {1: "video LLM alone", 2: "LLM + adaptive frame selection",
                       3: "LLM + tracking/event graph/temporal index",
                       4: "full agentic + evidence retrieval/verification"},
            "components": {"frames": len(frames), "entities": len(entities),
                           "events": len(events), "qa_runs": len(qa)}}
