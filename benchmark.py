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
    # 1. Order is timestamp-derived: every permutation of the same timestamped
    #    events must order identically (pure function of start/end/id).
    if len(events) >= 2:
        import random as _rnd
        fwd = temporal.order_events(events)
        fwd_ids = [e["event_id"] for e in fwd]
        fwd_keys = [(float(e.get("start_t", 0)), float(e.get("end_t", 0)),
                     str(e.get("event_id", ""))) for e in fwd]
        perms = [list(reversed(events))]
        perms.append(sorted(events, key=lambda e: str(e.get("event_id", ""))))
        perms.append(events[1:] + events[:1])  # rotation
        rng = _rnd.Random(0)
        for _ in range(3):
            p = list(events)
            rng.shuffle(p)
            perms.append(p)
        ok = all([e["event_id"] for e in temporal.order_events(p)] == fwd_ids
                 and [(float(e.get("start_t", 0)), float(e.get("end_t", 0)),
                       str(e.get("event_id", ""))) for e in temporal.order_events(p)] == fwd_keys
                 for p in perms)
        tests.append(_t("frame_shuffle", ok,
                        f"{len(events)} events identically ordered over "
                        f"{len(perms) + 1} permutations, not input position"))
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

    # 6. Occlusion recovery: same identity resumes, verified by an
    #    independent appearance check (recomputed descriptors, not the
    #    tracker's own claim) + spatial reachability. No other id may
    #    claim the resuming box.
    gaps = []
    for eid in {o["entity_id"] for o in observations}:
        rows = sorted((o for o in observations if o["entity_id"] == eid),
                      key=lambda o: float(o["t"]))
        for a, b in zip(rows, rows[1:]):
            if float(b["t"]) - float(a["t"]) > 1.5:
                gaps.append((eid, a, b))
    if gaps:
        import identity as ident_mod
        collisions, switches, implausible = 0, 0, 0
        details: list[str] = []
        by_fi = {int(r["frame_index"]): r for r in frames}
        for eid, before, after in gaps:
            for o in observations:
                if (o["entity_id"] != eid and int(o["frame_index"]) == int(after["frame_index"])
                        and _iou(o["bbox"], after["bbox"]) > 0.7):
                    collisions += 1
            dt = float(after["t"]) - float(before["t"])
            if not ident_mod.resume_plausible(before["bbox"], after["bbox"], dt):
                implausible += 1
                details.append(f"{eid} implausible resume over {dt:.1f}s")
            # Independent appearance re-check from stored crops.
            try:
                pb = by_fi.get(int(before["frame_index"]), {}).get("path", "")
                pa = by_fi.get(int(after["frame_index"]), {}).get("path", "")
                import os as _os
                if pb and pa and _os.path.exists(pb) and _os.path.exists(pa):
                    db = ident_mod.descriptor(pb, before["bbox"])
                    da = ident_mod.descriptor(pa, after["bbox"])
                    if db and da and ident_mod.cosine(db, da) < 0.50:
                        switches += 1
                        details.append(f"{eid} appearance switch "
                                       f"cos={ident_mod.cosine(db, da):.2f} over {dt:.1f}s")
            except Exception:
                pass
        ok = collisions == 0 and switches == 0 and implausible == 0
        tests.append(_t("occlusion", ok,
                        f"{len(gaps)} gap(s) >1.5s, {collisions} collision(s), "
                        f"{switches} appearance switch(es), {implausible} implausible; "
                        + "; ".join(details)[:200]))
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

    # 9. Interaction understanding: every multi-entity event must be grounded
    #    in co-present close boxes (precision), and sustained proximity must
    #    have produced an event (recall). No hallucinated or missed contact.
    eids = {o["entity_id"] for o in observations}
    if len(eids) >= 2 and observations and events:
        def _dist_norm(a: list[float], b: list[float]) -> float:
            ax, ay = (a[0] + a[2]) / 2.0, (a[1] + a[3]) / 2.0
            bx, by = (b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0
            da = max(1.0, ((a[2] - a[0]) ** 2 + (a[3] - a[1]) ** 2) ** 0.5)
            db = max(1.0, ((b[2] - b[0]) ** 2 + (b[3] - b[1]) ** 2) ** 0.5)
            return (((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5) / ((da + db) / 2.0)

        def _close(a: list[float], b: list[float]) -> bool:
            return _iou(a, b) > 0.02 or _dist_norm(a, b) < 1.2

        by_pair_frame: dict[tuple, dict[int, tuple]] = {}
        for o in observations:
            by_pair_frame.setdefault(o["entity_id"], {})[int(o["frame_index"])] = o
        id_list = sorted(eids)
        # Precision: each multi-entity event needs one supporting frame.
        hallucinated = 0
        multi = [e for e in events if len(e.get("entity_ids") or []) >= 2]
        for e in multi:
            ids_e = e.get("entity_ids") or []
            s, en = float(e.get("start_t", 0)), float(e.get("end_t", 0))
            ok = False
            for o in observations:
                if o["entity_id"] not in ids_e:
                    continue
                if not (s - 0.5 <= float(o["t"]) <= en + 0.5):
                    continue
                for o2 in observations:
                    if o2["entity_id"] not in ids_e or o2["entity_id"] == o["entity_id"]:
                        continue
                    if int(o2.get("frame_index", -1)) != int(o.get("frame_index", -1)):
                        continue
                    if _close(o["bbox"], o2["bbox"]):
                        ok = True
                        break
                if ok:
                    break
            if not ok:
                hallucinated += 1
        # Recall: sustained proximity (>=0.3s, gap<=1s) needs a covering event.
        missed = 0
        for i in range(len(id_list)):
            for j in range(i + 1, len(id_list)):
                a_id, b_id = id_list[i], id_list[j]
                ma = by_pair_frame.get(a_id, {})
                seq = []
                for fi in sorted(set(ma) & set(by_pair_frame.get(b_id, {}))):
                    oa, ob = ma[fi], by_pair_frame[b_id][fi]
                    if _close(oa["bbox"], ob["bbox"]):
                        seq.append((float(oa["t"] + ob["t"]) / 2.0, fi))
                runs: list[list] = []
                for t, fi in seq:
                    if runs and t - runs[-1][-1][0] <= 1.0:
                        runs[-1].append((t, fi))
                    else:
                        runs.append([(t, fi)])
                for run in runs:
                    if run[-1][0] - run[0][0] < 0.3 and len(run) < 2:
                        continue
                    rs, re = run[0][0], run[-1][0]
                    covered = any(
                        set([a_id, b_id]) <= set(ev.get("entity_ids") or [])
                        and float(ev.get("start_t", 0)) <= re + 0.5
                        and float(ev.get("end_t", 0)) >= rs - 0.5
                        for ev in events)
                    if not covered:
                        missed += 1
        if not multi and missed == 0:
            tests.append(_skip("interaction", "no sustained proximity to evaluate"))
        else:
            tests.append(_t("interaction", hallucinated == 0 and missed == 0,
                            f"{len(multi)} multi-entity event(s), "
                            f"{hallucinated} hallucinated, {missed} missed"))
    else:
        tests.append(_skip("interaction", "need >=2 identities with events"))

    # 9. Empty evidence must not verify.
    ok, reason = evmod.verify([], duration or 1.0, known)
    tests.append(_t("counterfactual", not ok, f"empty evidence rejected: {reason}"))

    # 10. Unsupported/unknown answers must abstain, regardless of model confidence.
    # Abstention confidence is certainty in the unknown status, not factual probability.
    abs_rec = evmod.abstain("Was there a helicopter?")
    gate = evmod.decision("unknown", 0.99, True)
    tests.append(_t("impossible_question",
                    abs_rec["abstained"] and abs_rec["verification"] == "unknown"
                    and abs_rec["confidence"] >= 0.5
                    and not abs_rec["evidence"] and gate["abstain"],
                    "abstains on unknown grounding even with 0.99 model confidence"))

    passed = sum(1 for t in tests if t["result"] == "pass")
    skipped = sum(1 for t in tests if t["result"] == "skip")
    failed = sum(1 for t in tests if t["result"] == "fail")
    evaluated = passed + failed  # applicable == evaluated; skips are not evidence
    coverage = [{"test": t["test"], "applicable": t["result"] != "skip",
                 "evaluated": t["result"] != "skip", "result": t["result"],
                 "reason": t.get("detail", "")} for t in tests]
    metrics = _session_metrics(events, observations, frames, duration or 0.0)
    return {"session_id": session_id, "tests": tests,
            "passed": passed, "skipped": skipped, "failed": failed,
            "total": len(tests), "applicable": evaluated, "evaluated": evaluated,
            "coverage": coverage, "metrics": metrics,
            "ablation": ablation_level(session_id)}


def _session_metrics(events: list[dict], observations: list[dict],
                     frames: list[dict], duration: float) -> dict:
    """Honest proxies, not ground truth. Identity precision/recall need labelled
    continuations; without them we report what is measurable: gap plausibility
    counts, and the unsupported-event rate from the evidence gate."""
    import identity as _ident
    import evidence as _ev
    known = {int(r.get("frame_index", -1)) for r in frames}
    times = {int(r.get("frame_index", -1)): float(r.get("t", 0.0)) for r in frames}
    gaps, implausible, switches = 0, 0, 0
    by_fi = {int(r.get("frame_index", -1)): r for r in frames}
    for eid in {o.get("entity_id") for o in observations}:
        rows = sorted((o for o in observations if o.get("entity_id") == eid),
                      key=lambda o: float(o.get("t", 0)))
        for a, b in zip(rows, rows[1:]):
            dt = float(b.get("t", 0)) - float(a.get("t", 0))
            if dt > 1.5:
                gaps += 1
                if not _ident.resume_plausible(a.get("bbox", [0, 0, 1, 1]),
                                               b.get("bbox", [0, 0, 1, 1]), dt):
                    implausible += 1
                    continue
                try:
                    import os as _os
                    pb = by_fi.get(int(a.get("frame_index", -1)), {}).get("path", "")
                    pa = by_fi.get(int(b.get("frame_index", -1)), {}).get("path", "")
                    if pb and pa and _os.path.exists(pb) and _os.path.exists(pa):
                        db = _ident.descriptor(pb, a.get("bbox", []))
                        da = _ident.descriptor(pa, b.get("bbox", []))
                        if db and da and _ident.cosine(db, da) < 0.50:
                            switches += 1
                except Exception:
                    pass
    bad_plausible = implausible + switches
    identity_precision_proxy = round(1.0 - bad_plausible / gaps, 4) if gaps else None
    # Recall needs ground-truth continuations: report unbridged gaps instead.
    unsupported, checkable = 0, 0
    for e in events:
        ev = e.get("evidence") or []
        if not ev:
            continue
        checkable += 1
        try:
            ok, _ = _ev.verify(ev, duration or 0.0, known, times)
        except Exception:
            ok = False
        if not ok:
            unsupported += 1
    return {"track_gaps_gt_1_5s": gaps, "implausible_resumes": implausible,
            "appearance_switches": switches,
            "identity_precision_proxy": identity_precision_proxy,
            "identity_recall": None,
            "identity_recall_note": "needs ground-truth continuations; "
                                    "unbridged gap count reported instead",
            "events_checkable": checkable,
            "events_unsupported": unsupported,
            "unsupported_event_rate": round(unsupported / checkable, 4) if checkable else None}


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
