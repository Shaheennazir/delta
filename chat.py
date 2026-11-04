"""Session chat: follow-up answers from saved analysis, intent-routed via Spark.

Routes (one cheap Spark classifier call):
  - followup:    answer from stored analysis + prior QA only (no video rewatch).
  - investigate: factual video question -> investigator.answer_question (retrieval-gated).
  - reanalyze:   new lens/task -> fresh intel.analyze_video pass on the same upload.

Every route persists to qa_runs so history survives restarts, and logs
sent/waiting/received to the trace store (visible under the internals toggle).
"""
from __future__ import annotations

import json
import logging
import uuid

log = logging.getLogger("delta.chat")

CLASSIFY_TIMEOUT = 60.0
FOLLOWUP_TIMEOUT = 90.0


async def _classify(message: str, analysis: str, history: list[dict]) -> dict:
    from spark import SPARK_MODEL, extract_text, post_unary
    hist = "\n".join(
        f"Q: {h.get('question','')[:160]}\nA: {h.get('answer','')[:300]}"
        for h in history[-6:]
    ) or "(no follow-ups yet)"
    body = await post_unary({
        "model": SPARK_MODEL,
        "instructions": "Route a video-chat follow-up. Strict JSON.",
        "input": [{"type": "message", "role": "user", "content": [{"type": "input_text", "text":
            f"Stored video analysis:\n{analysis[:4000]}\n\nPrior follow-ups:\n{hist[:2000]}\n\n"
            f"New user message: {message}\n\n"
            "Route: \"followup\" (clarify/summarise/explain the existing analysis, no new "
            "visual evidence needed) | \"investigate\" (specific factual who/what/when/where "
            "about the video, answerable from stored evidence) | \"reanalyze\" (new task, "
            "different lens, recount, full transcription, or anything the stored analysis "
            "does not cover). Return {\"route\": ..., \"reason\": str}."}]}],
        "text": {"format": {"type": "json_schema", "name": "Route",
                 "schema": {"type": "object",
                            "properties": {"route": {"type": "string"},
                                           "reason": {"type": "string"}},
                            "required": ["route", "reason"],
                            "additionalProperties": False}, "strict": True}},
    })
    try:
        d = json.loads(extract_text(body))
        route = str(d.get("route", "investigate")).strip().lower()
        if route not in ("followup", "investigate", "reanalyze"):
            route = "investigate"
        return {"route": route, "reason": str(d.get("reason", ""))[:300]}
    except Exception:
        return {"route": "investigate", "reason": "classifier fallback"}


async def _followup(session_id: str, message: str, analysis: str,
                    history: list[dict]) -> dict:
    import asyncio
    import store
    from spark import SPARK_MODEL, extract_text, post_unary
    hist = "\n".join(
        f"Q: {h.get('question','')[:200]}\nA: {h.get('answer','')[:500]}"
        for h in history[-6:]
    ) or "(none)"
    body = await asyncio.wait_for(post_unary({
        "model": SPARK_MODEL,
        "instructions": "Answer ONLY from the stored video analysis and prior answers. "
                        "Never invent unseen detail. Strict JSON.",
        "input": [{"type": "message", "role": "user", "content": [{"type": "input_text", "text":
            f"Stored analysis:\n{analysis[:5000]}\n\nPrior follow-ups:\n{hist[:2500]}\n\n"
            f"User: {message}\nReturn {{\"answer\": str, \"confidence\": 0-1}}. "
            "If the analysis does not cover it, say so in the answer with low confidence."}]}],
        "text": {"format": {"type": "json_schema", "name": "Followup",
                 "schema": {"type": "object",
                            "properties": {"answer": {"type": "string"},
                                           "confidence": {"type": "number"}},
                            "required": ["answer", "confidence"],
                            "additionalProperties": False}, "strict": True}},
    }), timeout=FOLLOWUP_TIMEOUT)
    try:
        d = json.loads(extract_text(body))
        answer = str(d.get("answer", ""))[:2000]
        conf = max(0.0, min(1.0, float(d.get("confidence", 0.5))))
    except Exception:
        answer, conf = "", 0.0
    if not answer:
        from evidence import abstain
        rec = abstain(message, "stored analysis does not cover this")
        rec["route"] = "followup"
        qa_id = f"qa_{uuid.uuid4().hex[:8]}"
        store.save_qa(session_id, qa_id, message, rec["answer"], [],
                      rec["confidence"], "unknown", rec["raw_confidence"])
        return {"qa_id": qa_id, "question": message, **rec}
    qa_id = f"qa_{uuid.uuid4().hex[:8]}"
    store.save_qa(session_id, qa_id, message, answer, [], conf, "inferred", conf)
    return {"qa_id": qa_id, "question": message, "answer": answer, "evidence": [],
            "confidence": conf, "raw_confidence": conf, "verification": "inferred",
            "abstained": False, "route": "followup"}


async def _reanalyze(session_id: str, message: str) -> dict:
    """Fresh full-video Spark pass with the new focus; verified against stored frames."""
    import asyncio
    import store
    import agent
    from sam_client import resolve_video
    sess = store.get_session(session_id) or {}
    video_url = sess.get("video_url") or ""
    try:
        turns = json.loads(sess.get("turns_json") or "[]")
        transcript = sess.get("transcript") or ""
        video = json.loads(sess.get("video_json") or "{}")
        duration = float(video.get("duration_s") or 0)
    except Exception:
        turns, transcript, duration = [], "", 0.0
    frames = store.get_frames(session_id)
    known = {int(r["frame_index"]) for r in frames}
    ftimes = {int(r["frame_index"]): float(r["t"]) for r in frames}

    agent.trace(session_id, "reanalyze", "sent", f"new focus -> Spark: {message[:100]}")
    agent.trace(session_id, "reanalyze", "waiting", "Spark re-watching video…")
    file_id, _ = await asyncio.wait_for(
        asyncio.to_thread(resolve_video, video_url), timeout=agent.UPLOAD_TIMEOUT)
    from intel import analyze_video
    intel = await asyncio.wait_for(
        analyze_video(file_id, message, transcript, turns, duration or 0.0),
        timeout=agent.UNDERSTAND_TIMEOUT)
    agent.trace(session_id, "reanalyze", "received",
                f"answer + {len(intel.get('moments', []))} moment(s)")

    import evidence as evmod
    spans = []
    for i, c in enumerate(intel.get("answer_claims", [])[:8]):
        fids = [int(r["frame_index"]) for r in frames
                if float(c["start_t"]) - 0.5 <= float(r.get("t", 0)) <= float(c["end_t"]) + 0.5][:6]
        spans.append({"start": round(float(c["start_t"]), 3), "end": round(float(c["end_t"]), 3),
                      "frame_ids": fids, "source_id": f"re_a_{i+1:02d}", "kind": "reanalysis",
                      "verification": "supported" if fids else "inferred"})
    ok, reason = evmod.verify(spans, duration, known, ftimes) if spans else (False, "no claims")
    support = len(spans) / max(1, len(intel.get("answer_claims", []) or [1]))
    detail = evmod.calibrate_detail(
        sum(c["confidence"] for c in intel.get("answer_claims", [])) / max(1, len(spans)) if spans else 0.0,
        len(spans), sum(len(s["frame_ids"]) for s in spans), support)
    verification = ("supported" if ok and any(s["verification"] == "supported" for s in spans)
                    else "inferred" if ok and spans else "unknown")
    gate = evmod.decision(verification, detail["calibrated"],
                          bool(intel.get("answer")) and bool(spans))
    qa_id = f"qa_{uuid.uuid4().hex[:8]}"
    if gate["abstain"]:
        rec = evmod.abstain(message, gate["reason"])
        rec["route"] = "reanalyze"
        store.save_qa(session_id, qa_id, message, rec["answer"], [],
                      rec["confidence"], "unknown", rec["raw_confidence"])
        return {"qa_id": qa_id, "question": message, **rec}
    store.save_qa(session_id, qa_id, message, intel["answer"], spans,
                  detail["calibrated"], verification, detail["raw"])
    return {"qa_id": qa_id, "question": message, "answer": intel["answer"],
            "evidence": spans, "confidence": detail["calibrated"],
            "raw_confidence": detail["raw"], "verification": verification,
            "abstained": False, "calibration": detail["method"],
            "summary": intel.get("summary", ""), "route": "reanalyze",
            "verify_reason": reason}


def _tokens(s: str) -> set[str]:
    import re
    return set(re.findall(r"[a-z0-9]+", s.lower()))


def _prior_abstention(message: str, history: list[dict]) -> dict | None:
    """Most recent prior QA that abstained on the same fact.

    Same fact = normalized exact match, or high token overlap at similar
    length. A substantially longer/shorter rephrase is treated as new
    evidence or a changed question and is NOT cached.
    """
    want = _tokens(message)
    if not want:
        return None
    for h in reversed(history):
        q = str(h.get("question", ""))
        if not q:
            continue
        ver = str(h.get("verification", ""))
        ans = str(h.get("answer", ""))
        abstained = ver == "unknown" or ans.startswith("Insufficient video evidence")
        if not abstained:
            continue
        if q.strip().lower() == message.strip().lower():
            return h
        got = _tokens(q)
        if not got:
            continue
        inter = len(want & got)
        union = len(want | got)
        jacc = inter / union if union else 0.0
        contain = inter / min(len(want), len(got))
        # New detail changes the length a lot -> rephrase/new evidence, don't cache.
        if max(len(want), len(got)) > 1.5 * min(len(want), len(got)):
            continue
        if jacc >= 0.5 or contain >= 0.8:
            return h
    return None


async def answer_chat(session_id: str, message: str) -> dict:
    """Intent-route one chat message; persist + return the answer record."""
    import asyncio
    import store
    import agent
    import investigator
    sess = store.get_session(session_id) or {}
    history = store.get_qa(session_id)
    analysis = investigator._analysis_context(sess)

    agent.trace(session_id, "chat_router", "sent", message[:160])
    prior = _prior_abstention(message, history)
    if prior is not None:
        detail = ("repeat of previously unanswerable question "
                  f"(qa {prior.get('qa_id', '?')}) — no reanalysis; "
                  "provide new evidence or rephrase to reopen")
        agent.trace(session_id, "chat_router", "received", detail[:300])
        qa_id = f"qa_{uuid.uuid4().hex[:8]}"
        store.save_qa(session_id, qa_id, message, str(prior.get("answer", "")),
                      list(prior.get("evidence") or []),
                      float(prior.get("confidence") or 0.0),
                      str(prior.get("verification") or "unknown"),
                      float(prior.get("raw_confidence") or prior.get("confidence") or 0.0))
        return {"qa_id": qa_id, "question": message,
                "answer": str(prior.get("answer", "")),
                "evidence": list(prior.get("evidence") or []),
                "confidence": float(prior.get("confidence") or 0.0),
                "raw_confidence": float(prior.get("raw_confidence") or prior.get("confidence") or 0.0),
                "verification": str(prior.get("verification") or "unknown"),
                "abstained": True, "calibration": "abstained",
                "confidence_basis": "status-certainty: certainty that no verified evidence exists",
                "route": "followup", "route_reason": detail,
                "cached_repeat": True}
    agent.trace(session_id, "chat_router", "waiting", "Spark routing intent…")
    try:
        route = await asyncio.wait_for(
            _classify(message, analysis, history), timeout=CLASSIFY_TIMEOUT)
    except Exception as e:
        route = {"route": "investigate", "reason": f"classifier error: {e}"[:200]}
    agent.trace(session_id, "chat_router", "received",
                f"route={route['route']}: {route.get('reason','')}"[:300])

    if route["route"] == "followup":
        rec = await _followup(session_id, message, analysis, history)
    elif route["route"] == "reanalyze":
        rec = await _reanalyze(session_id, message)
    else:
        rec = await investigator.answer_question(
            session_id, message, sess.get("video_url") or "")
    rec.setdefault("route", route["route"])
    rec.setdefault("route_reason", route.get("reason", ""))
    return rec
