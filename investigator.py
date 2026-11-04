"""Active-perception investigator: retrieve evidence, answer, verify, abstain.

Plan -> retrieve -> inspect -> gate:
  1. PLAN: Spark proposes a search interval/strategy (never timestamps used as
     evidence by themselves).
  2. RETRIEVE: retrieval.py ranks events/entities/speech/OCR/audio by the
     question and time hints. Spans and frame ids come from the index.
  3. INSPECT: Spark sees the retrieved evidence list + ≤6 stills and must cite
     evidence ids it used.
  4. GATE: every cited id is resolved to its stored span, frames are checked
     against the span, confidence is calibrated, low/unknown -> abstention.

No timestamp in the final evidence is authored by the model; all come from the
deterministic index or a bounded re-decode of the session's own upload.
"""
from __future__ import annotations

import io
import json
import logging
import os

import imageio.v3 as iio
import numpy as np

import evidence as evmod
import retrieval

log = logging.getLogger("delta.investigator")


def _load_video_bytes(video_url: str) -> bytes | None:
    try:
        if video_url.startswith("upload://"):
            from media import UPLOAD_DIR
            path = os.path.join(UPLOAD_DIR, os.path.basename(video_url[len("upload://"):]))
            with open(path, "rb") as f:
                return f.read()
        if os.path.exists(video_url):
            with open(video_url, "rb") as f:
                return f.read()
    except Exception:
        pass
    return None


def _graph_summary(events: list[dict], entities: list[dict]) -> str:
    lines = []
    for e in events[:20]:
        lines.append(f"{e['event_id']} [{e['start_t']:.1f}-{e['end_t']:.1f}s] "
                     f"{e.get('action','')} ent={','.join(e.get('entity_ids',[]))} "
                     f"{e.get('detail','')[:100]}")
    if entities:
        lines.append("entities: " + ", ".join(f"{x['entity_id']}={x.get('label','')}" for x in entities[:8]))
    return "\n".join(lines)[:4000] or "(empty event graph)"


def _analysis_context(sess: dict) -> str:
    """Prior Spark video understanding — the answer/analysis the pipeline already did."""
    try:
        themes = json.loads(sess.get("themes_json") or "[]")
    except Exception:
        themes = []
    try:
        moments = json.loads(sess.get("moments_json") or "[]")
    except Exception:
        moments = []
    lines = []
    if sess.get("answer"):
        lines.append(f"Direct answer already given to the viewer prompt: {sess['answer']}")
    if sess.get("summary"):
        lines.append(f"Video summary: {sess['summary']}")
    if themes:
        lines.append("Themes: " + ", ".join(str(t) for t in themes[:8]))
    for m in moments[:12]:
        lines.append(f"moment [{float(m.get('t_start', 0)):.1f}-{float(m.get('t_end', 0)):.1f}s] "
                     f"{m.get('label', '')}: {m.get('detail', '')}")
    transcript = (sess.get("transcript") or "").strip()
    if transcript:
        lines.append(f"Transcript: {transcript[:2000]}")
    return "\n".join(lines)[:6000] or "(no prior analysis)"


async def _plan(question: str, summary: str, duration: float, analysis: str = "") -> dict:
    from spark import SPARK_MODEL, extract_text, post_unary
    body = await post_unary({
        "model": SPARK_MODEL,
        "instructions": "You plan video evidence gathering. Strict JSON.",
        "input": [{"type": "message", "role": "user", "content": [{"type": "input_text", "text":
            f"Question: {question}\nDuration: {duration:.1f}s\n"
            f"Prior video analysis:\n{analysis}\nEvent graph:\n{summary}\n"
            "Return {\"targets\": [entity labels or objects to find], "
            "\"interval\": [start_s, end_s] most likely to hold the answer, "
            "\"strategy\": \"track|disappearance|transfer|placement|reading|ordering\", "
            "\"disproof\": \"what observation would disprove the claim\"}."}]}],
        "text": {"format": {"type": "json_schema", "name": "Plan",
                 "schema": {"type": "object",
                            "properties": {
                                "targets": {"type": "array", "items": {"type": "string"}},
                                "interval": {"type": "array", "items": {"type": "number"}},
                                "strategy": {"type": "string"}, "disproof": {"type": "string"}},
                            "required": ["targets", "interval", "strategy", "disproof"],
                            "additionalProperties": False}, "strict": True}},
    })
    try:
        d = json.loads(extract_text(body))
        iv = list(d.get("interval") or [0, duration])
        s = max(0.0, float(iv[0]) if len(iv) > 0 else 0.0)
        e = min(duration, float(iv[1]) if len(iv) > 1 else duration)
        if e <= s:
            s, e = 0.0, duration
        return {"targets": [str(t)[:60] for t in d.get("targets", [])][:4],
                "interval": [s, e], "strategy": str(d.get("strategy", ""))[:40],
                "disproof": str(d.get("disproof", ""))[:300]}
    except Exception:
        return {"targets": [], "interval": [0.0, duration], "strategy": "track", "disproof": ""}


def _evidence_lines(items: list[dict]) -> str:
    return "\n".join(
        f"{i['id']} | {i['kind']} | [{i['start']:.2f}-{i['end']:.2f}s] | "
        f"frames={i['frame_ids'][:4]} | {i['text'][:180]}"
        for i in items)[:6000]


async def _inspect(question: str, frame_paths: list[str], times: list[float],
                   analysis: str, items: list[dict]) -> dict:
    """Focused visual QA on ≤6 stills plus retrieved evidence; ids are mandatory."""
    import base64
    from spark import SPARK_MODEL, extract_text, post_unary
    content: list[dict] = [{"type": "input_text", "text":
        f"Question about this video: {question}\n"
        f"Prior full-video analysis:\n{analysis}\n"
        f"Retrieved evidence (id | kind | span | frames | text):\n{_evidence_lines(items)}\n"
        f"Stills are at times {['%.1f' % t for t in times]}s.\n"
        "Answer from the retrieved evidence and stills only. Return {\"answer\": str, "
        "\"evidence_ids\": [ids actually used], \"visible\": true/false, "
        "\"confidence\": 0-1, \"grounding\": \"observed|inferred|unknown\"} where observed "
        "means visible in a still, inferred means supported by an evidence item, unknown "
        "means neither. Cite at least one id for any factual answer."}]
    for p in frame_paths[:6]:
        try:
            with open(p, "rb") as f:
                content.append({"type": "input_image",
                                "image_url": f"data:image/jpeg;base64,{base64.b64encode(f.read()).decode()}"})
        except OSError:
            continue
    if len(content) == 1:
        return {"answer": "", "evidence_ids": [], "visible": False,
                "confidence": 0.0, "grounding": "unknown"}
    body = await post_unary({
        "model": SPARK_MODEL,
        "instructions": "Answer only from retrieved evidence and visible frames. Cite ids. Strict JSON.",
        "input": [{"type": "message", "role": "user", "content": content}],
        "text": {"format": {"type": "json_schema", "name": "Inspect",
                 "schema": {"type": "object",
                            "properties": {"answer": {"type": "string"},
                                           "evidence_ids": {"type": "array", "items": {"type": "string"}},
                                           "visible": {"type": "boolean"},
                                           "confidence": {"type": "number"},
                                           "grounding": {"type": "string"}},
                            "required": ["answer", "evidence_ids", "visible",
                                         "confidence", "grounding"],
                            "additionalProperties": False}, "strict": True}},
    })
    try:
        d = json.loads(extract_text(body))
        return {"answer": str(d.get("answer", ""))[:1000],
                "evidence_ids": [str(x)[:80] for x in (d.get("evidence_ids") or [])][:8],
                "visible": bool(d.get("visible", False)),
                "confidence": max(0.0, min(1.0, float(d.get("confidence", 0.0)))),
                "grounding": str(d.get("grounding", "unknown"))}
    except Exception:
        return {"answer": "", "evidence_ids": [], "visible": False,
                "confidence": 0.0, "grounding": "unknown"}


def _densify(video_bytes: bytes | None, session_id: str, start: float, end: float,
             fps: float, n: int = 6) -> tuple[list[str], list[float]]:
    """Decode up to n evenly spaced frames in [start,end] for close inspection."""
    if not video_bytes or end <= start:
        return [], []
    from media import frame_to_jpeg
    try:
        from media import probe
        meta = probe(video_bytes)
        base_fps = meta.get("fps") or fps or 30.0
    except Exception:
        base_fps = fps or 30.0
    times = [start + (end - start) * k / max(1, n - 1) for k in range(n)]
    sdir = os.path.join(os.path.dirname(__file__), "frames", session_id)
    os.makedirs(sdir, exist_ok=True)
    paths: list[str] = []
    for t in times:
        fi = int(round(t * base_fps))
        try:
            frame = np.asarray(iio.imread(io.BytesIO(video_bytes), index=fi, extension=".mp4"))
        except Exception:
            continue
        p = os.path.join(sdir, f"qa_{fi:06d}.jpg")
        with open(p, "wb") as f:
            f.write(frame_to_jpeg(frame))
        paths.append(p)
    return paths, times[:len(paths)]


async def answer_question(session_id: str, question: str, video_url: str = "") -> dict:
    """Run plan-retrieve-inspect-gate; persist + return the QA record."""
    import uuid
    import store
    events = store.get_events(session_id)
    entities, observations = store.get_entities(session_id)
    frames = store.get_frames(session_id)
    turns = []
    sess = store.get_session(session_id) or {}
    try:
        turns = json.loads(sess.get("turns_json") or "[]")
    except Exception:
        turns = []
    ocr = store.get_ocr(session_id)
    audio = store.get_audio_events(session_id)
    try:
        duration = float((json.loads(sess.get("video_json") or "{}")).get("duration_s", 0))
    except Exception:
        duration = 0.0
    if duration <= 0 and events:
        duration = max(float(e.get("end_t", 0)) for e in events)
    known_frames = {int(r["frame_index"]) for r in frames}
    frame_times = {int(r["frame_index"]): float(r["t"]) for r in frames}
    by_frame = {int(r["frame_index"]): r for r in frames}
    analysis = _analysis_context(sess)

    plan = await _plan(question, _graph_summary(events, entities), duration or 0.0, analysis)
    items = retrieval.retrieve(session_id, question, events, entities, observations,
                               turns, ocr, audio, frames, duration or 0.0)
    if not items:
        qa_id = f"qa_{uuid.uuid4().hex[:8]}"
        rec = evmod.abstain(question, "no retrievable evidence")
        rec["trace"] = {"plan": plan, "retrieved": []}
        store.save_qa(session_id, qa_id, question, rec["answer"], [],
                      rec["confidence"], "unknown", rec["raw_confidence"])
        return {"qa_id": qa_id, "question": question, **rec}

    s, e = plan["interval"]
    fids = [f for it in items[:6] for f in it["frame_ids"]]
    paths = [by_frame[f]["path"] for f in fids if f in by_frame
             and os.path.exists(by_frame[f].get("path", ""))][:6]
    times = [float(by_frame[f]["t"]) for f in fids if f in by_frame][:6]
    if len(paths) < 3 and video_url:
        vb = _load_video_bytes(video_url)
        dp, dt = _densify(vb, session_id, s, e, 30.0)
        if len(dp) > len(paths):
            paths, times = dp, dt

    insp = await _inspect(question, paths, times, analysis, items)
    by_id = {it["id"]: it for it in items}
    used = [by_id[i] for i in insp.get("evidence_ids", []) if i in by_id]
    spans = [{"start": round(float(it["start"]), 3), "end": round(float(it["end"]), 3),
              "frame_ids": [int(f) for f in it["frame_ids"]][:6],
              "source_id": it["id"], "kind": it["kind"],
              "verification": "supported" if it["frame_ids"] else "inferred"}
             for it in used]
    ok, reason = evmod.verify(spans, duration, known_frames, frame_times)
    support = (len(used) / max(1, len(insp.get("evidence_ids", [])))) if used else 0.0
    n_frames = sum(len(sp["frame_ids"]) for sp in spans)
    detail = evmod.calibrate_detail(insp.get("confidence", 0.0), len(spans),
                                    n_frames, support)
    verification = "supported" if (ok and spans and insp.get("grounding") != "unknown"
                                   and any(sp["verification"] == "supported" for sp in spans)) else (
        "inferred" if (ok and spans and insp.get("grounding") != "unknown") else "unknown")
    gate = evmod.decision(verification, detail["calibrated"],
                          bool(insp.get("answer")) and bool(spans))
    qa_id = f"qa_{uuid.uuid4().hex[:8]}"
    trace = {"plan": plan, "retrieved": [{k: it[k] for k in ("id", "kind", "start", "end", "score")}
                                         for it in items],
             "evidence_ids": insp.get("evidence_ids", []),
             "verify_reason": reason, "calibration": detail}
    if gate["abstain"]:
        rec = evmod.abstain(question, gate["reason"])
        rec["trace"] = trace
        store.save_qa(session_id, qa_id, question, rec["answer"], [],
                      rec["confidence"], "unknown", rec["raw_confidence"])
    else:
        rec = {"answer": insp["answer"], "evidence": spans,
               "confidence": detail["calibrated"], "raw_confidence": detail["raw"],
               "verification": verification, "abstained": False,
               "calibration": detail["method"], "trace": trace}
        store.save_qa(session_id, qa_id, question, rec["answer"], rec["evidence"],
                      rec["confidence"], rec["verification"], rec.get("raw_confidence", 0.0))
    return {"qa_id": qa_id, "question": question, **rec}
