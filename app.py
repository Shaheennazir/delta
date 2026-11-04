"""FastAPI: Video Intelligence Engine (prod).

Ingestion pipeline runs in BackgroundTasks (single worker); all state in
sqlite via store.py so restarts keep sessions. New endpoints expose the
engine: event graph, entity registry, frame store, OCR, audio events,
agentic QA (investigator), adversarial benchmark.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import uuid

from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse

import agent
import benchmark
import chat as chat_mod
import confidence
import events as ev_graph
import investigator
import store
import temporal
from media import UPLOAD_DIR, save_upload, validate_upload
from orchestrator import (
    get_progress,
    get_report,
    run_pipeline_async,
    save_progress,
    save_report,
)
from schemas import QAOutcome, QARequest, SessionReport, VideoSessionInput

log = logging.getLogger("delta.app")

app = FastAPI(title="delta — Video Intelligence Engine")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


@app.get("/health")
def health() -> dict:
    return {"ok": True}


@app.post("/api/v1/uploads")
async def upload_clip(file: UploadFile = File(...)) -> dict:
    data = await file.read()
    try:
        meta = validate_upload(data, file.filename or "clip.mp4")
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    path = save_upload(data, file.filename or "clip.mp4")
    handle = f"upload://{os.path.basename(path)}"
    return {"upload_id": handle, "video_url": handle, "meta": meta}


@app.post("/api/v1/sessions", status_code=202)
async def create_session(inp: VideoSessionInput, bg: BackgroundTasks) -> dict:
    session_id = f"ses_{uuid.uuid4().hex[:8]}"
    placeholder = SessionReport(session_id=session_id, status="processing")
    save_report(placeholder)
    save_progress(session_id, "queued", "Queued — pipeline starting…")
    store.upsert_session(session_id, status="processing", title=inp.title,
                         focus=inp.focus, video_url=inp.video_url)

    async def _run() -> None:
        try:
            report = await run_pipeline_async(inp, session_id=session_id)
        except Exception as e:  # surface reason instead of silent empty report
            log.exception("pipeline failed for %s", session_id)
            save_progress(session_id, "failed", "Analysis failed.",
                          error=f"{type(e).__name__}: {e}"[:500])
            report = SessionReport(session_id=session_id, status="failed")
        report.session_id = session_id
        save_report(report)

    bg.add_task(_run)
    return {"session_id": session_id, "status": "processing"}


def _require_report(session_id: str) -> SessionReport:
    report = get_report(session_id)
    if report is None:
        raise HTTPException(status_code=404, detail="unknown session_id")
    return report


@app.get("/api/v1/sessions")
def list_sessions() -> dict:
    return {"sessions": store.list_sessions()}


@app.get("/api/v1/uploads/{name}")
def serve_upload(name: str):
    base = os.path.abspath(UPLOAD_DIR)
    path = os.path.normpath(os.path.join(base, os.path.basename(name)))
    if not path.startswith(base) or not os.path.exists(path):
        raise HTTPException(status_code=404, detail="unknown upload")
    return FileResponse(path, media_type="video/mp4")


@app.get("/api/v1/sessions/{session_id}")
def read_session(session_id: str) -> SessionReport:
    return _require_report(session_id)


@app.get("/api/v1/sessions/{session_id}/segmentation")
def read_segmentation(session_id: str) -> dict:
    if get_report(session_id) is None:
        raise HTTPException(status_code=404, detail="unknown session_id")
    frames = store.get_frames(session_id)
    ents, obs = store.get_entities(session_id)
    lanes: dict[str, list] = {}
    for e in ents:
        lanes[e["entity_id"]] = [o for o in obs if o["entity_id"] == e["entity_id"]]
    return {"width": 0, "height": 0, "fps": 30.0,
            "n_frames": len(frames), "lanes": lanes}


@app.get("/api/v1/sessions/{session_id}/trace")
def read_trace(session_id: str) -> dict:
    if get_report(session_id) is None:
        raise HTTPException(status_code=404, detail="unknown session_id")
    return {"session_id": session_id, "trace": store.get_trace(session_id)}


@app.get("/api/v1/sessions/{session_id}/tools")
def read_tools(session_id: str) -> dict:
    if get_report(session_id) is None:
        raise HTTPException(status_code=404, detail="unknown session_id")
    return {"session_id": session_id, "tools": agent.TOOLS}


@app.get("/api/v1/sessions/{session_id}/progress")
def read_progress(session_id: str) -> dict:
    prog = get_progress(session_id)
    if prog is None:
        raise HTTPException(status_code=404, detail="unknown session_id")
    return prog


@app.get("/api/v1/sessions/{session_id}/stream")
async def stream_session(session_id: str):
    if get_report(session_id) is None:
        raise HTTPException(status_code=404, detail="unknown session_id")

    async def gen():
        for _ in range(3600):  # ~30 min cap for full engine runs
            report = get_report(session_id)
            prog = get_progress(session_id) or {}
            try:
                trace = store.get_trace(session_id)[-20:]
            except Exception:
                trace = []
            yield "data: " + json.dumps({
                "status": report.status if report else "processing",
                "stage": prog.get("stage", "processing"),
                "detail": prog.get("detail", ""),
                "error": prog.get("error"),
                "trace": trace,
            }) + "\n\n"
            if report and report.status in ("completed", "failed"):
                yield "data: " + json.dumps({"status": report.status,
                                             "stage": prog.get("stage", report.status),
                                             "detail": prog.get("detail", ""),
                                             "error": prog.get("error"),
                                             "done": True}) + "\n\n"
                return
            await asyncio.sleep(0.5)
        yield "data: " + json.dumps({"status": "timeout", "done": True}) + "\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream")


# ---- engine endpoints ----

@app.get("/api/v1/sessions/{session_id}/events")
def read_events(session_id: str) -> dict:
    _require_report(session_id)
    return {"session_id": session_id, "events": store.get_events(session_id)}


@app.get("/api/v1/sessions/{session_id}/entities")
def read_entities(session_id: str) -> dict:
    _require_report(session_id)
    ents, obs = store.get_entities(session_id)
    return {"session_id": session_id, "entities": ents, "observations": obs}


@app.get("/api/v1/sessions/{session_id}/frames")
def read_frames(session_id: str) -> dict:
    _require_report(session_id)
    return {"session_id": session_id, "frames": store.get_frames(session_id)}


@app.get("/api/v1/sessions/{session_id}/ocr")
def read_ocr(session_id: str) -> dict:
    _require_report(session_id)
    return {"session_id": session_id, "ocr": store.get_ocr(session_id)}


@app.get("/api/v1/sessions/{session_id}/audio-events")
def read_audio_events(session_id: str) -> dict:
    _require_report(session_id)
    return {"session_id": session_id, "audio_events": store.get_audio_events(session_id)}


@app.get("/api/v1/sessions/{session_id}/qa")
def list_qa(session_id: str) -> dict:
    _require_report(session_id)
    return {"session_id": session_id, "qa": store.get_qa(session_id)}


@app.get("/api/v1/sessions/{session_id}/state")
def read_state(session_id: str) -> dict:
    _require_report(session_id)
    events = store.get_events(session_id)
    return {"session_id": session_id,
            "states": ev_graph.state_timeline(events),
            "contradictions": temporal.contradictions(events)}


@app.get("/api/v1/sessions/{session_id}/alignment")
def read_alignment(session_id: str) -> dict:
    _require_report(session_id)
    sess = store.get_session(session_id) or {}
    return {"session_id": session_id,
            "offset_s": float(sess.get("av_offset_s") or 0.0),
            "confidence": float(sess.get("av_sync_confidence") or 0.0),
            "method": "rms-vs-motion-xcorr"}


@app.get("/api/v1/calibration")
def read_calibration() -> dict:
    rows = store.get_outcomes()
    pairs = [(float(r["raw_confidence"]), int(r["correct"])) for r in rows]
    return {"points": store.get_calibration(), "n_outcomes": len(pairs),
            "brier": confidence.brier(pairs),
            "ece": confidence.expected_calibration_error(pairs),
            "selective": confidence.selective_threshold(pairs) if pairs else None}


@app.post("/api/v1/sessions/{session_id}/qa/{qa_id}/outcome")
def record_qa_outcome(session_id: str, qa_id: str, body: QAOutcome) -> dict:
    _require_report(session_id)
    row = next((q for q in store.get_qa(session_id) if q["qa_id"] == qa_id), None)
    if row is None:
        raise HTTPException(status_code=404, detail="unknown qa_id")
    store.save_outcome(session_id, qa_id, float(row.get("raw_confidence") or row.get("confidence") or 0),
                       body.correct)
    return {"session_id": session_id, "qa_id": qa_id, **confidence.fit_from_outcomes()}


@app.post("/api/v1/sessions/{session_id}/outcome")
def record_answer_outcome(session_id: str, body: QAOutcome) -> dict:
    report = _require_report(session_id)
    store.save_outcome(session_id, "answer", float(report.answer_confidence), body.correct)
    return {"session_id": session_id, **confidence.fit_from_outcomes()}


@app.post("/api/v1/sessions/{session_id}/chat")
async def chat_message(session_id: str, body: QARequest) -> dict:
    report = _require_report(session_id)
    if report.status != "completed":
        raise HTTPException(status_code=409, detail="session not completed yet")
    return await chat_mod.answer_chat(session_id, body.question)


@app.post("/api/v1/sessions/{session_id}/qa")
async def ask_question(session_id: str, body: QARequest) -> dict:
    report = _require_report(session_id)
    if report.status != "completed":
        raise HTTPException(status_code=409, detail="session not completed yet")
    sess = store.get_session(session_id) or {}
    rec = await investigator.answer_question(session_id, body.question, sess.get("video_url") or "")
    return rec


@app.post("/api/v1/sessions/{session_id}/benchmark")
def run_benchmark(session_id: str) -> dict:
    _require_report(session_id)
    return benchmark.run_all(session_id)


@app.get("/api/v1/frames/{session_id}/{name}")
def serve_frame(session_id: str, name: str):
    base = os.path.join(os.path.dirname(__file__), "frames", session_id)
    path = os.path.normpath(os.path.join(base, name))
    if not path.startswith(base) or not os.path.exists(path):
        raise HTTPException(status_code=404, detail="unknown frame")
    return FileResponse(path, media_type="image/jpeg")
