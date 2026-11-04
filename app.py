"""FastAPI orchestration layer (spec section 2).

BackgroundTasks stand in for Celery+Redis in Phase 2 (single worker, in-memory
store). Swap `save_report`/`run_pipeline_async` internals when Redis lands.
"""
from __future__ import annotations

import asyncio
import json

from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from orchestrator import (
    get_lanes,
    get_report,
    run_pipeline_async,
    save_report,
)
from schemas import CoachSessionInput, SessionReport

app = FastAPI(title="delta — Coaching Analytics API")
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
    """Validate a phone clip and store it locally. Returns an upload:// handle."""
    from media import save_upload, validate_upload

    data = await file.read()
    try:
        meta = validate_upload(data, file.filename or "clip.mp4")
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    path = save_upload(data, file.filename or "clip.mp4")
    import os

    handle = f"upload://{os.path.basename(path)}"
    return {"upload_id": handle, "video_url": handle, "meta": meta}


@app.post("/api/v1/sessions", status_code=202)
async def create_session(inp: CoachSessionInput, bg: BackgroundTasks) -> dict:
    import uuid

    session_id = f"ses_{uuid.uuid4().hex[:8]}"
    placeholder = SessionReport(session_id=session_id, status="processing")
    save_report(placeholder)

    async def _run() -> None:
        try:
            report = await run_pipeline_async(inp, session_id=session_id)
        except Exception:
            report = SessionReport(session_id=session_id, status="failed")
        report.session_id = session_id  # keep the ID returned at POST time
        save_report(report)

    bg.add_task(_run)
    return {"session_id": session_id, "status": "processing"}


@app.get("/api/v1/sessions/{session_id}")
def read_session(session_id: str) -> SessionReport:
    report = get_report(session_id)
    if report is None:
        raise HTTPException(status_code=404, detail="unknown session_id")
    return report


@app.get("/api/v1/sessions/{session_id}/segmentation")
def read_segmentation(session_id: str) -> dict:
    """Raw SAM output-text lanes per role for SDK replay in the browser."""
    lanes = get_lanes(session_id)
    if not lanes or not lanes.get("lanes"):
        raise HTTPException(status_code=404, detail="no segmentation yet")
    return lanes


@app.get("/api/v1/sessions/{session_id}/stream")
async def stream_session(session_id: str):
    """SSE: live status + per-role lane growth, then done. Polls the store."""
    if get_report(session_id) is None:
        raise HTTPException(status_code=404, detail="unknown session_id")

    async def gen():
        for _ in range(1200):  # ~10 min cap
            report = get_report(session_id)
            lanes = get_lanes(session_id) or {"lanes": {}}
            yield "data: " + json.dumps({
                "status": report.status if report else "processing",
                "roles": sorted(lanes.get("lanes", {})),
                "chars": {k: len(v) for k, v in lanes.get("lanes", {}).items()},
            }) + "\n\n"
            if report and report.status in ("completed", "failed"):
                yield "data: " + json.dumps({"status": report.status, "done": True}) + "\n\n"
                return
            await asyncio.sleep(0.5)
        yield "data: " + json.dumps({"status": "timeout", "done": True}) + "\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream")
