"""FastAPI orchestration layer (spec section 2).

BackgroundTasks stand in for Celery+Redis in Phase 2 (single worker, in-memory
store). Swap `save_report`/`run_pipeline_async` internals when Redis lands.
"""
from __future__ import annotations

from fastapi import BackgroundTasks, FastAPI, HTTPException

from orchestrator import get_report, run_pipeline_async, save_report
from schemas import CoachSessionInput, SessionReport

app = FastAPI(title="delta — Coaching Analytics API")


@app.get("/health")
def health() -> dict:
    return {"ok": True}


@app.post("/api/v1/sessions", status_code=202)
async def create_session(inp: CoachSessionInput, bg: BackgroundTasks) -> dict:
    import uuid

    session_id = f"ses_{uuid.uuid4().hex[:8]}"
    placeholder = SessionReport(session_id=session_id, status="processing")
    save_report(placeholder)

    async def _run() -> None:
        report = await run_pipeline_async(inp)
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
