"""Full Video Intelligence Engine pipeline (real-only, no mocks).

Stages: queued -> uploading -> indexing -> transcribing -> observing ->
tracking -> ocr -> events -> understanding -> verifying -> completed/failed.

Perception sub-steps (SAM tracking, OCR, event proposal) degrade to honest
empty on error and lower the ablation level — they never invent data.
Only the core (upload/transcribe/understand) can fail the session; missing
key or core API error marks failed with the real error text.
"""
from __future__ import annotations

import json
import logging
import os
import time
import uuid

from schemas import (EvidenceSpan, SessionReport, TranscriptTurn, VideoMeta,
                     VideoMoment, VideoSessionInput)

log = logging.getLogger("delta.orchestrator")

# Back-compat shims: old app.py imported these from here.
_store: dict[str, SessionReport] = {}
_progress: dict[str, dict] = {}
_lanes: dict[str, dict] = {}


def save_progress(session_id: str, stage: str, detail: str = "", error: str | None = None) -> None:
    import store as _store_mod
    _store_mod.save_progress(session_id, stage, detail, error)
    _progress[session_id] = {"session_id": session_id, "stage": stage, "detail": detail,
                             "error": error, "updated_at": time.time()}


def get_progress(session_id: str) -> dict | None:
    import store as _store_mod
    return _store_mod.get_progress(session_id) or _progress.get(session_id)


def save_report(report: SessionReport) -> None:
    _store[report.session_id] = report


def get_report(session_id: str) -> SessionReport | None:
    if session_id in _store:
        return _store[session_id]
    import store as _store_mod
    row = _store_mod.get_session(session_id)
    if not row:
        return None
    return _row_to_report(row)


def save_lanes(session_id: str, payload: dict) -> None:
    _lanes[session_id] = payload


def get_lanes(session_id: str) -> dict | None:
    return _lanes.get(session_id, {"lanes": {}})


def _row_to_report(row: dict) -> SessionReport:
    try:
        video = json.loads(row.get("video_json") or "{}")
    except Exception:
        video = {}
    try:
        turns = json.loads(row.get("turns_json") or "[]")
    except Exception:
        turns = []
    try:
        moments = json.loads(row.get("moments_json") or "[]")
    except Exception:
        moments = []
    try:
        themes = json.loads(row.get("themes_json") or "[]")
    except Exception:
        themes = []
    try:
        answer_evidence = [EvidenceSpan(**s) for s in
                           json.loads(row.get("answer_evidence_json") or "[]")]
    except Exception:
        answer_evidence = []
    return SessionReport(
        session_id=row["session_id"], status=row.get("status") or "completed",
        title=row.get("title") or "", answer=row.get("answer") or "",
        answer_evidence=answer_evidence,
        answer_confidence=float(row.get("answer_confidence") or 0.0),
        answer_verification=row.get("answer_verification") or "unknown",
        answer_calibration=row.get("answer_calibration") or "uncalibrated",
        summary=row.get("summary") or "",
        themes=themes, transcript=row.get("transcript") or "",
        turns=[TranscriptTurn(**{k: t.get(k, 0 if k in ("turnId", "startMs", "endMs") else "")
                                 for k in ("turnId", "startMs", "endMs", "transcript", "speaker")})
               for t in turns],
        moments=[VideoMoment(**m) for m in moments],
        on_screen_text=row.get("on_screen_text") or "",
        has_audio=bool(row.get("has_audio")),
        video=VideoMeta(width=video.get("width", 0), height=video.get("height", 0),
                        fps=video.get("fps") or 30.0, duration_s=video.get("duration_s") or 0.0),
        av_offset_s=float(row.get("av_offset_s") or 0.0),
        av_sync_confidence=float(row.get("av_sync_confidence") or 0.0),
    )


async def run_pipeline_async(inp: VideoSessionInput, progress_cb=None,
                             session_id: str | None = None) -> SessionReport:
    """Agent pipeline: upload -> prepare (wav+frames once) -> 4 perception tools
    concurrently (transcribe | SAM track | OCR | audio events) -> event graph ->
    Spark understand -> evidence verify. No tool can stall the run: every tool
    has a budget and degrades to honest empty; only upload/probe failure fails."""
    import asyncio
    import store
    import agent
    from media import probe
    from sam_client import NoApiKey, api_key, local_video_bytes, resolve_video, upload_file

    session_id = session_id or f"ses_{uuid.uuid4().hex[:8]}"
    save_progress(session_id, "queued", "Queued — pipeline starting…")
    store.clear_trace(session_id)
    save_lanes(session_id, {"lanes": {}})
    try:
        api_key()
    except NoApiKey as e:
        save_progress(session_id, "failed", "Missing API key.", error=str(e))
        rep = SessionReport(session_id=session_id, status="failed", title=inp.title)
        save_report(rep)
        store.upsert_session(session_id, status="failed", title=inp.title,
                             focus=inp.focus, video_url=inp.video_url, error=str(e))
        return rep

    try:
        save_progress(session_id, "uploading", "Uploading clip to perception API…")
        store.append_trace(session_id, "upload", "sent", f"resolving {inp.video_url[:80]}")
        file_id, upload_task = None, None
        video_bytes = await asyncio.to_thread(local_video_bytes, inp.video_url)
        if video_bytes is not None:
            # Local bytes: upload while frames are being decoded below.
            upload_task = asyncio.create_task(upload_file(
                video_bytes, os.path.basename(inp.video_url.rstrip("/")) or "clip.mp4"))
            store.append_trace(session_id, "upload", "waiting",
                               "Files API upload (concurrent with local prepare)…")
        else:
            store.append_trace(session_id, "upload", "waiting", "Files API upload…")
            file_id, video_bytes = await asyncio.wait_for(
                resolve_video(inp.video_url), timeout=agent.UPLOAD_TIMEOUT)
            store.append_trace(session_id, "upload", "received",
                               f"file_id={file_id[:16]}… ({len(video_bytes or b'') / 1e6:.1f}MB)" if video_bytes
                               else f"file_id={file_id[:16]}… (remote bytes)")
        meta = probe(video_bytes) if video_bytes else {}
        fps = meta.get("fps") or inp.fps or 30.0
        duration_s = meta.get("duration_s") or 0.0
        W, H = meta.get("width", 0), meta.get("height", 0)
        n_frames = meta.get("n_frames") or int(duration_s * fps)
        store.upsert_session(session_id, status="processing", title=inp.title,
                             focus=inp.focus, video_url=inp.video_url,
                             video_json=json.dumps({"width": W, "height": H, "fps": fps,
                                                    "duration_s": duration_s}))
        store.append_trace(session_id, "prepare", "received",
                           f"{W}x{H} @ {fps:.1f}fps, {duration_s:.1f}s, focus: {inp.focus[:80]}")

        # ---- prepare: extract WAV once + observe frames (both local) ----
        save_progress(session_id, "preparing", "Cutting audio + indexing frames…")
        from transcribe import extract_wav
        wav_bytes = await asyncio.to_thread(extract_wav, video_bytes) if video_bytes else None
        store.append_trace(session_id, "prepare", "sent",
                           f"audio extract -> {'%.1fMB WAV' % (len(wav_bytes) / 1e6) if wav_bytes else 'no audio track'}")
        # These don't need frames: run them while frames decode.
        tr_task = asyncio.create_task(agent._transcribe_tool(session_id, wav_bytes))
        au_task = asyncio.create_task(agent._audio_ev_tool(session_id, wav_bytes))
        intel_task = asyncio.create_task(agent._understand_tool(
            session_id, file_id, inp.focus, tr_task, duration_s, upload_task=upload_task))
        save_progress(session_id, "observing", "Adaptive visual observation…")
        import observer
        try:
            _plan, frame_rows = await asyncio.to_thread(
                observer.observe, video_bytes, session_id, fps, duration_s)
            store.save_frames(session_id, frame_rows)
        except Exception as e:
            log.warning("observation failed: %s", e)
            store.append_trace(session_id, "observe_frames", "failed", str(e)[:200])
            frame_rows = []
        store.append_trace(session_id, "observe_frames", "received",
                           f"{len(frame_rows)} frames indexed")
        if progress_cb:
            progress_cb({"stage": "observed", "frames": len(frame_rows)})

        # ---- perceiving: 4 tools concurrently (transcribe | SAM | OCR | audio) ----
        if upload_task is not None:
            file_id = await asyncio.wait_for(upload_task, timeout=agent.UPLOAD_TIMEOUT)
            store.append_trace(session_id, "upload", "received",
                               f"file_id={file_id[:16]}… (uploaded during prepare)")
        save_progress(session_id, "perceiving",
                      "Transcribe + SAM tracking + OCR + audio events running concurrently…")
        perc = await agent.run_perception(
            session_id, file_id, frame_rows, wav_bytes, inp.focus, fps, n_frames, (W, H),
            duration_s=duration_s, transcript_task=tr_task, audio_task=au_task,
            understanding_task=intel_task)
        if progress_cb:
            progress_cb({"stage": "perceived", "entities": len(perc["entities"]),
                         "turns": len(perc["transcript"].get("turns", []))})

        # ---- measured A/V alignment: shift audio rows onto the video clock ----
        # Apply the measured lag only when the measurement itself is valid
        # (both signals non-flat); validity comes from av_align's method flag,
        # not from a confidence cutoff.
        import av_align
        av = await asyncio.to_thread(av_align.estimate_offset, wav_bytes, frame_rows, duration_s)
        measurable = av.get("method") not in (
            "insufficient-signal", "insufficient-audio", "flat-signal")
        offset = float(av.get("offset_s", 0.0)) if measurable else 0.0
        if offset and perc["audio"]:
            import audio_events
            perc["audio"] = audio_events.shift_events(perc["audio"], offset, duration_s)
            store.save_audio_events(session_id, perc["audio"])
        if offset and perc["transcript"].get("turns"):
            for t in perc["transcript"]["turns"]:
                t["startMs"] = max(0.0, float(t.get("startMs", 0)) - offset * 1000)
                t["endMs"] = max(float(t["startMs"]), float(t.get("endMs", 0)) - offset * 1000)
        store.upsert_session(session_id, av_offset_s=float(av.get("offset_s", 0.0)),
                             av_sync_confidence=float(av.get("confidence", 0.0)))
        store.append_trace(session_id, "av_align", "received",
                           f"offset {av.get('offset_s', 0):+.2f}s "
                           f"(confidence {av.get('confidence', 0):.2f}, {av.get('method', '')})")

        # ---- events -> understanding -> verifying (need all perception outputs) ----
        save_progress(session_id, "events", "Building event graph…")
        fused = await agent.run_fusion_understanding(
            session_id, file_id, inp.focus, perc, frame_rows, duration_s)
        graph_events, intel, verified_moments = (
            fused["graph_events"], fused["intel"], fused["verified"])
        tr = perc["transcript"]

        save_progress(session_id, "verifying", "Evidence verification…")

        level = 1 + (1 if frame_rows else 0) + (1 if (perc["entities"] or graph_events) else 0)
        report = SessionReport(
            session_id=session_id, status="completed", title=inp.title,
            answer=intel["answer"],
            answer_evidence=[EvidenceSpan(**s) for s in fused.get("answer_evidence", [])],
            answer_confidence=float(fused.get("answer_confidence", 0.0)),
            answer_verification=str(fused.get("answer_verification", "unknown")),
            answer_calibration=str(fused.get("answer_calibration", "uncalibrated")),
            summary=intel["summary"], themes=intel["themes"],
            transcript=tr["transcript"],
            turns=[TranscriptTurn(**{k: t.get(k, 0 if k in ("turnId", "startMs", "endMs") else "")
                                     for k in ("turnId", "startMs", "endMs", "transcript", "speaker")})
                   for t in tr["turns"]],
            moments=[VideoMoment(**m) for m in verified_moments],
            on_screen_text=intel["on_screen_text"],
            has_audio=bool(tr.get("has_audio")),
            video=VideoMeta(width=W, height=H, fps=fps, duration_s=duration_s),
            n_frames_indexed=len(frame_rows), n_entities=len(perc["entities"]),
            n_events=len(graph_events), ablation_level=level,
            av_offset_s=float(av.get("offset_s", 0.0)),
            av_sync_confidence=float(av.get("confidence", 0.0)),
        )
        save_report(report)
        store.upsert_session(
            session_id, status="completed", title=inp.title, focus=inp.focus,
            video_url=inp.video_url, answer=intel["answer"],
            answer_evidence_json=json.dumps(fused.get("answer_evidence", [])),
            answer_confidence=float(fused.get("answer_confidence", 0.0)),
            answer_verification=str(fused.get("answer_verification", "unknown")),
            answer_calibration=str(fused.get("answer_calibration", "uncalibrated")),
            summary=intel["summary"],
            themes_json=json.dumps(intel["themes"]), transcript=tr["transcript"],
            turns_json=json.dumps(tr["turns"]), moments_json=json.dumps(verified_moments),
            on_screen_text=intel["on_screen_text"], has_audio=int(tr.get("has_audio", False)),
            video_json=json.dumps({"width": W, "height": H, "fps": fps, "duration_s": duration_s}),
            av_offset_s=float(av.get("offset_s", 0.0)),
            av_sync_confidence=float(av.get("confidence", 0.0)))
        save_progress(session_id, "completed",
                      f"Done — {len(verified_moments)} moments, {len(graph_events)} events, "
                      f"{len(perc['entities'])} entities, {len(frame_rows)} frames.")
        return report
    except Exception as e:
        log.exception("pipeline failed for %s", session_id)
        save_progress(session_id, "failed", "Analysis failed.", error=f"{type(e).__name__}: {e}"[:500])
        rep = SessionReport(session_id=session_id, status="failed", title=inp.title)
        save_report(rep)
        try:
            import store as _s
            _s.upsert_session(session_id, status="failed", title=inp.title,
                              focus=inp.focus, video_url=inp.video_url, error=str(e)[:500])
        except Exception:
            pass
        return rep
