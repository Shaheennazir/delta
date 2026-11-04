"""Muse Spark contributor agent: question in, tools out, verified answer back.

The user asks what they want from the video (``focus``). The agent owns a
tool registry and runs perception tools *concurrently* — full-audio
transcribe, SAM segmentation/tracking, OCR, audio events — then fuses the
results into an event graph, understands with Spark, verifies evidence, and
answers. Every tool call logs sent / waiting / received / failed to the
trace store so the UI shows exactly what is going on.
"""
from __future__ import annotations

import asyncio
import json
import logging
import time

log = logging.getLogger("delta.agent")

# Per-tool wall-clock budgets. A tool that exceeds its budget degrades to
# honest empty (transcribe -> empty transcript) instead of stalling the run.
TRANSCRIBE_TIMEOUT = 240.0
TRACK_TIMEOUT = 240.0
OCR_TIMEOUT_TOTAL = 150.0
EVENT_GRAPH_TIMEOUT = 120.0
UNDERSTAND_TIMEOUT = 180.0
UPLOAD_TIMEOUT = 180.0

TOOLS = {
    "transcribe_full_audio": "Cut audio from the video, send the FULL track to "
                             "muse-voice-transcribe in one POST, return transcript+turns.",
    "segment_track_sam": "Propose trackable concepts from keyframes, run one SAM 3.1 "
                          "stream per concept concurrently, return entity registry.",
    "ocr_frames": "Still-image OCR on the sharpest keyframes (concurrent passes + "
                  "multi-frame voting), return on-screen texts.",
    "detect_audio_events": "Deterministic non-speech audio events (impacts, swells, "
                            "silence) from the extracted WAV.",
    "extract_event_graph": "Fuse entities+speech+OCR+audio into a timestamped event graph.",
    "understand_verify": "Spark summary/moments/themes from video+transcript, "
                         "evidence-gated before returning.",
}


def trace(session_id: str, tool: str, event: str, detail: str = "") -> dict:
    """Log sent|waiting|received|failed and mirror the latest line to progress."""
    import store
    row = store.append_trace(session_id, tool, event, detail)
    try:
        cur = store.get_progress(session_id) or {}
        waiting = len([t for t in store.get_trace(session_id) if t["event"] == "waiting"])
        store.save_progress(session_id, cur.get("stage", "perceiving"),
                            f"{tool}: {event} {detail}".strip()[:300] +
                            (f" · {waiting} tool(s) still waiting" if waiting else ""))
    except Exception:
        pass
    return row


async def _transcribe_tool(session_id: str, wav_bytes: bytes | None) -> dict:
    from transcribe import transcribe_video
    if not wav_bytes:
        trace(session_id, "transcribe_full_audio", "received", "no audio track — empty transcript")
        return {"transcript": "", "turns": [], "audioDurationMs": 0, "has_audio": False}
    trace(session_id, "transcribe_full_audio", "sent",
          f"full audio -> muse-voice-transcribe ({len(wav_bytes) / 1e6:.1f}MB WAV, one POST)")
    trace(session_id, "transcribe_full_audio", "waiting", "voice model working…")
    t0 = time.time()
    try:
        out = await asyncio.wait_for(
            transcribe_video(b"", wav_bytes=wav_bytes), timeout=TRANSCRIBE_TIMEOUT)
        trace(session_id, "transcribe_full_audio", "received",
              f"transcript {len(out.get('transcript', ''))} chars, "
              f"{len(out.get('turns', []))} turns in {time.time() - t0:.0f}s")
        return out
    except (asyncio.TimeoutError, TimeoutError) as e:
        trace(session_id, "transcribe_full_audio", "failed", f"timeout ({e}) — continuing without speech")
        return {"transcript": "", "turns": [], "audioDurationMs": 0, "has_audio": False}
    except Exception as e:
        trace(session_id, "transcribe_full_audio", "failed", f"{type(e).__name__}: {e}"[:300])
        raise


async def _track_tool(session_id: str, file_id: str, frame_rows: list[dict],
                      focus: str, fps: float, n_frames: int, media_wh: tuple) -> tuple[list, list]:
    import store
    import entities as ent_mod
    trace(session_id, "segment_track_sam", "sent", "keyframes -> Spark concept proposal")
    try:
        keys = [r for r in frame_rows if r.get("keyframe")] or frame_rows
        mid = frame_rows[len(frame_rows) // 2] if frame_rows else None
        key_paths = [r["path"] for r in (keys[:2] + ([mid] if mid else []))[:3]]
        concepts = await asyncio.wait_for(
            ent_mod.propose_concepts(key_paths, focus), timeout=90.0) if key_paths else []
        if not concepts:
            trace(session_id, "segment_track_sam", "received", "no trackable concepts — empty registry")
            store.save_entities(session_id, [], [])
            return [], []
        trace(session_id, "segment_track_sam", "sent",
              f"{len(concepts)} SAM streams: {[c.get('prompt', '')[:40] for c in concepts]}")
        trace(session_id, "segment_track_sam", "waiting", "SAM 3.1 segmenting+tracking…")
        t0 = time.time()
        entities, observations = await ent_mod.track_session(
            file_id, concepts, fps, n_frames, media_wh, timeout_per_concept=TRACK_TIMEOUT)
        trace(session_id, "segment_track_sam", "received",
              f"{len(entities)} entities, {len(observations)} observations in {time.time() - t0:.0f}s")
        store.save_entities(session_id, entities, observations)
        return entities, observations
    except (asyncio.TimeoutError, Exception) as e:
        trace(session_id, "segment_track_sam", "failed", f"{type(e).__name__}: {e}"[:300])
        store.save_entities(session_id, [], [])
        return [], []


async def _ocr_tool(session_id: str, frame_rows: list[dict]) -> list[dict]:
    import store
    import ocr as ocr_mod
    trace(session_id, "ocr_frames", "sent", f"{min(5, len(frame_rows))} sharpest keyframes -> Spark OCR (concurrent)")
    trace(session_id, "ocr_frames", "waiting", "reading on-screen text…")
    t0 = time.time()
    try:
        rows = await asyncio.wait_for(ocr_mod.run_ocr(frame_rows), timeout=OCR_TIMEOUT_TOTAL)
        trace(session_id, "ocr_frames", "received", f"{len(rows)} voted text(s) in {time.time() - t0:.0f}s")
        store.save_ocr(session_id, rows)
        return rows
    except (asyncio.TimeoutError, Exception) as e:
        trace(session_id, "ocr_frames", "failed", f"{type(e).__name__}: {e}"[:300])
        store.save_ocr(session_id, [])
        return []


async def _audio_ev_tool(session_id: str, wav_bytes: bytes | None) -> list[dict]:
    import store
    import audio_events as audio_ev
    trace(session_id, "detect_audio_events", "sent", "WAV -> energy/onset detector (local)")
    try:
        rows = audio_ev.detect(wav_bytes)
        trace(session_id, "detect_audio_events", "received", f"{len(rows)} audio event(s)")
        store.save_audio_events(session_id, rows)
        return rows
    except Exception as e:
        trace(session_id, "detect_audio_events", "failed", f"{type(e).__name__}: {e}"[:300])
        store.save_audio_events(session_id, [])
        return []


async def _understand_tool(session_id: str, file_id: str | None, focus: str,
                           transcript_task, duration_s: float, upload_task=None) -> dict:
    """Spark video understanding as soon as upload + transcript are both ready.

    Called before frame observation finishes so the model watches the video
    while local decode runs; the result is awaited later in fusion.
    """
    from intel import analyze_video

    tr = await asyncio.ensure_future(transcript_task)
    if upload_task is not None:
        file_id = await upload_task
    trace(session_id, "understand_verify", "sent",
          f"video + transcript ({len(tr.get('transcript', ''))} chars) -> muse-spark")
    trace(session_id, "understand_verify", "waiting", "Spark watching video…")
    return await asyncio.wait_for(
        analyze_video(file_id or "", focus, tr.get("transcript", ""),
                      tr.get("turns", []), duration_s),
        timeout=UNDERSTAND_TIMEOUT)


async def run_perception(session_id: str, file_id: str, frame_rows: list[dict],
                         wav_bytes: bytes | None, focus: str, fps: float,
                         n_frames: int, media_wh: tuple, duration_s: float = 0.0,
                         transcript_task=None, audio_task=None,
                         understanding_task=None) -> dict:
    """Run the four perception tools concurrently. Never raises — honest empty.

    Transcript/audio/understanding may be pre-started by the caller (they need
    no frames) so remote work overlaps local frame decode and SAM tracking.
    """
    tr_task = asyncio.ensure_future(transcript_task or _transcribe_tool(session_id, wav_bytes))
    au_task = asyncio.ensure_future(audio_task or _audio_ev_tool(session_id, wav_bytes))
    intel_task = asyncio.ensure_future(
        understanding_task or _understand_tool(session_id, file_id, focus, tr_task, duration_s))
    tk_task = _track_tool(session_id, file_id, frame_rows, focus, fps, n_frames, media_wh)
    ocr_task = _ocr_tool(session_id, frame_rows)
    try:
        tr, (entities, observations), ocr_rows, audio_rows = await asyncio.gather(
            tr_task, tk_task, ocr_task, au_task)
    except Exception:
        intel_task.cancel()
        raise
    return {"transcript": tr, "entities": entities, "observations": observations,
            "ocr": ocr_rows, "audio": audio_rows, "intel_task": intel_task}


async def run_fusion_understanding(session_id: str, file_id: str, focus: str,
                                   perc: dict, frame_rows: list[dict],
                                   duration_s: float) -> dict:
    """Event graph -> await pre-started Spark understanding -> evidence verification."""
    import store
    import temporal
    import evidence as evmod

    trace(session_id, "extract_event_graph", "sent",
          f"fusing {len(perc['entities'])} entities + {len(perc['transcript'].get('turns', []))} turns + "
          f"{len(perc['ocr'])} ocr + {len(perc['audio'])} audio")
    trace(session_id, "extract_event_graph", "waiting", "Spark proposing events…")
    t0 = time.time()
    try:
        import events as ev_graph
        graph_events, _ = await asyncio.wait_for(
            ev_graph.extract_events(
                perc["entities"], perc["observations"], perc["transcript"]["turns"],
                perc["ocr"], perc["audio"], duration_s, focus),
            timeout=EVENT_GRAPH_TIMEOUT)
        trace(session_id, "extract_event_graph", "received",
              f"{len(graph_events)} grounded event(s) in {time.time() - t0:.0f}s")
    except (asyncio.TimeoutError, Exception) as e:
        trace(session_id, "extract_event_graph", "failed", f"{type(e).__name__}: {e}"[:300])
        graph_events = []
    store.save_events(session_id, graph_events)

    intel_task = perc.get("intel_task")
    if intel_task is None:  # direct callers: start now
        trace(session_id, "understand_verify", "sent",
              f"video + transcript ({len(perc['transcript'].get('transcript', ''))} chars) -> muse-spark")
        trace(session_id, "understand_verify", "waiting", "Spark watching video…")
        from intel import analyze_video
        intel_task = asyncio.create_task(asyncio.wait_for(
            analyze_video(file_id, focus, perc["transcript"]["transcript"],
                          perc["transcript"]["turns"], duration_s or 0.0),
            timeout=UNDERSTAND_TIMEOUT))
    intel = await intel_task
    trace(session_id, "understand_verify", "received",
          f"summary + {len(intel['moments'])} moment(s) — verifying evidence…")

    known_frames = {int(r["frame_index"]) for r in frame_rows}
    verified = []
    for m in intel["moments"]:
        ok, _ = evmod.verify([{"start": m["t_start"], "end": m["t_end"],
                               "frame_ids": sorted(known_frames)[:4]}],
                             duration_s, known_frames) if known_frames else (True, "no frame store")
        if ok:
            verified.append(m)
    _ = temporal.contradictions(
        [{"event_id": m["moment_id"], "start_t": m["t_start"], "end_t": m["t_end"]}
         for m in verified])
    trace(session_id, "understand_verify", "received",
          f"{len(verified)}/{len(intel['moments'])} moments grounded in frame store")
    return {"intel": intel, "verified": verified, "graph_events": graph_events}
