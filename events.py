"""Explicit event graph construction (temporal fusion).

Fuses entity trajectories + speech turns + OCR texts + audio events into
timestamped VideoEvents via one Spark proposal call, then enforces
determinism: clamp to duration, drop end<=start, order, contradiction
report. The LLM proposes; temporal.py disposes.
"""
from __future__ import annotations

import json
import logging

log = logging.getLogger("delta.events")


def _ctx(entities, observations, turns, ocr_rows, audio_rows, duration: float) -> str:
    lines = [f"Duration: {duration:.1f}s."]
    lines.append(f"Entities ({len(entities)}): " +
                 "; ".join(f"{e['entity_id']}={e.get('label','')} ({len([o for o in observations if o['entity_id']==e['entity_id']])} obs)"
                           for e in entities[:8]) or "none")
    for t in (turns or [])[:20]:
        lines.append(f"speech [{t.get('startMs',0)/1000:.1f}-{t.get('endMs',0)/1000:.1f}s] {t.get('transcript','')[:120]}")
    for o in (ocr_rows or [])[:10]:
        lines.append(f"onscreen [{o['t']:.1f}s] {o['text'][:120]}")
    for a in (audio_rows or [])[:10]:
        lines.append(f"audio {a['kind']} [{a['start_t']:.1f}-{a['end_t']:.1f}s] {a.get('detail','')}")
    return "\n".join(lines)[:6000]


async def extract_events(entities: list[dict], observations: list[dict], turns: list[dict],
                         ocr_rows: list[dict], audio_rows: list[dict],
                         duration: float, focus: str = "") -> tuple[list[dict], list[str]]:
    """Returns (events, contradictions). Empty (honest) when model yields none."""
    from spark import SPARK_MODEL, extract_text, post_unary
    import temporal
    if duration <= 0:
        return [], ["unknown duration"]
    schema = {
        "type": "object",
        "properties": {
            "events": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "action": {"type": "string"},
                        "entity_ids": {"type": "array", "items": {"type": "string"}},
                        "start_t": {"type": "number"},
                        "end_t": {"type": "number"},
                        "evidence_frames": {"type": "array", "items": {"type": "integer"}},
                        "confidence": {"type": "number"},
                        "state_before": {"type": "object"},
                        "state_after": {"type": "object"},
                        "detail": {"type": "string"},
                    },
                    "required": ["action", "entity_ids", "start_t", "end_t",
                                 "evidence_frames", "confidence",
                                 "state_before", "state_after", "detail"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["events"],
        "additionalProperties": False,
    }
    prompt = (
        f"Focus: {focus}\n{_ctx(entities, observations, turns, ocr_rows, audio_rows, duration)}\n"
        "Return 3-15 events covering movement, interaction, state transition, action. "
        "Each: action (verb phrase), entity_ids (subset of known ids or []), start_t/end_t seconds, "
        "evidence_frames (known frame indices or []), confidence 0-1, state_before/state_after "
        "small string maps, detail one sentence. Order by time. If unsure, omit.")
    body = await post_unary({
        "model": SPARK_MODEL,
        "instructions": "You are an event-graph extractor. Propose ONLY events with direct "
                        "evidence in the provided signals. Strict JSON.",
        "input": [{"type": "message", "role": "user", "content": [
            {"type": "input_text", "text": prompt}]}],
        "text": {"format": {"type": "json_schema", "name": "EventGraph",
                            "schema": schema, "strict": True}},
    })
    try:
        raw = json.loads(extract_text(body)).get("events", []) or []
    except Exception as e:
        log.warning("event proposal parse failed: %s", e)
        return [], [f"proposal parse failed: {e}"]
    known = {e["entity_id"] for e in entities}
    framed: set[int] = set()
    for o in observations:
        framed.add(int(o.get("frame_index", -1)))
    shaped = []
    for i, e in enumerate(raw[:15]):
        try:
            s, en = float(e["start_t"]), float(e["end_t"])
        except (KeyError, TypeError, ValueError):
            continue
        ev_frames = [int(f) for f in (e.get("evidence_frames") or []) if int(f) in framed][:12]
        shaped.append({
            "event_id": f"E{i+1:02d}", "action": str(e.get("action", ""))[:120],
            "entity_ids": [x for x in (e.get("entity_ids") or []) if x in known][:4],
            "start_t": s, "end_t": en,
            "evidence": [{"start": s, "end": en, "frame_ids": ev_frames}],
            "confidence": max(0.0, min(1.0, float(e.get("confidence", 0.5)))),
            "state_before": {k: str(v)[:200] for k, v in (e.get("state_before") or {}).items()},
            "state_after": {k: str(v)[:200] for k, v in (e.get("state_after") or {}).items()},
            "detail": str(e.get("detail", ""))[:400]})
    events = temporal.validate_clamp(shaped, duration)
    return events, temporal.contradictions(events)
