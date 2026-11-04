"""Explicit event graph: deterministic candidates, LLM labels only.

Timestamps are no longer invented by the language model. Candidate events are
extracted from real signals first — entity presence/motion segments, speech
turns, OCR rows, audio events — with exact start/end from those signals. Spark
may then name each candidate (action/detail/confidence) by candidate_id; it
cannot add, move or stretch an event. Events carry typed state transitions
(presence, motion/direction, speaking, on-screen text, audio) and are validated
by temporal.py.
"""
from __future__ import annotations

import json
import logging

log = logging.getLogger("delta.events")

MOTION_THRESHOLD = 0.15  # bbox diagonals per second
PRESENCE_GAP_S = 2.0
MAX_CANDIDATES = 80


def _frames_in(frame_rows: list[dict], start: float, end: float, limit: int = 12) -> list[int]:
    return [int(r["frame_index"]) for r in (frame_rows or [])
            if start - 0.15 <= float(r.get("t", 0)) <= end + 0.15][:limit]


def _candidate(cid: str, action: str, detail: str, start: float, end: float,
               entity_ids: list[str], source: str, confidence: float,
               frame_rows: list[dict], before: dict | None = None,
               after: dict | None = None) -> dict:
    return {"candidate_id": cid, "action": action, "detail": detail,
            "start_t": round(float(start), 3), "end_t": round(float(end), 3),
            "entity_ids": entity_ids, "source": source,
            "confidence": round(max(0.0, min(1.0, float(confidence))), 3),
            "state_before": before or {}, "state_after": after or {},
            "evidence_frames": _frames_in(frame_rows, start, end)}


def _segments(observations: list[dict], gap_s: float = PRESENCE_GAP_S) -> list[list[dict]]:
    obs = sorted(observations, key=lambda o: float(o["t"]))
    out: list[list[dict]] = []
    for o in obs:
        if out and float(o["t"]) - float(out[-1][-1]["t"]) <= gap_s:
            out[-1].append(o)
        else:
            out.append([o])
    return out


def _motion_runs(obs: list[dict]) -> list[dict]:
    """Runs of consecutive observations whose normalised displacement is real motion."""
    runs: list[dict] = []
    cur = None
    for a, b in zip(obs, obs[1:]):
        dt = max(1e-3, float(b["t"]) - float(a["t"]))
        cx = (float(a["bbox"][0]) + float(a["bbox"][2])) / 2
        cy = (float(a["bbox"][1]) + float(a["bbox"][3])) / 2
        dx = (float(b["bbox"][0]) + float(b["bbox"][2])) / 2 - cx
        dy = (float(b["bbox"][1]) + float(b["bbox"][3])) / 2 - cy
        diag = max(1.0, max(float(a["bbox"][2]) - float(a["bbox"][0]),
                           float(a["bbox"][3]) - float(a["bbox"][1])))
        speed = ((dx * dx + dy * dy) ** 0.5 / diag) / dt
        if speed >= MOTION_THRESHOLD:
            if cur is None:
                cur = {"start": float(a["t"]), "end": float(b["t"]), "dx": dx, "dy": dy,
                       "peak": speed}
            else:
                cur["end"] = float(b["t"])
                cur["dx"] += dx
                cur["dy"] += dy
                cur["peak"] = max(cur["peak"], speed)
        elif cur is not None:
            runs.append(cur)
            cur = None
    if cur is not None:
        runs.append(cur)
    return runs


def _direction(dx: float, dy: float) -> str:
    if abs(dx) >= abs(dy):
        return "right" if dx >= 0 else "left"
    return "down" if dy >= 0 else "up"


def build_candidates(entities: list[dict], observations: list[dict],
                     turns: list[dict], ocr_rows: list[dict],
                     audio_rows: list[dict], duration: float,
                     frame_rows: list[dict] | None = None) -> list[dict]:
    """Deterministic signal-derived events. Empty signal => empty graph."""
    cands: list[dict] = []
    labels = {e["entity_id"]: (e.get("label") or e.get("kind") or "object")
              for e in entities}
    for ent in entities:
        eid = ent["entity_id"]
        obs = [o for o in observations if o.get("entity_id") == eid]
        if not obs:
            continue
        label = labels.get(eid, "object")
        conf = float(ent.get("confidence", 0.5))
        for seg in _segments(obs):
            s, e = float(seg[0]["t"]), float(seg[-1]["t"])
            if e - s < 0.2:
                continue
            cands.append(_candidate(
                f"cand_e_{len(cands) + 1:03d}", f"{label} present",
                f"{label} observed in {len(seg)} frame(s)", s, e, [eid], "entity",
                conf, frame_rows or [], {"presence": "absent"}, {"presence": "present"}))
            for run in _motion_runs(seg):
                if run["end"] - run["start"] < 0.2:
                    continue
                direction = _direction(run["dx"], run["dy"])
                cands.append(_candidate(
                    f"cand_e_{len(cands) + 1:03d}", f"{label} moves {direction}",
                    f"net displacement ({run['dx']:.1f},{run['dy']:.1f}) px, "
                    f"peak {run['peak']:.2f} diag/s", run["start"], run["end"], [eid],
                    "entity", conf, frame_rows or [],
                    {"motion": "still"}, {"motion": "moving", "direction": direction}))

    for i, t in enumerate(turns or []):
        s, e = float(t.get("startMs", 0)) / 1000.0, float(t.get("endMs", 0)) / 1000.0
        if e <= s:
            continue
        speaker = t.get("speaker")
        cands.append(_candidate(
            f"cand_s_{i + 1:03d}", "speech", str(t.get("transcript", ""))[:400], s, e,
            [], "speech", 0.9, frame_rows or [],
            {"speaking": "false"},
            {"speaking": "true", **({"speaker": str(speaker)} if speaker else {})}))

    for i, r in enumerate(ocr_rows or []):
        t = float(r.get("t", 0))
        e = min(duration or t + 0.5, t + 0.5)
        if e <= t:
            continue
        cands.append(_candidate(
            f"cand_o_{i + 1:03d}", "on-screen text", str(r.get("text", ""))[:300], t, e,
            [], "ocr", float(r.get("confidence", 0.5)), frame_rows or [],
            {}, {"on_screen_text": str(r.get("text", ""))[:200]}))

    for r in audio_rows or []:
        s, e = float(r.get("start_t", 0)), float(r.get("end_t", 0))
        if e <= s:
            continue
        kind = str(r.get("kind", "event"))
        cands.append(_candidate(
            f"cand_a_{len(cands) + 1:03d}", f"audio {kind}",
            str(r.get("detail", ""))[:300], s, e, [], "audio",
            float(r.get("confidence", 0.5)), frame_rows or [],
            {}, {"audio": kind}))

    cands.sort(key=lambda c: (c["start_t"], c["end_t"], c["candidate_id"]))
    return cands[:MAX_CANDIDATES]


def state_timeline(events: list[dict]) -> list[dict]:
    """Per-entity typed state intervals, merged when contiguous and unchanged."""
    rows: list[dict] = []
    for e in sorted(events, key=lambda x: float(x.get("start_t", 0))):
        for eid in e.get("entity_ids") or []:
            for key, value in (e.get("state_after") or {}).items():
                rows.append({"entity_id": eid, "key": key, "value": value,
                             "start": float(e.get("start_t", 0)),
                             "end": float(e.get("end_t", 0))})
    rows.sort(key=lambda r: (r["entity_id"], r["key"], r["start"]))
    merged: list[dict] = []
    for r in rows:
        last = merged[-1] if merged else None
        if (last and last["entity_id"] == r["entity_id"] and last["key"] == r["key"]
                and last["value"] == r["value"] and r["start"] - last["end"] <= 0.5):
            last["end"] = max(last["end"], r["end"])
        else:
            merged.append(dict(r))
    return merged


async def extract_events(entities: list[dict], observations: list[dict], turns: list[dict],
                         ocr_rows: list[dict], audio_rows: list[dict],
                         duration: float, focus: str = "",
                         frame_rows: list[dict] | None = None) -> tuple[list[dict], list[str]]:
    """Candidate extraction (deterministic) -> optional Spark labels -> validate."""
    import temporal
    if duration <= 0:
        return [], ["unknown duration"]
    cands = build_candidates(entities, observations, turns, ocr_rows, audio_rows,
                             duration, frame_rows)
    if not cands:
        return [], []
    labels: dict[str, dict] = {}
    try:
        from spark import SPARK_MODEL, extract_text, post_unary
        lines = [f"{c['candidate_id']} [{c['start_t']:.2f}-{c['end_t']:.2f}s] "
                 f"{c['action']} | {c['detail'][:110]} | {','.join(c['entity_ids'])}"
                 for c in cands]
        schema = {"type": "object", "properties": {"labels": {"type": "array", "items": {
            "type": "object",
            "properties": {"candidate_id": {"type": "string"}, "action": {"type": "string"},
                           "detail": {"type": "string"}, "confidence": {"type": "number"}},
            "required": ["candidate_id", "action", "detail", "confidence"],
            "additionalProperties": False}}},
            "required": ["labels"], "additionalProperties": False}
        body = await post_unary({
            "model": SPARK_MODEL,
            "instructions": "Name signal-derived events. Never invent an id, time, entity "
                            "or observation. Strict JSON.",
            "input": [{"type": "message", "role": "user", "content": [{"type": "input_text", "text":
                f"Focus: {focus}\n{duration:.1f}s video. Candidates (id [start-end] base | detail | entities):\n"
                + "\n".join(lines)[:8000] +
                "\nFor every candidate return a concise action verb phrase and one-sentence "
                "detail grounded in the candidate. Confidence 0-1. Use only these ids."}]}],
            "text": {"format": {"type": "json_schema", "name": "EventLabels",
                                "schema": schema, "strict": True}},
        })
        for item in json.loads(extract_text(body)).get("labels", []) or []:
            cid = str(item.get("candidate_id", ""))
            if cid:
                labels[cid] = item
    except Exception as e:
        log.warning("event labelling failed, keeping deterministic labels: %s", e)

    known = {e["entity_id"] for e in entities}
    events = []
    for c in cands:
        lab = labels.get(c["candidate_id"], {})
        action = str(lab.get("action") or c["action"])[:120]
        detail = str(lab.get("detail") or c["detail"])[:400]
        conf = lab.get("confidence", c["confidence"])
        try:
            conf = max(0.0, min(1.0, float(conf)))
        except (TypeError, ValueError):
            conf = c["confidence"]
        events.append({
            "event_id": "", "candidate_id": c["candidate_id"], "source": c["source"],
            "action": action, "detail": detail,
            "entity_ids": [x for x in c["entity_ids"] if x in known][:4],
            "start_t": c["start_t"], "end_t": c["end_t"],
            "evidence": [{"start": c["start_t"], "end": c["end_t"],
                          "frame_ids": c["evidence_frames"]}],
            "confidence": conf, "state_before": c["state_before"],
            "state_after": c["state_after"],
        })
    events = temporal.validate_clamp(events, duration)
    for i, e in enumerate(events):
        e["event_id"] = f"E{i + 1:02d}"
    return events, temporal.contradictions(events, duration)
