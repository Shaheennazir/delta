"""Query-directed evidence retrieval over the stored temporal index.

Builds one uniform evidence item type from events, entities/tracks, speech
turns, OCR rows and audio events. Scores lexical relevance (IDF-weighted)
plus temporal hints in the question and evidence confidence, then returns
exact spans + frame ids. The investigator cites these item ids; spans are
never re-estimated by the language model.
"""
from __future__ import annotations

import math
import re

STOP = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "to", "of", "in",
    "on", "at", "for", "and", "or", "what", "which", "who", "when", "where",
    "how", "why", "did", "does", "do", "that", "this", "it", "with", "from",
    "by", "as", "before", "after", "during", "video", "clip", "show", "me",
    "there", "their", "his", "her", "they", "we", "you", "i",
}


def tokens(text: str) -> list[str]:
    return [w for w in re.findall(r"[a-z0-9]+", (text or "").lower())
            if len(w) > 1 and w not in STOP]


def time_hint(question: str, duration: float) -> tuple[float, float] | None:
    q = (question or "").lower()
    pair = re.search(r"(?:between|from)\s*(\d+(?:\.\d+)?)\s*(?:s|sec|seconds?)?\s*(?:and|to)\s*"
                     r"(\d+(?:\.\d+)?)", q)
    if pair:
        a, b = sorted((float(pair.group(1)), float(pair.group(2))))
        return max(0.0, a), min(duration, b) if duration else b
    at = re.search(r"(?:at|around|near)\s*(\d+(?:\.\d+)?)\s*(?:s|sec|seconds?)", q)
    if at:
        t = float(at.group(1))
        return max(0.0, t - 2.0), min(duration, t + 2.0) if duration else t + 2.0
    mmss = re.search(r"\b(\d{1,2}):(\d{2})\b", q)
    if mmss:
        t = int(mmss.group(1)) * 60 + int(mmss.group(2))
        return max(0.0, t - 2.0), min(duration, t + 2.0) if duration else t + 2.0
    return None


def _frames_in(frames: list[dict], start: float, end: float, limit: int = 6) -> list[int]:
    return [int(r["frame_index"]) for r in frames
            if start - 0.25 <= float(r.get("t", 0)) <= end + 0.25][:limit]


def build(session_id: str, events: list[dict], entities: list[dict],
          observations: list[dict], turns: list[dict], ocr_rows: list[dict],
          audio_rows: list[dict], frames: list[dict], duration: float) -> list[dict]:
    labels = {e["entity_id"]: e.get("label", "") for e in entities}
    items: list[dict] = []

    for e in events:
        ids = e.get("entity_ids") or []
        text = " ".join([e.get("action", ""), e.get("detail", ""),
                         " ".join(labels.get(i, i) for i in ids)])
        fids = [int(f) for sp in (e.get("evidence") or [])
                for f in (sp.get("frame_ids") or [])]
        items.append({"id": e["event_id"], "kind": "event", "text": text,
                      "start": float(e.get("start_t", 0)), "end": float(e.get("end_t", 0)),
                      "frame_ids": fids or _frames_in(frames, float(e.get("start_t", 0)),
                                                      float(e.get("end_t", 0))),
                      "confidence": float(e.get("confidence", 0.0))})

    for ent in entities:
        obs = [o for o in observations if o.get("entity_id") == ent["entity_id"]]
        if not obs:
            continue
        start = min(float(o["t"]) for o in obs)
        end = max(float(o["t"]) for o in obs)
        items.append({"id": ent["entity_id"], "kind": "entity",
                      "text": f"{ent.get('label', '')} {ent.get('kind', '')}",
                      "start": start, "end": end,
                      "frame_ids": [int(o.get("frame_index", 0)) for o in obs][-6:],
                      "confidence": float(ent.get("confidence", 0.0))})

    for i, t in enumerate(turns or []):
        s, e = float(t.get("startMs", 0)) / 1000.0, float(t.get("endMs", 0)) / 1000.0
        if e <= s:
            continue
        items.append({"id": f"turn_{i + 1:03d}", "kind": "speech",
                      "text": str(t.get("transcript", "")), "start": s, "end": e,
                      "frame_ids": _frames_in(frames, s, e),
                      "confidence": 0.9 if t.get("transcript") else 0.0})

    for i, r in enumerate(ocr_rows or []):
        t = float(r.get("t", 0))
        items.append({"id": f"ocr_{i + 1:03d}", "kind": "ocr",
                      "text": str(r.get("text", "")), "start": t, "end": t + 0.5,
                      "frame_ids": [int(r.get("frame_index", 0))],
                      "confidence": float(r.get("confidence", 0.0))})

    for r in audio_rows or []:
        items.append({"id": r.get("event_id", ""), "kind": "audio",
                      "text": f"{r.get('kind', '')} {r.get('detail', '')}",
                      "start": float(r.get("start_t", 0)), "end": float(r.get("end_t", 0)),
                      "frame_ids": _frames_in(frames, float(r.get("start_t", 0)),
                                              float(r.get("end_t", 0))),
                      "confidence": float(r.get("confidence", 0.0))})
    return [i for i in items if i["end"] > i["start"] or i["kind"] == "entity"]


def retrieve(session_id: str, question: str, events: list[dict], entities: list[dict],
             observations: list[dict], turns: list[dict], ocr_rows: list[dict],
             audio_rows: list[dict], frames: list[dict], duration: float,
             top_k: int = 12) -> list[dict]:
    items = build(session_id, events, entities, observations, turns, ocr_rows,
                  audio_rows, frames, duration)
    if not items:
        return []
    q = set(tokens(question))
    df: dict[str, int] = {}
    docs = [set(tokens(i["text"])) for i in items]
    for d in docs:
        for w in d:
            df[w] = df.get(w, 0) + 1
    n = len(items)
    hint = time_hint(question, duration)
    # Score = TF-IDF lexical match + time-hint overlap fraction, both derived
    # from the query and index (no tuned weights). Stored confidence is only a
    # tiebreak, never added to the score.
    for item, doc in zip(items, docs):
        lexical = sum(math.log((n + 1) / (df[w] + 0.5)) for w in q & doc)
        norm = math.sqrt(max(1, len(doc))) if doc else 1.0
        score = lexical / norm
        if hint:
            overlap = max(0.0, min(item["end"], hint[1]) - max(item["start"], hint[0]))
            score += overlap / max(0.01, hint[1] - hint[0])
        item["score"] = round(score, 4)
    ranked = sorted(items, key=lambda i: (-i["score"], -float(i.get("confidence", 0.0)),
                                          i["start"], i["id"]))
    if q:
        matched = [i for i in ranked if i["score"] > 0.0]
        ranked = matched or ranked
    return ranked[:top_k]
