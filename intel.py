"""Real video understanding via muse-spark-1.3-contributor (Responses + Files API).

No mocks. Takes an uploaded file_id + transcript and returns grounded
summary/themes/moments/on-screen-text. Every moment must carry timestamps —
this is the engineering fix for language priors and shuffled-frame shortcuts.
"""
from __future__ import annotations

import json
import logging

from sam_client import api_key  # noqa: F401  (key check: raises NoApiKey when missing)
from spark import SPARK_MODEL, extract_text, post_unary

log = logging.getLogger("delta.intel")

SYSTEM = """ROLE: Video Intelligence analyst.
INPUT: a viewer prompt (question or focus) plus a video file and its speech transcript with turn timestamps.
TASK:
1. Watch the video (frames + embedded audio) and read the transcript.
2. Answer the viewer prompt directly first (1-3 sentences). It may be a question ("which side played better?") or an instruction ("summarize…"); answer from the video, not from assumptions.
3. Then write a concise summary (3-6 sentences), 3-8 themes, and 3-12 timestamped moments.
4. Every moment MUST be grounded in something visible or audible at that time — never guess from language priors alone. If unsure, drop the moment.
5. Transcribe visible on-screen text briefly (empty string when none).
6. Timestamps are seconds from video start; end > start; keep within duration."""

SCHEMA = {
    "type": "object",
    "properties": {
        "answer": {"type": "string"},
        "summary": {"type": "string"},
        "themes": {"type": "array", "items": {"type": "string"}},
        "moments": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "t_start": {"type": "number"},
                    "t_end": {"type": "number"},
                    "label": {"type": "string"},
                    "detail": {"type": "string"},
                },
                "required": ["t_start", "t_end", "label", "detail"],
                "additionalProperties": False,
            },
        },
        "on_screen_text": {"type": "string"},
    },
    "required": ["answer", "summary", "themes", "moments", "on_screen_text"],
    "additionalProperties": False,
}


def _strict(schema: dict) -> dict:
    """Responses API strict json_schema needs required=[every property], recursively."""
    import copy

    s = copy.deepcopy(schema)

    def fix(node) -> None:
        if isinstance(node, dict):
            if isinstance(node.get("properties"), dict):
                node["required"] = sorted(node["properties"])
                node["additionalProperties"] = False
            for sub in node.values():
                fix(sub)
        elif isinstance(node, list):
            for sub in node:
                fix(sub)

    fix(s)
    return s


async def analyze_video(file_id: str, focus: str, transcript: str,
                        turns: list | None = None, duration_s: float = 0.0) -> dict:
    """Real Spark call. Raises on any API/model error (no fallback)."""
    turn_txt = ""
    for t in (turns or [])[:40]:
        spk = t.get("speaker") or "?"
        turn_txt += f"[{t.get('startMs', 0) / 1000:.1f}-{t.get('endMs', 0) / 1000:.1f}s spk{spk}] {t.get('transcript', '')}\n"
    prompt = (
        f"Viewer prompt: {focus}\n"
        f"Duration: {duration_s:.1f}s\n"
        f"Speech transcript:\n{transcript or '(no speech detected)'}\n"
        f"Turns:\n{turn_txt or '(none)'}\n"
        "First answer the viewer prompt directly. Then return summary, themes, grounded "
        "moments with second timestamps, and on-screen text."
    )
    body = await post_unary({
        "model": SPARK_MODEL,
        "instructions": SYSTEM,
        "input": [{"type": "message", "role": "user", "content": [
            {"type": "input_text", "text": prompt},
            {"type": "input_file", "file_id": file_id},
        ]}],
        "text": {"format": {
            "type": "json_schema", "name": "VideoUnderstanding",
            "schema": _strict(SCHEMA), "strict": True,
        }},
    })
    out = json.loads(extract_text(body))
    # Clamp timestamps into range; drop invalid moments.
    moments = []
    for i, m in enumerate(out.get("moments", []) or []):
        try:
            s, e = float(m["t_start"]), float(m["t_end"])
        except (KeyError, TypeError, ValueError):
            continue
        if duration_s > 0:
            s, e = max(0.0, s), min(duration_s, max(e, s + 0.5))
        if e <= s:
            continue
        moments.append({"moment_id": f"m_{i + 1:02d}", "t_start": round(s, 2),
                        "t_end": round(e, 2), "label": str(m.get("label", ""))[:120],
                        "detail": str(m.get("detail", ""))[:600]})
    return {"answer": str(out.get("answer", ""))[:1500],
            "summary": str(out.get("summary", ""))[:4000],
            "themes": [str(t)[:80] for t in out.get("themes", [])][:8],
            "moments": moments[:12],
            "on_screen_text": str(out.get("on_screen_text", ""))[:2000]}
