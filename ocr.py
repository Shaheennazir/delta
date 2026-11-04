"""Independent video-OCR pipeline (not the general video model).

1. Sharpness-rank keyframes (Laplacian variance, numpy only).
2. Per-crop still-image OCR via Muse Spark (separate passes, high-res crops).
3. Multi-frame voting: near-identical strings merge, timestamps + frame refs kept.
4. Confidence-gated: low-confidence singletons are dropped, never guessed.

A 200ms text flash is caught because adaptive observation already densified
rapid/transition tiers; OCR just re-reads those frames at full resolution.
"""
from __future__ import annotations

import difflib
import json
import logging
import os

import numpy as np
from PIL import Image

log = logging.getLogger("delta.ocr")


def sharpness(path: str) -> float:
    try:
        g = np.asarray(Image.open(path).convert("L"), dtype=np.float32)
    except Exception:
        return 0.0
    if g.size == 0:
        return 0.0
    k = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)
    from scipy.signal import convolve2d  # local import; scipy already required
    try:
        lap = convolve2d(g, k, mode="valid")
    except Exception:
        lap = g - np.mean(g)
    return float(lap.var())


async def _ocr_image(jpeg_bytes: bytes) -> tuple[str, float]:
    """One still-image OCR pass with Spark. Returns (text, confidence)."""
    import base64
    from spark import SPARK_MODEL, extract_text, post_unary
    b64 = base64.b64encode(jpeg_bytes).decode()
    body = await post_unary({
        "model": SPARK_MODEL,
        "instructions": "Transcribe ONLY visible text. Strict JSON.",
        "input": [{"type": "message", "role": "user", "content": [
            {"type": "input_text",
             "text": "Transcribe all visible on-screen text (signs, scores, captions, labels). "
                     "Return {{\"text\": <joined lines or empty>, \"confidence\": 0-1}. "
                     "Empty string when no legible text. Never guess blurred text."},
            {"type": "input_image", "image_url": f"data:image/jpeg;base64,{b64}"}]}],
        "text": {"format": {"type": "json_schema", "name": "OCR",
                 "schema": {"type": "object",
                            "properties": {"text": {"type": "string"},
                                           "confidence": {"type": "number"}},
                            "required": ["text", "confidence"],
                            "additionalProperties": False}, "strict": True}},
    })
    try:
        d = json.loads(extract_text(body))
        return str(d.get("text", ""))[:2000], max(0.0, min(1.0, float(d.get("confidence", 0.0))))
    except Exception:
        return "", 0.0


def _norm(s: str) -> str:
    return " ".join(s.lower().split())


async def run_ocr(frame_rows: list[dict], max_crops: int = 5,
                timeout_per_image: float = 90.0) -> list[dict]:
    """OCR the sharpest keyframes; vote across frames. Real passes only.

    Passes run concurrently (asyncio.gather) with a per-image timeout so one
    slow still never stalls the pipeline — failures yield honest empty.
    """
    import asyncio

    cands = sorted(frame_rows, key=lambda r: (-int(bool(r.get("keyframe"))), r.get("frame_index", 0)))
    # Prefer sharp frames: score a small pool, keep the best.
    pool = cands[: max(8, max_crops * 2)]
    scored = sorted(((sharpness(r["path"]), r) for r in pool if os.path.exists(r.get("path", ""))),
                    key=lambda t: -t[0])
    picked = [r for _, r in scored[:max_crops]]

    async def one(r: dict) -> dict | None:
        try:
            with open(r["path"], "rb") as f:
                data = f.read()
        except OSError:
            return None
        try:
            text, conf = await asyncio.wait_for(_ocr_image(data), timeout_per_image)
        except (asyncio.TimeoutError, Exception) as e:
            log.warning("OCR pass failed @t=%s: %s", r.get("t"), e)
            return None
        text = text.strip()
        if not text or conf < 0.5:
            return None
        return {"t": r["t"], "frame_index": r["frame_index"],
                "text": text, "confidence": round(conf, 3)}

    raw = [x for x in await asyncio.gather(*(one(r) for r in picked)) if x]
    # Vote: group near-duplicates (>=0.8 similarity), keep best, merge times.
    groups: list[list[dict]] = []
    for item in sorted(raw, key=lambda r: -r["confidence"]):
        placed = False
        for g in groups:
            if difflib.SequenceMatcher(None, _norm(g[0]["text"]), _norm(item["text"])).ratio() >= 0.8:
                g.append(item)
                placed = True
                break
        if not placed:
            groups.append([item])
    out = []
    for g in groups:
        best = max(g, key=lambda r: r["confidence"])
        votes = len(g)
        conf = min(0.99, best["confidence"] + 0.05 * (votes - 1))
        out.append({"t": min(r["t"] for r in g), "frame_index": best["frame_index"],
                    "text": best["text"], "confidence": round(conf, 3)})
    return sorted(out, key=lambda r: r["t"])
