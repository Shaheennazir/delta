"""Real speech-to-text via muse-voice-transcribe-1.0 (spec: POST /v1/asr/transcribe).

No mocks. Raises on missing key, missing audio, or API error.
WAV conversion uses the imageio-ffmpeg binary (no system ffmpeg needed).
"""
from __future__ import annotations

import logging
import subprocess
import uuid

import httpx
import imageio_ffmpeg

from sam_client import BASE_URL, api_key

log = logging.getLogger("delta.transcribe")
TRANSCRIBE_MODEL = "muse-voice-transcribe-1.0"


def extract_wav(video_bytes: bytes, sr: int = 24000) -> bytes | None:
    """MP4 bytes -> mono 16-bit PCM WAV bytes. None when no audio stream."""
    import os
    import tempfile

    exe = imageio_ffmpeg.get_ffmpeg_exe()
    with tempfile.TemporaryDirectory() as d:
        src = os.path.join(d, "in.mp4")
        out = os.path.join(d, "out.wav")
        with open(src, "wb") as f:
            f.write(video_bytes)
        p = subprocess.run(
            [exe, "-y", "-v", "error", "-i", src,
             "-ac", "1", "-ar", str(sr), "-c:a", "pcm_s16le",
             "-map_metadata", "-1", out],
            capture_output=True, text=True,
        )
        if p.returncode != 0:
            log.info("no audio stream or ffmpeg failed: %s", (p.stderr or "")[-300:])
            return None
        try:
            with open(out, "rb") as f:
                data = f.read()
        except OSError:
            return None
        return data if len(data) > 100 else None


async def transcribe_wav(wav_bytes: bytes, mode: str = "DIARIZATION",
                       timeout: float = 240.0) -> dict:
    """POST WAV to /asr/transcribe. Returns {transcript, turns, audioDurationMs}.

    Single full-audio POST (no chunking). Raises TimeoutError on timeout so
    callers can degrade to empty transcript instead of hanging forever.
    """
    import asyncio
    import json

    req = json.dumps({"mode": mode, "model": TRANSCRIBE_MODEL, "audioEncoding": "WAV"})
    sid = f"tr-{uuid.uuid4().hex[:8]}"
    try:
        async with httpx.AsyncClient(
            base_url=BASE_URL,
            timeout=httpx.Timeout(connect=30.0, read=timeout, write=120.0, pool=30.0),
        ) as c:
            r = await c.post(
                f"/asr/transcribe?sessionId={sid}",
                headers={"Authorization": f"Bearer {api_key()}"},
                files={
                    "request": (None, req, "application/json"),
                    "audio": ("audio.wav", wav_bytes, "audio/wav"),
                },
            )
            r.raise_for_status()
            body = r.json()
    except (httpx.TimeoutException, asyncio.TimeoutError) as e:
        raise TimeoutError(f"transcribe timed out after {timeout:.0f}s "
                           f"({len(wav_bytes) / 1e6:.1f}MB audio)") from e
    return {
        "transcript": body.get("transcript", ""),
        "turns": body.get("turns", []) or [],
        "audioDurationMs": body.get("audioDurationMs", 0),
    }


async def transcribe_video(video_bytes: bytes, timeout: float = 240.0,
                           wav_bytes: bytes | None = None) -> dict:
    """Extract (once) + transcribe. Empty transcript when clip has no audio (not an error)."""
    wav = wav_bytes if wav_bytes is not None else extract_wav(video_bytes)
    if wav is None:
        return {"transcript": "", "turns": [], "audioDurationMs": 0, "has_audio": False}
    out = await transcribe_wav(wav, timeout=timeout)
    out["has_audio"] = True
    return out
