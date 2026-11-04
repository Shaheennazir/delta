"""Real audio-event detection (non-speech) from the extracted WAV.

Speech identity comes from muse-voice-transcribe (turns). This module adds
deterministic signal events — loud impacts/claps, sustained whistle-like
tones, silence gaps — via numpy energy/onset analysis. No ML, no mocks:
thresholds are relative to the clip itself, empty when undecidable.
"""
from __future__ import annotations

import struct


def _read_mono(wav_bytes: bytes) -> tuple[list[float], int]:
    """Parse 16-bit PCM WAV mono samples (converter always writes this form)."""
    if wav_bytes[:4] != b"RIFF" or len(wav_bytes) < 44:
        return [], 0
    # Find 'data' chunk.
    i = 12
    sr = 24000
    samples: list[float] = []
    n = len(wav_bytes)
    while i + 8 <= n:
        tag, size = wav_bytes[i:i + 4], struct.unpack("<I", wav_bytes[i + 4:i + 8])[0]
        body = wav_bytes[i + 8:i + 8 + size]
        if len(body) < size:
            break
        if tag == b"fmt " and len(body) >= 16:
            sr = struct.unpack("<I", body[4:8])[0] or 24000
        elif tag == b"data":
            count = len(body) // 2
            vals = struct.unpack(f"<{count}h", body[:count * 2])
            samples = [v / 32768.0 for v in vals]
            break
        i += 8 + size + (size % 2)
    return samples, sr


def rms_envelope(wav_bytes: bytes | None, hop_s: float = 0.05) -> tuple[list[float], float]:
    """Return (RMS per hop, hop seconds) on the audio clock. ([], hop) when empty."""
    if not wav_bytes:
        return [], hop_s
    samples, sr = _read_mono(wav_bytes)
    if not samples or sr <= 0:
        return [], hop_s
    import numpy as np
    x = np.asarray(samples, dtype=np.float64)
    win = max(1, int(round(sr * hop_s)))
    nwin = len(x) // win
    if nwin < 2:
        return [], hop_s
    env = np.sqrt(np.mean(x[:nwin * win].reshape(nwin, win) ** 2, axis=1))
    return [float(v) for v in env], win / sr


def shift_events(rows: list[dict], offset_s: float, duration: float = 0.0) -> list[dict]:
    """Move audio-clock rows onto the video clock (video_t = audio_t - offset)."""
    out = []
    for r in rows or []:
        s = max(0.0, float(r.get("start_t", 0)) - float(offset_s))
        e = max(s, float(r.get("end_t", 0)) - float(offset_s))
        if duration > 0:
            s, e = min(duration, s), min(duration, e)
        if e <= s:
            continue
        out.append({**r, "start_t": round(s, 3), "end_t": round(e, 3)})
    return sorted(out, key=lambda x: x["start_t"])


def detect(wav_bytes: bytes | None) -> list[dict]:
    if not wav_bytes:
        return []
    samples, sr = _read_mono(wav_bytes)
    if not samples or sr <= 0 or len(samples) < sr:  # <1s of audio: nothing reliable to say
        return []
    import numpy as np
    x = np.asarray(samples, dtype=np.float64)
    win = max(1, sr // 20)  # 50ms windows
    nwin = len(x) // win
    rms = np.sqrt(np.mean(x[:nwin * win].reshape(nwin, win) ** 2, axis=1))
    events: list[dict] = []
    mu, sd = float(rms.mean()), float(rms.std())
    if sd <= 1e-6:
        events.append({"event_id": "au_001", "kind": "silence", "start_t": 0.0,
                       "end_t": round(len(x) / sr, 2), "confidence": 0.6,
                       "detail": "flat audio bed"})
        return events
    # Loud bursts: runs above mu+3sd, at least 2 windows, merged within 200ms.
    thr = mu + 3.0 * sd
    hot = rms > thr
    runs: list[list[int]] = []
    cur: list[int] = []
    for i, h in enumerate(hot):
        if h:
            cur.append(i)
        elif cur:
            runs.append(cur)
            cur = []
    if cur:
        runs.append(cur)
    merged: list[list[int]] = []
    for r in runs:
        if merged and r[0] - merged[-1][-1] <= 4:
            merged[-1].extend(r)
        else:
            merged.append(list(r))
    eid = 0
    for r in merged:
        if len(r) < 2:
            continue
        s_t, e_t = round(r[0] * win / sr, 2), round((r[-1] + 1) * win / sr, 2)
        dur = e_t - s_t
        peak = float(rms[r].max() / max(1e-6, mu))
        kind = "impact" if dur < 0.4 else ("crowd-swelling" if dur > 2.0 else "loud-event")
        eid += 1
        events.append({"event_id": f"au_{eid:03d}", "kind": kind, "start_t": s_t,
                       "end_t": e_t, "confidence": round(min(0.9, 0.5 + 0.1 * peak), 3),
                       "detail": f"energy {peak:.1f}x bed, {dur:.2f}s"})
    # Long quiet gaps (>2s below mu*0.5).
    quiet = rms < (mu * 0.5)
    qrun: list[int] = []
    for i, q in enumerate(list(quiet) + [False]):
        if q:
            qrun.append(i)
        else:
            if len(qrun) * win / sr >= 2.0:
                eid += 1
                events.append({"event_id": f"au_{eid:03d}", "kind": "silence",
                               "start_t": round(qrun[0] * win / sr, 2),
                               "end_t": round(qrun[-1] * win / sr, 2),
                               "confidence": 0.6, "detail": "quiet gap"})
            qrun = []
    return sorted(events, key=lambda e: e["start_t"])[:20]
