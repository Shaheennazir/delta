"""Measured audio-visual alignment (not assumed from a shared container).

Audio RMS and visual motion are sampled on a common 50 ms grid and cross-
correlated over ±2 s. The peak lag is the audio-clock offset relative to the
video clock; audio rows are shifted by it before fusion. Confidence is the peak
Pearson correlation, 0 when either signal is flat — an unmeasurable pair says so
instead of claiming sync.
"""
from __future__ import annotations

import numpy as np

HOP_S = 0.05
MAX_LAG_S = 2.0


def _resample(times: list[float], values: list[float], duration: float,
              hop: float = HOP_S) -> np.ndarray:
    n = max(2, int(round(duration / hop)) + 1)
    grid = np.arange(n) * hop
    if not times:
        return np.zeros(n)
    t = np.asarray(times, dtype=np.float64)
    v = np.asarray(values, dtype=np.float64)
    order = np.argsort(t)
    return np.interp(grid, t[order], v[order], left=v[order][0], right=v[order][-1])


def motion_series(frame_rows: list[dict], duration: float, hop: float = HOP_S) -> np.ndarray:
    rows = [r for r in (frame_rows or []) if r.get("motion") is not None]
    return _resample([float(r.get("t", 0)) for r in rows],
                     [float(r.get("motion", 0)) for r in rows], duration, hop)


def event_series(rows: list[dict], duration: float, hop: float = HOP_S) -> np.ndarray:
    n = max(2, int(round(duration / hop)) + 1)
    out = np.zeros(n)
    for r in rows or []:
        s = int(round(float(r.get("start_t", 0)) / hop))
        e = int(round(float(r.get("end_t", 0)) / hop))
        out[max(0, s):min(n, max(s + 1, e))] = 1.0
    return out


def _norm(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    sd = float(x.std())
    return (x - x.mean()) / sd if sd > 1e-9 else np.zeros_like(x)


def lag_curve(audio: np.ndarray, visual: np.ndarray,
              max_lag_s: float = MAX_LAG_S, hop: float = HOP_S) -> tuple[list[float], list[float]]:
    """Pearson correlation per lag. Positive lag => audio is late (offset > 0)."""
    a, v = _norm(audio), _norm(visual)
    n = min(len(a), len(v))
    a, v = a[:n], v[:n]
    L = int(round(max_lag_s / hop))
    lags, scores = [], []
    for k in range(-L, L + 1):
        if k >= 0:
            x, y = a[k:], v[:n - k]
        else:
            x, y = a[:n + k], v[-k:]
        lags.append(round(k * hop, 3))
        scores.append(round(float(np.mean(x * y)), 6) if len(x) > 4 else 0.0)
    return lags, scores


def _best(lags: list[float], scores: list[float]) -> dict:
    if not scores:
        return {"offset_s": 0.0, "confidence": 0.0}
    i = int(np.argmax(scores))
    return {"offset_s": float(lags[i]), "confidence": round(max(0.0, scores[i]), 4)}


def estimate_offset(wav_bytes: bytes | None, frame_rows: list[dict], duration: float) -> dict:
    """Audio RMS vs visual motion cross-correlation."""
    base = {"offset_s": 0.0, "confidence": 0.0, "hop_s": HOP_S,
            "method": "rms-vs-motion-xcorr", "n_audio": 0}
    if not wav_bytes or duration <= 0 or not frame_rows:
        return {**base, "method": "insufficient-signal"}
    import audio_events
    env, hop = audio_events.rms_envelope(wav_bytes, HOP_S)
    if len(env) < 10:
        return {**base, "method": "insufficient-audio"}
    audio = _resample([i * hop for i in range(len(env))], env, duration, HOP_S)
    visual = motion_series(frame_rows, duration, HOP_S)
    if float(audio.std()) <= 1e-9 or float(visual.std()) <= 1e-9:
        return {**base, "method": "flat-signal", "n_audio": len(env)}
    lags, scores = lag_curve(audio, visual)
    best = _best(lags, scores)
    return {**base, **best, "n_audio": len(env), "curve_peak": max(scores) if scores else 0.0}


def events_offset(audio_rows: list[dict], frame_rows: list[dict],
                  duration: float) -> dict:
    """Event-vs-motion alignment; used for adversarial desync checks."""
    if not audio_rows or duration <= 0 or not frame_rows:
        return {"offset_s": 0.0, "confidence": 0.0, "method": "insufficient-signal"}
    a = event_series(audio_rows, duration)
    v = motion_series(frame_rows, duration)
    if float(a.std()) <= 1e-9 or float(v.std()) <= 1e-9:
        return {"offset_s": 0.0, "confidence": 0.0, "method": "flat-signal"}
    lags, scores = lag_curve(a, v)
    return {"method": "events-vs-motion-xcorr", **_best(lags, scores)}
