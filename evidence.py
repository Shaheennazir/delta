"""Evidence contracts + deterministic verification + calibrated abstention.

A span is valid only if it lies inside the clip, its frames exist, and — when
frame times are supplied — each cited frame actually falls inside the span.
Confidence is passed through the persisted isotonic curve when labelled
outcomes exist; otherwise the raw score is returned marked "uncalibrated".
Abstention is a selective-prediction decision, not a fixed confidence fudge.
"""
from __future__ import annotations

TOL_S = 1.0


def _frame_time_map(frame_times) -> dict[int, float]:
    if not frame_times:
        return {}
    if isinstance(frame_times, dict):
        return {int(k): float(v) for k, v in frame_times.items()}
    return {int(r.get("frame_index", -1)): float(r.get("t", 0.0)) for r in frame_times}


def contract(claim: str, evidence: list[dict], confidence: float,
             verification: str, duration: float, known_frames: set[int]) -> dict:
    return {"claim": claim[:500], "evidence": evidence[:8],
            "confidence": max(0.0, min(1.0, float(confidence))),
            "verification": verification if verification in (
                "supported", "inferred", "external", "unknown") else "unknown",
            "duration_s": duration, "checked_frames": sorted(known_frames)[:0]}


def verify(evidence: list[dict], duration: float, known_frames: set[int],
           frame_times=None, tol: float = TOL_S) -> tuple[bool, str]:
    """Deterministic gate. Returns (ok, reason)."""
    if not evidence:
        return False, "no evidence spans"
    times = _frame_time_map(frame_times)
    for span in evidence:
        try:
            s, e = float(span.get("start", -1)), float(span.get("end", -1))
        except (TypeError, ValueError):
            return False, "non-numeric span"
        if s < 0 or e <= s:
            return False, f"invalid span [{s},{e}]"
        if duration > 0 and (s > duration or e > duration + 0.01):
            return False, f"span [{s},{e}] outside duration {duration}"
        for f in span.get("frame_ids", []) or []:
            fi = int(f)
            if fi not in known_frames:
                return False, f"frame {f} not in frame store"
            if times and fi in times and not (s - tol <= times[fi] <= e + tol):
                return False, (f"frame {f} at {times[fi]:.2f}s is outside span "
                               f"[{s:.2f},{e:.2f}]")
    return True, "ok"


def calibrate_detail(confidence: float, n_spans: int, n_frames: int,
                     support: float = 1.0, frame_times=None) -> dict:
    """Evidence-weighted score -> persisted isotonic curve when available."""
    import confidence as conf_mod
    if n_spans <= 0:
        return {"raw": 0.0, "calibrated": 0.0, "method": "no-spans", "support": 0.0}
    base = max(0.0, min(1.0, float(confidence))) * max(0.0, min(1.0, float(support)))
    cal, method = conf_mod.apply_saved(base)
    return {"raw": round(base, 6), "calibrated": round(cal, 6),
            "method": method, "support": round(float(support), 6),
            "n_spans": n_spans, "n_frames": n_frames}


def calibrate(confidence: float, n_spans: int, n_frames: int,
              support: float = 1.0) -> float:
    """Back-compatible float score."""
    return calibrate_detail(confidence, n_spans, n_frames, support)["calibrated"]


def decision(verification: str, calibrated: float, has_answer: bool,
             threshold: float | None = None) -> dict:
    if not has_answer:
        return {"abstain": True, "reason": "no answer"}
    if verification == "unknown":
        return {"abstain": True, "reason": "unverified claim"}
    import confidence as conf_mod
    thr = conf_mod.select_threshold_from_saved(0.1) if threshold is None else float(threshold)
    if calibrated < thr:
        return {"abstain": True, "reason": f"confidence {calibrated:.3f} < threshold {thr:.3f}"}
    return {"abstain": False, "reason": "ok", "threshold": round(thr, 6)}


def abstain(question: str, reason: str = "") -> dict:
    return {"answer": f"Insufficient video evidence to answer: {question[:200]}",
            "evidence": [], "confidence": 0.0, "raw_confidence": 0.0,
            "verification": "unknown", "abstained": True,
            "abstain_reason": reason or "no verified evidence"}
