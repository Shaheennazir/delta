"""Evidence-constrained reasoning: every factual claim carries a contract.

Contract: {claim, evidence[{start, end, frame_ids}], confidence, verification}
verification in {supported, inferred, external, unknown}.
The verifier is deterministic: timestamps must lie in [0, duration], frame ids
must exist in the frame store, and confidence must be calibrated to evidence
density. Anything ungrounded forces abstention — the model may not override.
"""
from __future__ import annotations


def contract(claim: str, evidence: list[dict], confidence: float,
             verification: str, duration: float, known_frames: set[int]) -> dict:
    return {"claim": claim[:500], "evidence": evidence[:8],
            "confidence": max(0.0, min(1.0, float(confidence))),
            "verification": verification if verification in (
                "supported", "inferred", "external", "unknown") else "unknown",
            "duration_s": duration, "checked_frames": sorted(known_frames)[:0]}


def verify(evidence: list[dict], duration: float, known_frames: set[int]) -> tuple[bool, str]:
    """Deterministic gate. Returns (ok, reason)."""
    if not evidence:
        return False, "no evidence spans"
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
            if int(f) not in known_frames:
                return False, f"frame {f} not in frame store"
    return True, "ok"


def calibrate(confidence: float, n_spans: int, n_frames: int) -> float:
    """Down-weight thin evidence; never up-weight."""
    if n_spans == 0 or n_frames == 0:
        return 0.0
    density = min(1.0, (n_spans * 2 + n_frames) / 8.0)
    return round(float(confidence) * (0.5 + 0.5 * density), 3)


def abstain(question: str) -> dict:
    return {"answer": f"Insufficient video evidence to answer: {question[:200]}",
            "evidence": [], "confidence": 0.0, "verification": "unknown"}
