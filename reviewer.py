"""Stage 4 reviewer: violation keyframes -> coaching cues (spec 5.3).

Real path sends each violation's cropped keyframe + metric to muse-spark-1.3
and returns its 2-sentence intervention verbatim. Offline or on any error it
falls back to deterministic mock feedback.
"""
from __future__ import annotations

import logging
import os

log = logging.getLogger("delta.reviewer")

REVIEWER_SYSTEM_PROMPT = """ROLE: UEFA Pro License Technical Coach.
INPUT:
1. Drill Setup & Objectives.
2. Quantitative Violation Log from the Kinematics Engine (timestamps, track IDs, measured values vs. thresholds).
3. High-resolution cropped keyframes at the exact timestamps of each violation, with offending track IDs highlighted.
TASK:
For each logged violation, inspect the player's body shape, head orientation (scanning), footwork, and spatial relationship in the keyframe crop. Write a concise, actionable 2-sentence coaching intervention:
- Sentence 1 (Diagnosis): State the exact mechanical or tactical cause visible in the frame combined with the metric.
- Sentence 2 (Coaching Cue): Provide a concrete cue the coach can shout on the pitch to fix it."""


def mock_feedback(rule_id: str, measured: float, threshold: float,
                   track_id: int, timestamp_s: float) -> str:
    if "dwell" in rule_id:
        return (
            f"At 0:{timestamp_s:04.1f}, Player #{track_id} received flat-footed with hips "
            f"closed and needed {measured:.2f}s ({threshold:.1f}s limit) to shift feet. "
            "Cue: 'Scan before the ball travels and receive on the back foot.'"
        )
    if "split" in rule_id:
        return (
            f"At 0:{timestamp_s:04.1f}, the pass split both defenders who pressed flat "
            "without staggering cover. "
            "Cue: 'Stagger — one presses, one covers the lane!'"
        )
    return (
        f"At 0:{timestamp_s:04.1f}, Player #{track_id} breached {rule_id} "
        f"({measured:.2f} vs limit {threshold:.2f}). "
        f"Cue: 'Reset shape and re-do the rep at full tempo.'"
    )


async def review_with_spark(
    drill_setup: str,
    violations: list[dict],
    crops: dict[str, bytes] | None = None,
) -> dict[str, str]:
    """Real reviewer call with keyframe crops; falls back to mock per violation."""
    has_key = bool(os.getenv("META_API_KEY") or os.getenv("MODEL_API_KEY") or os.getenv("SAM_API_KEY"))
    if not has_key:
        return {v["key"]: mock_feedback(v["rule_id"], v["measured"], v["threshold"],
                                        v["track_id"], v["t"]) for v in violations}
    from spark import SPARK_MODEL, jpeg_data_url, post_unary, extract_text

    out: dict[str, str] = {}
    for v in violations:
        try:
            content: list[dict] = []
            jpeg = (crops or {}).get(v["key"])
            if jpeg:
                content.append({"type": "input_image", "image_url": jpeg_data_url(jpeg)})
            content.append({"type": "input_text", "text": (
                f"Drill: {drill_setup}\nViolation: rule={v['rule_id']} "
                f"track={v['track_id']} t={v['t']:.2f}s "
                f"measured={v['measured']:.2f} threshold={v['threshold']:.2f}"
            )})
            body = await post_unary({
                "model": SPARK_MODEL,
                "instructions": REVIEWER_SYSTEM_PROMPT,
                "input": [{"type": "message", "role": "user", "content": content}],
            })
            text = extract_text(body).strip()
            out[v["key"]] = text or mock_feedback(
                v["rule_id"], v["measured"], v["threshold"], v["track_id"], v["t"])
        except Exception as e:
            log.warning("spark reviewer failed for %s, using mock: %s", v["key"], e)
            out[v["key"]] = mock_feedback(
                v["rule_id"], v["measured"], v["threshold"], v["track_id"], v["t"])
    return out
