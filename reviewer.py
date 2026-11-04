"""Stage 4 reviewer: violation keyframes -> coaching cues (spec 5.3). Offline-safe mock."""
from __future__ import annotations

import os

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


async def review_with_spark(drill_setup: str, violations: list[dict]) -> dict[str, str]:
    """Real reviewer call; falls back to mock offline or on any error."""
    api_key = os.getenv("META_API_KEY")
    if not api_key:
        return {v["key"]: mock_feedback(v["rule_id"], v["measured"], v["threshold"],
                                        v["track_id"], v["t"]) for v in violations}
    try:
        from openai import AsyncOpenAI

        client = AsyncOpenAI(api_key=api_key, base_url="https://api.meta.ai/v1")
        out: dict[str, str] = {}
        for v in violations:
            resp = await client.chat.completions.create(
                model="muse-spark-1.3",
                messages=[
                    {"role": "system", "content": REVIEWER_SYSTEM_PROMPT},
                    {"role": "user", "content": (
                        f"Drill: {drill_setup}\nViolation: rule={v['rule_id']} "
                        f"track={v['track_id']} t={v['t']:.2f}s "
                        f"measured={v['measured']:.2f} threshold={v['threshold']:.2f}"
                    )},
                ],
            )
            out[v["key"]] = (resp.choices[0].message.content or "").strip() or mock_feedback(
                v["rule_id"], v["measured"], v["threshold"], v["track_id"], v["t"])
        return out
    except Exception:
        return {v["key"]: mock_feedback(v["rule_id"], v["measured"], v["threshold"],
                                        v["track_id"], v["t"]) for v in violations}
