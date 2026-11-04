"""Stage 2 planner: muse-spark-1.3 Frame-0 grounding -> DrillAnalysisPlan (spec 5.2).

Real path sends Frame 0 (when local bytes exist) + drill text with a
json_schema response format, validated by DrillAnalysisPlan. Offline or on any
error it falls back to the deterministic mock plan.
"""
from __future__ import annotations

import logging
import os

from schemas import (
    CalibrationSpec,
    CoachSessionInput,
    DrillAnalysisPlan,
    MetricRule,
    SamConceptPrompt,
)

log = logging.getLogger("delta.planner")

PLANNER_SYSTEM_PROMPT = """ROLE: Football Computer Vision Architect.
INPUT:
1. Frame 0 image of a football training drill.
2. Coach's Drill Setup & Objectives.
3. Coach's Analysis Request.
TASK:
1. Inspect Frame 0 carefully. Identify the distinct visual clothing/attributes of each player group, the ball, and training markers.
2. Formulate `sam_concept_prompts`: Each prompt MUST be a short, concrete, singular noun phrase describing visual appearance (e.g., "soccer ball", "player in yellow vest", "red cone"). NEVER include verbs, numbers, or multiple concepts in one prompt.
3. Translate the Coach's Analysis Request into 1 to 4 deterministic `metric_rules` using ONLY the 5 supported primitives: `proximity_timer`, `tether_distance`, `line_intersection`, `mask_kinematics`, `mask_pose_axis`.
4. Output strictly adhering to the `DrillAnalysisPlan` JSON schema."""


def mock_plan(inp: CoachSessionInput) -> DrillAnalysisPlan:
    goals = inp.analysis_goals.lower()
    rules: list[MetricRule] = []
    if any(k in goals for k in ("dwell", "hold", "touch", "second")):
        rules.append(MetricRule(
            rule_id="dwell_time_check", name="Ball Dwell Time",
            primitive="proximity_timer", subject_role="attacker",
            target_role="ball", contact_radius_m=0.45,
            operator="gt", threshold=1.0, unit="seconds", severity="warning",
        ))
    if any(k in goals for k in ("split", "line", "pass")):
        rules.append(MetricRule(
            rule_id="defensive_split_check", name="Defenders Split by Pass",
            primitive="line_intersection", moving_role="ball",
            anchor_role="defender", max_tether_distance_m=3.0,
            operator="eq", threshold=1.0, unit="boolean", severity="critical",
        ))
    if any(k in goals for k in ("speed", "sprint", "decel", "stance")):
        rules.append(MetricRule(
            rule_id="sprint_check", name="Sprint Speed",
            primitive="mask_kinematics", target_role="attacker",
            kinematic_feature="speed", operator="gt", threshold=5.0,
            unit="m/s", severity="info",
        ))
    if any(k in goals for k in ("lean", "posture", "plant", "strike", "shoot")):
        rules.append(MetricRule(
            rule_id="strike_posture_check", name="Strike Posture",
            primitive="mask_pose_axis", subject_role="attacker",
            ball_role="ball", measure="torso_lean_deg",
            operator="gt", threshold=20.0, unit="degrees", severity="warning",
        ))
    if not rules:  # default: smallest useful plan
        rules.append(MetricRule(
            rule_id="dwell_time_check", name="Ball Dwell Time",
            primitive="proximity_timer", subject_role="attacker",
            target_role="ball", contact_radius_m=0.45,
            operator="gt", threshold=1.0, unit="seconds", severity="warning",
        ))
    return DrillAnalysisPlan(
        visual_grounding_notes="Mock grounding: attackers in yellow bibs, defenders in blue, ball, orange cones.",
        sam_concept_prompts=[
            SamConceptPrompt(role="ball", prompt="soccer ball"),
            SamConceptPrompt(role="attacker", prompt="player in yellow bib"),
            SamConceptPrompt(role="defender", prompt="player in blue shirt"),
            SamConceptPrompt(role="marker", prompt="orange training cone"),
        ],
        calibration=CalibrationSpec(),
        metric_rules=rules[:4],
    )


async def plan_with_spark(inp: CoachSessionInput, frame0_jpeg: bytes | None = None) -> DrillAnalysisPlan:
    """Real muse-spark-1.3 call with Frame-0 vision; falls back to mock."""
    if not (os.getenv("META_API_KEY") or os.getenv("MODEL_API_KEY") or os.getenv("SAM_API_KEY")):
        return mock_plan(inp)
    try:
        from spark import SPARK_MODEL, jpeg_data_url, post_unary, extract_text

        content: list[dict] = []
        if frame0_jpeg:
            content.append({"type": "input_image", "image_url": jpeg_data_url(frame0_jpeg)})
        content.append({"type": "input_text", "text": (
            f"Drill Setup: {inp.drill_setup}\nObjectives: {inp.drill_title}\n"
            f"Analysis Request: {inp.analysis_goals}\n"
            + ("The attached image is Frame 0 of the drill clip." if frame0_jpeg else
               "(No frame image available; infer generic bib colors.)")
        )})
        body = await post_unary({
            "model": SPARK_MODEL,
            "instructions": PLANNER_SYSTEM_PROMPT,
            "input": [{"type": "message", "role": "user", "content": content}],
            "text": {"format": {
                "type": "json_schema", "name": "DrillAnalysisPlan",
                "schema": DrillAnalysisPlan.model_json_schema(), "strict": True,
            }},
        })
        plan = DrillAnalysisPlan.model_validate_json(extract_text(body))
        log.info("spark planner ok: %d rules", len(plan.metric_rules))
        return plan
    except Exception as e:
        log.warning("spark planner failed, using mock: %s", e)
        return mock_plan(inp)
