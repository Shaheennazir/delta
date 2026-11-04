"""Stage 2 planner: muse-spark-1.3 Frame-0 grounding -> DrillAnalysisPlan (spec 5.2).

Offline-safe: without META_API_KEY returns a deterministic mock plan derived
from analysis_goals keywords, capped at 4 rules using only the 5 primitives.
"""
from __future__ import annotations

import os

from schemas import (
    CalibrationSpec,
    CoachSessionInput,
    DrillAnalysisPlan,
    MetricRule,
    SamConceptPrompt,
)

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


async def plan_with_spark(inp: CoachSessionInput) -> DrillAnalysisPlan:
    """Real muse-spark-1.3 call; falls back to mock offline or on any error."""
    api_key = os.getenv("META_API_KEY")
    if not api_key:
        return mock_plan(inp)
    try:
        from openai import AsyncOpenAI

        client = AsyncOpenAI(api_key=api_key, base_url="https://api.meta.ai/v1")
        resp = await client.chat.completions.create(
            model="muse-spark-1.3",
            messages=[
                {"role": "system", "content": PLANNER_SYSTEM_PROMPT},
                {"role": "user", "content": (
                    f"Drill Setup: {inp.drill_setup}\nObjectives: {inp.drill_title}\n"
                    f"Analysis Request: {inp.analysis_goals}\n"
                    "(Frame 0 unavailable in CLI mode; infer generic bib colors.)"
                )},
            ],
            response_format={"type": "json_object"},
        )
        return DrillAnalysisPlan.model_validate_json(resp.choices[0].message.content or "{}")
    except Exception:
        return mock_plan(inp)
