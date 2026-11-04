"""Strict JSON contracts (spec section 3). Pydantic v2 forbids extras: Spark never emits code."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

Primitive = Literal[
    "proximity_timer", "tether_distance", "line_intersection",
    "mask_kinematics", "mask_pose_axis",
]


class CoachSessionInput(BaseModel):
    model_config = {"extra": "forbid"}

    drill_title: str = Field(min_length=1)
    drill_setup: str = Field(min_length=1)
    analysis_goals: str = Field(min_length=1)
    video_url: str = Field(min_length=1)
    fps: float = Field(default=30.0, gt=0, le=120)


class SamConceptPrompt(BaseModel):
    model_config = {"extra": "forbid"}

    role: str = Field(min_length=1)
    prompt: str = Field(min_length=1)


class CalibrationSpec(BaseModel):
    model_config = {"extra": "forbid"}

    method: Literal["ball_diameter", "homography"] = "ball_diameter"
    reference_size_meters: float = 0.22
    fallback_grid_width_meters: float = 10.0


class MetricRule(BaseModel):
    model_config = {"extra": "forbid"}

    rule_id: str
    name: str
    primitive: Primitive
    subject_role: str | None = None
    target_role: str | None = None
    moving_role: str | None = None
    anchor_role: str | None = None
    role_a: str | None = None
    role_b: str | None = None
    ball_role: str | None = None
    contact_radius_m: float = 0.45
    max_tether_distance_m: float = 3.0
    pair_mode: Literal["nearest", "all"] = "nearest"
    kinematic_feature: Literal["speed", "acceleration", "decel"] = "speed"
    measure: str | None = None
    operator: Literal["gt", "lt", "eq"] = "gt"
    threshold: float = 1.0
    unit: str = "seconds"
    severity: Literal["info", "warning", "critical"] = "warning"


class DrillAnalysisPlan(BaseModel):
    model_config = {"extra": "forbid"}

    visual_grounding_notes: str
    sam_concept_prompts: list[SamConceptPrompt] = Field(min_length=1, max_length=8)
    calibration: CalibrationSpec = CalibrationSpec()
    metric_rules: list[MetricRule] = Field(min_length=1, max_length=4)


class HudOverlay(BaseModel):
    model_config = {"extra": "forbid"}

    type: Literal["ring_highlight", "tether_line", "label"] = "ring_highlight"
    target_track_id: int = 0
    color: str = "#EF4444"
    label: str = ""


class CoachingEvent(BaseModel):
    model_config = {"extra": "forbid"}

    event_id: str
    rule_id: str
    timestamp_start_s: float
    timestamp_end_s: float
    measured_value: float
    involved_track_ids: list[int]
    hud_overlay: HudOverlay
    coach_feedback: str = ""


class SessionReport(BaseModel):
    model_config = {"extra": "forbid"}

    session_id: str
    status: Literal["queued", "processing", "completed", "failed"] = "completed"
    summary_metrics: dict[str, float] = {}
    events: list[CoachingEvent] = []
    plan: DrillAnalysisPlan | None = None
