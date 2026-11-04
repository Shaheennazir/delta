"""Video-intelligence contracts (real-only). Pydantic v2 forbids extras on input."""
from __future__ import annotations

from pydantic import BaseModel, Field


class VideoSessionInput(BaseModel):
    model_config = {"extra": "forbid"}

    title: str = Field(min_length=1)
    focus: str = Field(min_length=1)
    video_url: str = Field(min_length=1)
    fps: float = Field(default=30.0, gt=0, le=120)


class TranscriptTurn(BaseModel):
    model_config = {"extra": "forbid"}

    turnId: int = 0
    startMs: float = 0.0
    endMs: float = 0.0
    transcript: str = ""
    speaker: str | None = None


class VideoMoment(BaseModel):
    model_config = {"extra": "forbid"}

    moment_id: str
    t_start: float
    t_end: float
    label: str = ""
    detail: str = ""


class VideoMeta(BaseModel):
    model_config = {"extra": "forbid"}

    width: int = 0
    height: int = 0
    fps: float = 30.0
    duration_s: float = 0.0


class SessionReport(BaseModel):
    model_config = {"extra": "ignore"}  # forward-compat: engine adds counts

    session_id: str
    status: str = "completed"
    title: str = ""
    answer: str = ""
    answer_evidence: list["EvidenceSpan"] = []
    answer_confidence: float = 0.0
    answer_verification: str = "unknown"
    answer_calibration: str = "uncalibrated"
    summary: str = ""
    themes: list[str] = []
    transcript: str = ""
    turns: list[TranscriptTurn] = []
    moments: list[VideoMoment] = []
    on_screen_text: str = ""
    has_audio: bool = False
    video: VideoMeta | None = None
    n_frames_indexed: int = 0
    n_entities: int = 0
    n_events: int = 0
    ablation_level: int = 1
    av_offset_s: float = 0.0
    av_sync_confidence: float = 0.0


# ---- engine contracts ----

class EvidenceSpan(BaseModel):
    model_config = {"extra": "forbid"}
    start: float
    end: float
    frame_ids: list[int] = []
    source_id: str = ""
    kind: str = ""
    verification: str = "inferred"


class AnswerClaim(BaseModel):
    model_config = {"extra": "forbid"}
    claim_id: str
    text: str
    start_t: float
    end_t: float
    confidence: float = 0.0
    verification: str = "inferred"


class GraphEvent(BaseModel):
    model_config = {"extra": "forbid"}
    event_id: str
    candidate_id: str = ""
    source: str = ""
    action: str = ""
    entity_ids: list[str] = []
    start_t: float
    end_t: float
    evidence: list[EvidenceSpan] = []
    confidence: float = 0.0
    state_before: dict = {}
    state_after: dict = {}
    detail: str = ""


class Entity(BaseModel):
    model_config = {"extra": "forbid"}
    entity_id: str
    label: str = ""
    kind: str = "object"
    confidence: float = 0.0


class Observation(BaseModel):
    model_config = {"extra": "forbid"}
    entity_id: str
    t: float
    frame_index: int = 0
    bbox: list[float] = Field(default_factory=lambda: [0, 0, 0, 0])
    track_conf: float = 0.0


class QARequest(BaseModel):
    model_config = {"extra": "forbid"}
    question: str = Field(min_length=3, max_length=1000)


class QAResponse(BaseModel):
    model_config = {"extra": "forbid"}
    qa_id: str
    question: str
    answer: str
    evidence: list[EvidenceSpan] = []
    confidence: float = 0.0
    verification: str = "unknown"
    abstained: bool = False


class QAOutcome(BaseModel):
    model_config = {"extra": "forbid"}
    correct: bool


SessionReport.model_rebuild()
