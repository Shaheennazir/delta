/** Mirrors backend schemas.py (VideoSessionInput + SessionReport + engine contracts). */
export interface TranscriptTurn {
  turnId: number;
  startMs: number;
  endMs: number;
  transcript: string;
  speaker: string | null;
}

export interface VideoMoment {
  moment_id: string;
  t_start: number;
  t_end: number;
  label: string;
  detail: string;
}

export interface SessionReport {
  session_id: string;
  status: string;
  title: string;
  answer?: string;
  answer_evidence?: EvidenceSpan[];
  answer_confidence?: number;
  answer_verification?: string;
  answer_calibration?: string;
  summary: string;
  themes: string[];
  transcript: string;
  turns: TranscriptTurn[];
  moments: VideoMoment[];
  on_screen_text: string;
  has_audio: boolean;
  video?: { width: number; height: number; fps: number; duration_s: number } | null;
  n_frames_indexed?: number;
  n_entities?: number;
  n_events?: number;
  ablation_level?: number;
  av_offset_s?: number;
  av_sync_confidence?: number;
}

export interface SessionProgress {
  session_id: string;
  stage: string;
  detail: string;
  error: string | null;
  updated_at: number;
}

export interface VideoSessionPayload {
  title: string;
  focus: string;
  video_url: string;
  fps: number;
}

export interface EvidenceSpan {
  start: number;
  end: number;
  frame_ids: number[];
  source_id?: string;
  kind?: string;
  verification?: string;
}

export interface GraphEvent {
  event_id: string;
  candidate_id?: string;
  source?: string;
  action: string;
  entity_ids: string[];
  start_t: number;
  end_t: number;
  evidence: EvidenceSpan[];
  confidence: number;
  state_before: Record<string, unknown>;
  state_after: Record<string, unknown>;
  detail: string;
}

export interface Entity {
  entity_id: string;
  label: string;
  kind: string;
  confidence: number;
}

export interface Observation {
  entity_id: string;
  t: number;
  frame_index: number;
  bbox: number[];
  track_conf: number;
}

export interface QARecord {
  qa_id: string;
  question: string;
  answer: string;
  evidence: EvidenceSpan[];
  confidence: number;
  raw_confidence?: number;
  verification: string;
  abstained?: boolean;
  calibration?: string;
}

export interface BenchmarkTest {
  test: string;
  result: "pass" | "fail" | "skip";
  detail: string;
}
