/** Mirrors backend schemas.py (SessionReport + DrillAnalysisPlan subset). */
export interface HudOverlay {
  type: "ring_highlight" | "tether_line" | "label";
  target_track_id: number;
  color: string;
  label: string;
  /** Real foot-contact anchor as media fractions; null on mock reports. */
  anchor_x: number | null;
  anchor_y: number | null;
}

export interface CoachingEvent {
  event_id: string;
  rule_id: string;
  timestamp_start_s: number;
  timestamp_end_s: number;
  measured_value: number;
  involved_track_ids: number[];
  hud_overlay: HudOverlay;
  coach_feedback: string;
}

export interface SessionReport {
  session_id: string;
  status: "queued" | "processing" | "completed" | "failed";
  summary_metrics: Record<string, number>;
  events: CoachingEvent[];
  video?: { width: number; height: number; fps: number; duration_s: number } | null;
}

export interface SegmentationPayload {
  width: number;
  height: number;
  fps: number;
  lanes: Record<string, string>;
}

export interface CoachSessionPayload {
  drill_title: string;
  drill_setup: string;
  analysis_goals: string;
  video_url: string;
  fps: number;
}

export const SEVERITY_COLOR: Record<string, string> = {
  dwell_time_check: "#F59E0B",
  defensive_split_check: "#EF4444",
};
