"use client";

import { SEVERITY_COLOR, type CoachingEvent } from "@/lib/types";

interface Props {
  events: CoachingEvent[];
  selectedId: string | null;
  onSelect: (e: CoachingEvent) => void;
}

/** Clicking a card seeks the video and isolates involved_track_ids (spec 7.1). */
export function CoachDiagnosticCards({ events, selectedId, onSelect }: Props) {
  if (events.length === 0) {
    return (
      <div className="rounded-lg bg-slate-900 p-4 text-sm text-slate-400">
        No violations flagged — clean session. Submit the drill form to analyze a clip.
      </div>
    );
  }
  return (
    <div className="space-y-3">
      {events.map((evt) => {
        const border = SEVERITY_COLOR[evt.rule_id] ?? "#EF4444";
        const active = evt.event_id === selectedId;
        return (
          <button
            key={evt.event_id}
            onClick={() => onSelect(evt)}
            className={`w-full rounded-lg bg-slate-900 p-4 text-left hover:bg-slate-800 ${
              active ? "ring-2 ring-white" : ""
            } border-l-4`}
            style={{ borderLeftColor: border }}
          >
            <div className="flex justify-between text-xs text-slate-400">
              <span>{evt.rule_id}</span>
              <span>{evt.timestamp_start_s.toFixed(1)}s</span>
            </div>
            <p className="mt-2 text-sm text-slate-200">{evt.coach_feedback || evt.hud_overlay.label}</p>
          </button>
        );
      })}
    </div>
  );
}
