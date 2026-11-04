"use client";

import { SEVERITY_COLOR, type CoachingEvent } from "@/lib/types";

interface Props {
  events: CoachingEvent[];
  currentTime: number;
  duration: number;
  selectedId: string | null;
  onSeek: (t: number) => void;
  onSelect: (e: CoachingEvent) => void;
}

export function ViolationTimelineBar({ events, currentTime, duration, selectedId, onSeek, onSelect }: Props) {
  const span = Math.max(duration, ...events.map((e) => e.timestamp_end_s), 1);
  return (
    <div className="rounded-xl border border-slate-800 bg-slate-900 p-4">
      <input
        type="range"
        aria-label="Video scrubber"
        min={0}
        max={span}
        step={0.05}
        value={Math.min(currentTime, span)}
        onChange={(e) => onSeek(Number(e.target.value))}
        className="w-full accent-red-500"
      />
      <div className="relative mt-2 h-2 rounded bg-slate-800">
        {events.map((e) => {
          const left = (e.timestamp_start_s / span) * 100;
          const width = Math.max(((e.timestamp_end_s - e.timestamp_start_s) / span) * 100, 1.2);
          const color = SEVERITY_COLOR[e.rule_id] ?? "#EF4444";
          return (
            <button
              key={e.event_id}
              title={`${e.rule_id} @ ${e.timestamp_start_s.toFixed(1)}s`}
              aria-label={`Seek to ${e.event_id}`}
              onClick={() => onSelect(e)}
              className="absolute top-0 h-full rounded"
              style={{
                left: `${left}%`,
                width: `${width}%`,
                background: color,
                outline: e.event_id === selectedId ? "2px solid #fff" : "none",
              }}
            />
          );
        })}
      </div>
    </div>
  );
}
