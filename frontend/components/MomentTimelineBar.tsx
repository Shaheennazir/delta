"use client";

import type { VideoMoment } from "@/lib/types";

interface Props {
  moments: VideoMoment[];
  currentTime: number;
  duration: number;
  selectedId: string | null;
  onSeek: (t: number) => void;
  onSelect: (m: VideoMoment) => void;
}

export function MomentTimelineBar({ moments, currentTime, duration, selectedId, onSeek, onSelect }: Props) {
  const span = Math.max(duration, ...moments.map((m) => m.t_end), 1);
  return (
    <div className="rounded-xl border border-slate-800 bg-slate-900 p-4">
      <input type="range" aria-label="Video scrubber" min={0} max={span} step={0.05}
        value={Math.min(currentTime, span)} onChange={(e) => onSeek(Number(e.target.value))}
        className="w-full accent-red-500" />
      <div className="relative mt-2 h-2 rounded bg-slate-800">
        {moments.map((m) => {
          const left = (m.t_start / span) * 100;
          const width = Math.max(((m.t_end - m.t_start) / span) * 100, 1.2);
          return (
            <button key={m.moment_id} title={`${m.label} @ ${m.t_start.toFixed(1)}s`}
              aria-label={`Seek to ${m.moment_id}`} onClick={() => onSelect(m)}
              className="absolute top-0 h-full rounded bg-red-500"
              style={{ left: `${left}%`, width: `${width}%`,
                outline: m.moment_id === selectedId ? "2px solid #fff" : "none" }} />
          );
        })}
      </div>
    </div>
  );
}
