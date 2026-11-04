"use client";

import type { VideoMoment } from "@/lib/types";

interface Props {
  moments: VideoMoment[];
  selectedId: string | null;
  onSelect: (m: VideoMoment) => void;
}

export function MomentCards({ moments, selectedId, onSelect }: Props) {
  if (moments.length === 0) {
    return (
      <div className="rounded-lg bg-slate-900 p-4 text-sm text-slate-400">
        No moments yet — submit a video to get grounded highlights.
      </div>
    );
  }
  return (
    <div className="space-y-3">
      {moments.map((m) => (
        <button key={m.moment_id} onClick={() => onSelect(m)}
          className={`w-full rounded-lg bg-slate-900 p-4 text-left hover:bg-slate-800 ${m.moment_id === selectedId ? "ring-2 ring-white" : ""} border-l-4 border-red-500`}>
          <div className="flex justify-between text-xs text-slate-400">
            <span>{m.label}</span>
            <span>{m.t_start.toFixed(1)}s → {m.t_end.toFixed(1)}s</span>
          </div>
          <p className="mt-2 text-sm text-slate-200">{m.detail}</p>
        </button>
      ))}
    </div>
  );
}
