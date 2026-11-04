"use client";

import type { VideoMoment } from "@/lib/types";
import { fmt } from "./MomentTimelineBar";

interface Props {
  moments: VideoMoment[];
  selectedId: string | null;
  currentTime: number;
  onSelect: (m: VideoMoment) => void;
}

export function MomentCards({ moments, selectedId, currentTime, onSelect }: Props) {
  if (moments.length === 0) {
    return (
      <div className="rounded-2xl border border-dashed border-white/10 bg-white/[0.02] p-5 text-center text-[13px] text-[#63615c]">
        No moments yet — they appear here once the pipeline grounds them.
      </div>
    );
  }
  return (
    <ol className="space-y-2">
      {moments.map((m, i) => {
        const live = currentTime >= m.t_start && currentTime <= m.t_end;
        const sel = m.moment_id === selectedId;
        return (
          <li key={m.moment_id} style={{ animationDelay: `${Math.min(i * 60, 420)}ms` }} className="rise">
            <button
              onClick={() => onSelect(m)}
              className={`pressable lift group w-full rounded-2xl border p-3.5 text-left ${
                sel
                  ? "border-[#ff3d2e]/60 bg-[#ff3d2e]/[0.08] shadow-[0_0_30px_rgba(255,61,46,0.18)]"
                  : live
                    ? "border-white/15 bg-white/[0.05]"
                    : "border-white/[0.07] bg-white/[0.025] hover:border-white/15 hover:bg-white/[0.05]"
              }`}
            >
              <div className="flex items-center gap-2.5">
                <span
                  className={`font-mono2 flex h-7 w-7 shrink-0 items-center justify-center rounded-lg text-[11px] font-semibold ${
                    sel ? "bg-[#ff3d2e] text-white" : "bg-white/10 text-[#a7a5a0]"
                  }`}
                >
                  {String(i + 1).padStart(2, "0")}
                </span>
                <span className="min-w-0 flex-1 truncate text-[13px] font-semibold text-[#f4f1ea]">{m.label}</span>
                {live && <span className="live-dot h-1.5 w-1.5 shrink-0 rounded-full bg-[#ff3d2e]" />}
              </div>
              <p className="mt-1.5 line-clamp-2 text-[12.5px] leading-relaxed text-[#a7a5a0]">{m.detail}</p>
              <div className="font-mono2 mt-2 flex items-center gap-2 text-[10.5px] text-[#63615c]">
                <span className="rounded-full bg-white/[0.06] px-2 py-0.5 tabular-nums text-[#ff8d82]">
                  {fmt(m.t_start)} → {fmt(m.t_end)}
                </span>
                <span className="opacity-0 transition-opacity duration-200 group-hover:opacity-100">seek →</span>
              </div>
            </button>
          </li>
        );
      })}
    </ol>
  );
}
