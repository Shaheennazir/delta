"use client";

import { useMemo, useState } from "react";
import type { VideoMoment } from "@/lib/types";

interface Props {
  moments: VideoMoment[];
  currentTime: number;
  duration: number;
  selectedId: string | null;
  onSeek: (t: number) => void;
  onSelect: (m: VideoMoment) => void;
}

export function fmt(t: number) {
  if (!isFinite(t) || t < 0) t = 0;
  const m = Math.floor(t / 60);
  const s = t - m * 60;
  return `${m}:${s.toFixed(1).padStart(4, "0")}`;
}

export function MomentTimelineBar({ moments, currentTime, duration, selectedId, onSeek, onSelect }: Props) {
  const [hover, setHover] = useState<VideoMoment | null>(null);
  const span = Math.max(duration || 0, ...moments.map((m) => m.t_end), 1);
  const pct = Math.min(100, (currentTime / span) * 100);

  const active = useMemo(
    () => moments.find((m) => currentTime >= m.t_start && currentTime <= m.t_end) ?? null,
    [moments, currentTime]
  );

  return (
    <div className="bezel">
      <div className="bezel-inner px-4 pb-3.5 pt-3">
        <div className="mb-2 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <span className="font-mono2 text-[10px] uppercase tracking-[0.22em] text-[#63615c]">Timeline</span>
            {active ? (
              <span key={active.moment_id} className="fade-in max-w-[46%] truncate rounded-full bg-[#ff3d2e]/15 px-2.5 py-1 text-[11px] text-[#ff8d82]">
                {active.label} · {fmt(active.t_start)}
              </span>
            ) : (
              <span className="text-[11px] text-[#63615c]">{moments.length} moments indexed</span>
            )}
          </div>
          <div className="font-mono2 text-[11px] tabular-nums text-[#a7a5a0]">
            {fmt(currentTime)} <span className="text-[#63615c]">/ {fmt(span)}</span>
          </div>
        </div>

        <input
          type="range"
          aria-label="Video scrubber"
          min={0}
          max={span}
          step={0.05}
          value={Math.min(currentTime, span)}
          onChange={(e) => onSeek(Number(e.target.value))}
          className="scrub w-full"
        />

        <div
          className="relative mt-1.5 h-9 rounded-xl border border-white/[0.06] bg-black/60 px-1"
          onMouseLeave={() => setHover(null)}
        >
          {/* playhead */}
          <div
            className="pointer-events-none absolute top-1 bottom-1 z-10 w-[2px] rounded bg-white shadow-[0_0_12px_rgba(255,255,255,0.8)]"
            style={{ left: `calc(${(pct).toFixed(2)}% )` }}
          >
            <div className="absolute -top-1 -left-[5px] h-[12px] w-[12px] rounded-full border-2 border-[#ff3d2e] bg-white" />
          </div>
          {/* progress wash */}
          <div
            className="pointer-events-none absolute top-1 bottom-1 left-1 rounded-lg bg-gradient-to-r from-[#ff3d2e]/25 to-[#ff3d2e]/5"
            style={{ width: `calc(${pct}% - 4px)` }}
          />
          {moments.map((m, i) => {
            const left = (m.t_start / span) * 100;
            const width = Math.max(((m.t_end - m.t_start) / span) * 100, 2.2);
            const isSel = m.moment_id === selectedId;
            const isPast = m.t_end <= currentTime;
            return (
              <button
                key={m.moment_id}
                title={`${m.label} @ ${fmt(m.t_start)}`}
                aria-label={`Seek to ${m.moment_id}`}
                onMouseEnter={() => setHover(m)}
                onFocus={() => setHover(m)}
                onClick={() => onSelect(m)}
                onDoubleClick={() => onSeek(m.t_start)}
                className="pressable group absolute top-1/2 h-6 -translate-y-1/2 overflow-hidden rounded-md"
                style={{
                  left: `${left}%`,
                  width: `${width}%`,
                  background: isSel
                    ? "#ff3d2e"
                    : isPast
                      ? "rgba(255,61,46,0.55)"
                      : "rgba(255,255,255,0.16)",
                  boxShadow: isSel ? "0 0 18px rgba(255,61,46,0.55)" : "none",
                  outline: isSel ? "1px solid rgba(255,255,255,0.7)" : "1px solid rgba(255,255,255,0.08)",
                }}
              >
                <span className="font-mono2 absolute inset-0 flex items-center justify-center text-[9px] font-semibold text-white/90">
                  {String(i + 1).padStart(2, "0")}
                </span>
              </button>
            );
          })}
          {hover && (
            <div className="fade-in pointer-events-none absolute -top-14 z-20 max-w-[240px] rounded-xl border border-white/10 bg-[#14151a]/95 px-3 py-2 shadow-2xl backdrop-blur">
              <div className="truncate text-xs font-semibold text-white">{hover.label}</div>
              <div className="font-mono2 mt-0.5 text-[10px] text-[#ff8d82]">
                {fmt(hover.t_start)} → {fmt(hover.t_end)}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
