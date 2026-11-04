"use client";

import type { CoachingEvent } from "@/lib/types";

/**
 * Tactical SVG layer synced to video.currentTime. Renders ring highlights for
 * events active at `currentTime` and tether lines for split-pass events.
 *
 * Positions are deterministic placeholders derived from track id until the
 * backend streams per-frame mask geometry (foot contact points) — the overlay
 * contract (event -> anchor point) already matches that future shape.
 */
interface Props {
  events: CoachingEvent[];
  currentTime: number;
  isolatedTrackIds: number[];
}

function anchorFor(trackId: number, evt?: CoachingEvent): { x: number; y: number } {
  // Real foot-contact anchor from the backend (media fractions) when present.
  if (evt?.hud_overlay.anchor_x != null && evt?.hud_overlay.anchor_y != null) {
    return { x: evt.hud_overlay.anchor_x * 1000, y: evt.hud_overlay.anchor_y * 1000 };
  }
  // Deterministic placeholder for mock/demo reports without mask geometry.
  const x = 200 + ((trackId * 137) % 600);
  const y = 250 + ((trackId * 251) % 500);
  return { x, y };
}

export function TacticalSvgOverlay({ events, currentTime, isolatedTrackIds }: Props) {
  const active = events.filter(
    (e) => currentTime >= e.timestamp_start_s && currentTime <= Math.max(e.timestamp_end_s, e.timestamp_start_s + 0.5),
  );
  const isolated = new Set(isolatedTrackIds);

  return (
    <svg viewBox="0 0 1000 1000" preserveAspectRatio="xMidYMid meet" className="pointer-events-none absolute inset-0 h-full w-full">
      {active.map((e) => {
        const dim = isolatedTrackIds.length > 0 && !e.involved_track_ids.some((id) => isolated.has(id));
        const p = anchorFor(e.hud_overlay.target_track_id, e);
        return (
          <g key={e.event_id} opacity={dim ? 0.15 : 1}>
            {e.rule_id.includes("split") && e.involved_track_ids.length >= 2 ? (
              <line
                x1={anchorFor(e.involved_track_ids[1]).x}
                y1={anchorFor(e.involved_track_ids[1]).y}
                x2={anchorFor(e.involved_track_ids[2] ?? e.involved_track_ids[1]).x}
                y2={anchorFor(e.involved_track_ids[2] ?? e.involved_track_ids[1]).y}
                stroke={e.hud_overlay.color}
                strokeWidth={6}
                strokeDasharray="18 12"
              />
            ) : null}
            <circle cx={p.x} cy={p.y} r={54} fill="none" stroke={e.hud_overlay.color} strokeWidth={7} />
            <circle cx={p.x} cy={p.y} r={8} fill={e.hud_overlay.color} />
            <text x={p.x} y={p.y - 70} textAnchor="middle" fill="#fff" fontSize={34} fontWeight={700}>
              {e.hud_overlay.label}
            </text>
          </g>
        );
      })}
    </svg>
  );
}
