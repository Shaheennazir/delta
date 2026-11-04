"use client";

import type { VideoSegmentationResult } from "@meta-sam/parser";
import { SamVideoPlayer, type SeekRequest } from "./SamVideoPlayer";
import { TacticalSvgOverlay } from "./TacticalSvgOverlay";
import type { CoachingEvent } from "@/lib/types";

interface Props {
  videoUrl: string;
  events: CoachingEvent[];
  currentTime: number;
  isolatedTrackIds: number[];
  seekRequest: SeekRequest | null;
  samResult?: VideoSegmentationResult | null;
  onTimeUpdate: (t: number) => void;
  onDuration: (d: number) => void;
}

/** Relative container locking video + overlay layers (spec 7.1). */
export function InteractiveStage({
  videoUrl,
  events,
  currentTime,
  isolatedTrackIds,
  seekRequest,
  samResult,
  onTimeUpdate,
  onDuration,
}: Props) {
  return (
    <div className="relative aspect-video w-full overflow-hidden rounded-xl border border-slate-800 bg-black">
      <SamVideoPlayer
        src={videoUrl}
        result={samResult}
        hiddenIds={[]}
        seekRequest={seekRequest}
        onTimeUpdate={onTimeUpdate}
        onDuration={onDuration}
      />
      <TacticalSvgOverlay events={events} currentTime={currentTime} isolatedTrackIds={isolatedTrackIds} />
    </div>
  );
}
