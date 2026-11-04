"use client";

import { useEffect, useRef } from "react";
import { Video, type VideoRef } from "@meta-sam/react";
import type { VideoSegmentationResult } from "@meta-sam/parser";

export interface SeekRequest {
  t: number;
  nonce: number;
}

interface Props {
  src: string;
  result?: VideoSegmentationResult | null;
  hiddenIds?: string[];
  seekRequest: SeekRequest | null;
  onTimeUpdate?: (t: number) => void;
  onDuration?: (d: number) => void;
}

/**
 * Canvas-only SAM video player. The browser replays the uploaded clip while
 * the parsed segmentation result (fetched from /segmentation) drives the
 * mask overlays — zero backend video re-encoding.
 */
export function SamVideoPlayer({ src, result, hiddenIds, seekRequest, onTimeUpdate, onDuration }: Props) {
  const ref = useRef<VideoRef>(null);
  const nonceRef = useRef(0);

  useEffect(() => {
    if (!seekRequest || seekRequest.nonce === nonceRef.current) return;
    nonceRef.current = seekRequest.nonce;
    const player = ref.current;
    if (player) {
      void player.seek(seekRequest.t).then(() => player.pause()).catch(() => {});
    }
  }, [seekRequest]);

  return (
    <Video
      ref={ref}
      src={src}
      result={result ?? undefined}
      hiddenIds={hiddenIds ?? []}
      objectFit="contain"
      onTimeChange={onTimeUpdate}
      onDurationChange={onDuration}
      onError={(e) => console.error("Video failed", e)}
      className="h-full w-full"
    />
  );
}
