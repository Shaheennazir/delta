"use client";

import { useCallback, useRef, useState } from "react";
import type { VideoSegmentationResult } from "@meta-sam/parser";
import { CoachDiagnosticCards } from "@/components/CoachDiagnosticCards";
import { DrillConfigSidebar } from "@/components/DrillConfigSidebar";
import { InteractiveStage } from "@/components/InteractiveStage";
import { ViolationTimelineBar } from "@/components/ViolationTimelineBar";
import type { SeekRequest } from "@/components/SamVideoPlayer";
import { createSession, fetchSegmentation, fetchSession, subscribeSession } from "@/lib/api";
import { resultsFromSegmentation } from "@/lib/sam";
import type { CoachSessionPayload, CoachingEvent, SessionReport } from "@/lib/types";

/** CoachingWorkspace: sidebar form + review stage + timeline + diagnostic cards (spec 7.1). */
export default function CoachingWorkspace() {
  const [videoSrc, setVideoSrc] = useState("");
  const [report, setReport] = useState<SessionReport | null>(null);
  const [samResult, setSamResult] = useState<VideoSegmentationResult | null>(null);
  const [busy, setBusy] = useState(false);
  const [liveMsg, setLiveMsg] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const [selected, setSelected] = useState<CoachingEvent | null>(null);
  const [seekRequest, setSeekRequest] = useState<SeekRequest | null>(null);
  const [nonce, setNonce] = useState(0);
  const unsubRef = useRef<(() => void) | null>(null);

  const seekTo = useCallback(
    (t: number) => {
      const n = nonce + 1;
      setNonce(n);
      setSeekRequest({ t, nonce: n });
      setCurrentTime(t);
    },
    [nonce],
  );

  const handleSelect = useCallback(
    (evt: CoachingEvent) => {
      setSelected(evt);
      seekTo(evt.timestamp_start_s);
    },
    [seekTo],
  );

  const loadSegmentation = useCallback(async (sessionId: string) => {
    try {
      const seg = await fetchSegmentation(sessionId);
      setSamResult(resultsFromSegmentation(seg.lanes));
    } catch {
      setSamResult(null); // mock path or not ready yet: events render without masks
    }
  }, []);

  const handleSubmit = useCallback(
    async (payload: CoachSessionPayload, playSrc: string) => {
      unsubRef.current?.();
      setBusy(true);
      setError(null);
      setReport(null);
      setSamResult(null);
      setSelected(null);
      setVideoSrc(playSrc);
      setLiveMsg("Queued…");
      try {
        const sessionId = await createSession(payload);
        unsubRef.current = subscribeSession(
          sessionId,
          (e) => {
            if (e.done) {
              setLiveMsg(e.status === "completed" ? null : `Session ${e.status}`);
            } else {
              const roles = e.roles ?? [];
              setLiveMsg(roles.length > 0 ? `Segmenting: ${roles.join(", ")}…` : `Status: ${e.status}…`);
              if (roles.length > 0) void loadSegmentation(sessionId);
            }
          },
          () => {},
        );
        // Poll for the final report (mock path has no lanes; SSE gives progress).
        const started = Date.now();
        for (;;) {
          const done = await fetchSession(sessionId);
          setReport(done);
          if (done.status === "completed" || done.status === "failed") break;
          if (Date.now() - started > 15 * 60_000) throw new Error("Timed out waiting for session report");
          await new Promise((r) => setTimeout(r, 1500));
        }
        await loadSegmentation(sessionId);
      } catch (err) {
        setError(err instanceof Error ? err.message : "Analysis failed");
      } finally {
        setBusy(false);
        unsubRef.current?.();
      }
    },
    [loadSegmentation],
  );

  const events = report?.events ?? [];
  const metrics = report?.summary_metrics;

  return (
    <main className="mx-auto grid max-w-7xl grid-cols-12 gap-6 p-6">
      <div className="col-span-12 lg:col-span-3">
        <DrillConfigSidebar busy={busy} status={liveMsg ?? report?.status ?? "idle"} onSubmit={handleSubmit} />
      </div>
      <div className="col-span-12 space-y-4 lg:col-span-6">
        {videoSrc ? (
          <InteractiveStage
            videoUrl={videoSrc}
            events={events}
            currentTime={currentTime}
            isolatedTrackIds={selected?.involved_track_ids ?? []}
            seekRequest={seekRequest}
            samResult={samResult}
            onTimeUpdate={setCurrentTime}
            onDuration={setDuration}
          />
        ) : (
          <div className="flex aspect-video items-center justify-center rounded-xl border border-dashed border-slate-700 text-sm text-slate-500">
            Upload a clip to start the review stage
          </div>
        )}
        <ViolationTimelineBar
          events={events}
          currentTime={currentTime}
          duration={duration}
          selectedId={selected?.event_id ?? null}
          onSeek={seekTo}
          onSelect={handleSelect}
        />
        {metrics && (
          <div className="grid grid-cols-2 gap-3 text-sm sm:grid-cols-4">
            {Object.entries(metrics).map(([k, v]) => (
              <div key={k} className="rounded-lg bg-slate-900 p-3">
                <div className="text-xs text-slate-400">{k}</div>
                <div className="text-lg font-semibold">{typeof v === "number" ? v.toFixed(2) : v}</div>
              </div>
            ))}
          </div>
        )}
        {error && <div className="rounded-lg bg-red-950 p-3 text-sm text-red-200">{error}</div>}
      </div>
      <div className="col-span-12 max-h-[75vh] space-y-4 overflow-y-auto lg:col-span-3">
        <CoachDiagnosticCards events={events} selectedId={selected?.event_id ?? null} onSelect={handleSelect} />
      </div>
    </main>
  );
}
