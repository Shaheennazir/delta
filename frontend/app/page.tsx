"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { MomentCards } from "@/components/MomentCards";
import { MomentTimelineBar } from "@/components/MomentTimelineBar";
import { BenchmarkPanel, EntityList, EventList, QaPanel, TracePanel } from "@/components/EnginePanels";
import { VideoConfigSidebar } from "@/components/VideoConfigSidebar";
import { createSession, fetchEntities, fetchEvents, fetchProgress, fetchSession, fetchTrace, subscribeSession, type TraceRow } from "@/lib/api";
import type { Entity, GraphEvent, SessionProgress, SessionReport, VideoMoment, VideoSessionPayload } from "@/lib/types";

const STAGES = ["queued", "uploading", "preparing", "observing", "perceiving", "events", "understanding", "verifying", "completed"] as const;

function stageIndex(s: string | undefined): number {
  const i = STAGES.indexOf(s as (typeof STAGES)[number]);
  return i < 0 ? -1 : i;
}

export default function VideoWorkspace() {
  const [videoSrc, setVideoSrc] = useState("");
  const [report, setReport] = useState<SessionReport | null>(null);
  const [busy, setBusy] = useState(false);
  const [progress, setProgress] = useState<SessionProgress | null>(null);
  const [elapsed, setElapsed] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const [selected, setSelected] = useState<VideoMoment | null>(null);
  const [sessionId, setSessionId] = useState("");
  const [events, setEvents] = useState<GraphEvent[]>([]);
  const [entities, setEntities] = useState<Entity[]>([]);
  const [trace, setTrace] = useState<TraceRow[]>([]);
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const unsubRef = useRef<(() => void) | null>(null);
  const startedRef = useRef(0);

  useEffect(() => {
    if (!busy) return;
    const t = setInterval(() => setElapsed(Math.floor((Date.now() - startedRef.current) / 1000)), 500);
    return () => clearInterval(t);
  }, [busy]);

  const seekTo = useCallback((t: number) => {
    const v = videoRef.current;
    if (v) { try { v.currentTime = t; void v.pause(); } catch {} }
    setCurrentTime(t);
  }, []);

  const handleSelect = useCallback((m: VideoMoment) => {
    setSelected(m);
    seekTo(m.t_start);
  }, [seekTo]);

  const handleSubmit = useCallback(async (payload: VideoSessionPayload, playSrc: string) => {
    unsubRef.current?.();
    setBusy(true);
    setError(null);
    setReport(null);
    setSelected(null);
    setTrace([]);
    setVideoSrc(playSrc);
    setProgress({ session_id: "", stage: "queued", detail: "Queued — pipeline starting…", error: null, updated_at: Date.now() / 1000 });
    setElapsed(0);
    startedRef.current = Date.now();
    try {
      const sessionId = await createSession(payload);
      unsubRef.current = subscribeSession(sessionId, (e) => {
        if (e.stage || e.detail) {
          setProgress((p) => ({ session_id: sessionId, stage: e.stage ?? p?.stage ?? "processing",
            detail: e.detail ?? "", error: e.error ?? null, updated_at: Date.now() / 1000 }));
        }
        if (Array.isArray((e as { trace?: TraceRow[] }).trace)) setTrace((e as { trace?: TraceRow[] }).trace as TraceRow[]);
      }, () => {});
      const started = Date.now();
      for (;;) {
        const [done, prog] = await Promise.all([fetchSession(sessionId), fetchProgress(sessionId).catch(() => null)]);
        if (prog) setProgress(prog);
        fetchTrace(sessionId).then(setTrace).catch(() => {});
        if (done.status === "completed" || done.status === "failed") {
          setReport(done);
          if (done.status === "completed") {
            setSessionId(sessionId);
            fetchEvents(sessionId).then(setEvents).catch(() => {});
            fetchEntities(sessionId).then((r) => setEntities(r.entities)).catch(() => {});
          }
          if (done.status === "failed") setError(prog?.error ? `Analysis failed: ${prog.error}` : "Analysis failed.");
          break;
        }
        if (Date.now() - started > 15 * 60_000) throw new Error("Timed out (15 min). Try a shorter clip.");
        await new Promise((r) => setTimeout(r, 1500));
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Analysis failed");
    } finally {
      setBusy(false);
      unsubRef.current?.();
    }
  }, []);

  const moments = report?.moments ?? [];
  const showProgress = busy || (progress && progress.stage !== "completed" && !report);

  return (
    <main className="mx-auto grid max-w-7xl grid-cols-12 gap-6 p-6">
      <div className="col-span-12 lg:col-span-3">
        <VideoConfigSidebar busy={busy} status={progress?.stage ?? "idle"} onSubmit={handleSubmit} />
      </div>
      <div className="col-span-12 space-y-4 lg:col-span-6">
        {showProgress && progress && (
          <div className="rounded-xl border border-slate-800 bg-slate-900 p-4 text-sm" aria-live="polite">
            <div className="flex items-center justify-between">
              <span className="font-semibold text-slate-200">Analysis: {progress.stage}</span>
              <span className="text-xs text-slate-400">{elapsed}s elapsed</span>
            </div>
            {progress.detail && <div className="mt-1 text-xs text-slate-400">{progress.detail}</div>}
            <ol className="mt-3 flex flex-wrap gap-1.5">
              {STAGES.map((s) => {
                const cur = stageIndex(progress.stage);
                const mine = stageIndex(s);
                const done = mine < cur || progress.stage === "completed";
                const active = mine === cur && progress.stage !== "completed";
                return (
                  <li key={s} className={`rounded-full px-2.5 py-1 text-[11px] ${
                    done ? "bg-emerald-900 text-emerald-200" : active ? "bg-red-600 text-white" : "bg-slate-800 text-slate-500"}`}>
                    {done ? "✓ " : ""}{s}
                  </li>
                );
              })}
            </ol>
            <div className="mt-2 text-[11px] text-slate-500">Agent runs transcribe + SAM + OCR + audio events concurrently. Each tool reports sent → waiting → received.</div>
          </div>
        )}
        {trace.length > 0 && <TracePanel trace={trace} />}
        {videoSrc ? (
          <video ref={videoRef} src={videoSrc} controls preload="metadata"
            className="aspect-video w-full rounded-xl border border-slate-800 bg-black"
            onTimeUpdate={(e) => setCurrentTime(e.currentTarget.currentTime)}
            onLoadedMetadata={(e) => setDuration(e.currentTarget.duration)} />
        ) : (
          <div className="flex aspect-video items-center justify-center rounded-xl border border-dashed border-slate-700 text-sm text-slate-500">
            Upload a video to start
          </div>
        )}
        <MomentTimelineBar moments={moments} currentTime={currentTime} duration={duration || report?.video?.duration_s || 0}
          selectedId={selected?.moment_id ?? null} onSeek={seekTo} onSelect={handleSelect} />
        {report?.status === "completed" && (
          <div className="space-y-3">
            {report.answer && (
              <div className="rounded-xl border border-red-900/50 bg-slate-900 p-4">
                <div className="text-xs uppercase tracking-wider text-red-300">Answer</div>
                <p className="mt-1 text-sm text-slate-100">{report.answer}</p>
                <div className="mt-2 text-xs text-slate-400">
                  {(report.answer_confidence ?? 0).toFixed(2)} · {report.answer_verification ?? "unknown"} ·{" "}
                  {report.answer_calibration ?? "uncalibrated"}
                  {(report.answer_evidence ?? []).length > 0 && (
                    <> · {report.answer_evidence!.map((s) => `${s.start.toFixed(1)}–${s.end.toFixed(1)}s`).join(", ")}</>
                  )}
                </div>
              </div>
            )}
            {report.summary && (
              <div className="rounded-xl bg-slate-900 p-4">
                <div className="text-xs uppercase tracking-wider text-slate-400">Summary</div>
                <p className="mt-1 text-sm text-slate-100">{report.summary}</p>
              </div>
            )}
            {report.themes.length > 0 && (
              <div className="flex flex-wrap gap-1.5">
                {report.themes.map((t) => (
                  <span key={t} className="rounded-full bg-slate-800 px-3 py-1 text-xs text-slate-200">{t}</span>
                ))}
              </div>
            )}
            {report.on_screen_text && (
              <div className="rounded-xl bg-slate-900 p-4 text-sm">
                <div className="text-xs uppercase tracking-wider text-slate-400">On-screen text</div>
                <p className="mt-1 text-slate-200">{report.on_screen_text}</p>
              </div>
            )}
            <div className="rounded-xl bg-slate-900 p-4 text-sm">
              <div className="text-xs uppercase tracking-wider text-slate-400">
                Transcript {report.has_audio ? "" : "(no audio detected)"}
              </div>
              <p className="mt-1 whitespace-pre-wrap text-slate-200">{report.transcript || "—"}</p>
              {report.turns.length > 0 && (
                <div className="mt-2 space-y-1 text-xs text-slate-400">
                  {report.turns.slice(0, 20).map((t) => (
                    <div key={t.turnId}>[{(t.startMs / 1000).toFixed(1)}s{ t.speaker ? ` spk ${t.speaker}` : ""}] {t.transcript}</div>
                  ))}
                </div>
              )}
            </div>
            {(report.n_frames_indexed || report.n_entities || report.n_events) && (
              <div className="rounded-xl bg-slate-900 p-4 text-xs text-slate-400">
                Engine: {report.n_frames_indexed ?? 0} frames indexed · {report.n_entities ?? 0} entities ·{" "}
                {report.n_events ?? 0} events · ablation level {report.ablation_level ?? 1}/4
              </div>
            )}
            {entities.length > 0 && <EntityList entities={entities} />}
            {sessionId && <QaPanel sessionId={sessionId} onSeek={seekTo} />}
            {sessionId && <BenchmarkPanel sessionId={sessionId} />}
          </div>
        )}
        {report?.status === "completed" && moments.length === 0 && !busy && (
          <div className="rounded-lg bg-slate-900 p-3 text-sm text-slate-400">Completed — no grounded moments returned.</div>
        )}
        {error && <div className="rounded-lg bg-red-950 p-3 text-sm text-red-200">{error}</div>}
      </div>
      <div className="col-span-12 max-h-[75vh] space-y-4 overflow-y-auto lg:col-span-3">
        {busy && !report ? (
          <div className="rounded-lg bg-slate-900 p-4 text-sm text-slate-400">Analyzing… moments appear here.</div>
        ) : (
          <>
            {events.length > 0 && (
              <div className="space-y-2">
                <div className="text-xs uppercase tracking-wider text-slate-400">Event graph</div>
                <EventList events={events} onSeek={seekTo} />
              </div>
            )}
            <MomentCards moments={moments} selectedId={selected?.moment_id ?? null} onSelect={handleSelect} />
          </>
        )}
      </div>
    </main>
  );
}
