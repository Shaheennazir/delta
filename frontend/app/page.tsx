"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { MomentTimelineBar } from "@/components/MomentTimelineBar";
import { EntityList, EventList, TracePanel } from "@/components/EnginePanels";
import {
  createSession, fetchEntities, fetchEvents, fetchProgress, fetchSession, fetchTrace,
  listQa, listSessions, sendChat, subscribeSession, uploadClip, uploadUrlToHttp,
  type SessionSummary, type TraceRow,
} from "@/lib/api";
import type { Entity, EvidenceSpan, GraphEvent, SessionProgress, SessionReport, VideoMoment, VideoSessionPayload } from "@/lib/types";

type Msg = {
  id: string;
  role: "user" | "assistant" | "status";
  text: string;
  evidence?: EvidenceSpan[];
  confidence?: number;
  verification?: string;
  route?: string;
};

const STAGES = ["queued", "uploading", "preparing", "observing", "perceiving", "events", "understanding", "verifying", "completed"] as const;
const ORDER = STAGES.filter((s) => s !== "completed");
const PRESETS = [
  "Summarize what happens and the key takeaways",
  "Who is in the video and what do they do?",
  "List the key moments with timestamps",
  "Transcribe what is said and shown on screen",
];
const LS_KEY = "delta-session-id";

let _id = 0;
const nid = () => `m${Date.now().toString(36)}${(_id++).toString(36)}`;
const ago = (t: number) => {
  const s = Math.max(0, Math.floor(Date.now() / 1000 - t));
  if (s < 60) return "just now";
  if (s < 3600) return `${Math.floor(s / 60)}m ago`;
  if (s < 86400) return `${Math.floor(s / 3600)}h ago`;
  return `${Math.floor(s / 86400)}d ago`;
};

export default function VideoChat() {
  const [videoSrc, setVideoSrc] = useState("");
  const [sessionId, setSessionId] = useState("");
  const [sessions, setSessions] = useState<SessionSummary[]>([]);
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [report, setReport] = useState<SessionReport | null>(null);
  const [msgs, setMsgs] = useState<Msg[]>([]);
  const [input, setInput] = useState("");
  const [focus, setFocus] = useState(PRESETS[0]);
  const [fileName, setFileName] = useState("");
  const [videoUrl, setVideoUrl] = useState("");
  const [uploading, setUploading] = useState(false);
  const [busy, setBusy] = useState(false);
  const [answering, setAnswering] = useState(false);
  const [progress, setProgress] = useState<SessionProgress | null>(null);
  const [elapsed, setElapsed] = useState(0);
  const [trace, setTrace] = useState<TraceRow[]>([]);
  const [events, setEvents] = useState<GraphEvent[]>([]);
  const [entities, setEntities] = useState<Entity[]>([]);
  const [showEngine, setShowEngine] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const [error, setError] = useState<string | null>(null);

  const videoRef = useRef<HTMLVideoElement | null>(null);
  const fileRef = useRef<HTMLInputElement | null>(null);
  const bottomRef = useRef<HTMLDivElement | null>(null);
  const startedRef = useRef(0);
  const unsubRef = useRef<(() => void) | null>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [msgs, progress?.detail]);

  useEffect(() => {
    if (!busy) return;
    const t = setInterval(() => setElapsed(Math.floor((Date.now() - startedRef.current) / 1000)), 500);
    return () => clearInterval(t);
  }, [busy]);

  const refreshSessions = useCallback(async () => {
    try { setSessions(await listSessions()); } catch {}
  }, []);

  const seekTo = useCallback((t: number) => {
    const v = videoRef.current;
    if (v) { try { v.currentTime = t; v.pause(); } catch {} }
    setCurrentTime(t);
  }, []);

  const push = useCallback((m: Msg) => setMsgs((p) => [...p, m]), []);

  /** Poll a running session until completed/failed; shared by new + resumed sessions. */
  const watchSession = useCallback(async (sid: string, statusId: string) => {
    const started = Date.now();
    for (;;) {
      const [done, prog] = await Promise.all([fetchSession(sid), fetchProgress(sid).catch(() => null)]);
      if (prog) setProgress(prog);
      fetchTrace(sid).then(setTrace).catch(() => {});
      if (done.status === "completed" || done.status === "failed") {
        setReport(done);
        setMsgs((p) => p.filter((m) => m.id !== statusId));
        if (done.status === "completed") {
          push({ id: nid(), role: "assistant", text: done.answer || done.summary || "Analysis complete.",
            evidence: done.answer_evidence ?? [], confidence: done.answer_confidence, verification: done.answer_verification });
          fetchEvents(sid).then(setEvents).catch(() => {});
          fetchEntities(sid).then((r) => setEntities(r.entities)).catch(() => {});
        } else {
          setError(prog?.error ? `Analysis failed: ${prog.error}` : "Analysis failed.");
        }
        void refreshSessions();
        break;
      }
      if (Date.now() - started > 15 * 60_000) throw new Error("Timed out (15 min). Try a shorter clip.");
      await new Promise((r) => setTimeout(r, 1500));
    }
  }, [push, refreshSessions]);

  /** Load a completed session's full state (report + thread + engine data). Persistent across restarts. */
  const openSession = useCallback(async (s: SessionSummary, playSrc?: string) => {
    unsubRef.current?.();
    setBusy(false); setAnswering(false); setError(null);
    setSessionId(s.session_id);
    try { localStorage.setItem(LS_KEY, s.session_id); } catch {}
    setSidebarOpen(false);
    setReport(null); setMsgs([]); setEvents([]); setEntities([]); setTrace([]);
    setVideoSrc(playSrc ?? (s.video_url?.startsWith("upload://") ? uploadUrlToHttp(s.video_url) : ""));
    setCurrentTime(0); setDuration(0);
    setProgress(null);
    if (s.status !== "completed" && s.status !== "failed") {
      setBusy(true);
      setProgress({ session_id: s.session_id, stage: "queued", detail: "Resuming — pipeline still running…", error: null, updated_at: Date.now() / 1000 });
      setElapsed(0); startedRef.current = Date.now();
      const statusId = nid();
      push({ id: nid(), role: "user", text: s.focus || s.title });
      push({ id: statusId, role: "status", text: "Resuming analysis…" });
      unsubRef.current = subscribeSession(s.session_id, (e) => {
        if (e.stage || e.detail) {
          setProgress((p) => ({ session_id: s.session_id, stage: e.stage ?? p?.stage ?? "processing",
            detail: e.detail ?? "", error: e.error ?? null, updated_at: Date.now() / 1000 }));
        }
      }, () => {});
      try {
        await watchSession(s.session_id, statusId);
      } catch (err) {
        setMsgs((p) => p.filter((m) => m.id !== statusId));
        setError(err instanceof Error ? err.message : "Analysis failed");
      } finally {
        setBusy(false);
        unsubRef.current?.();
      }
      return;
    }
    try {
      const [done, qa] = await Promise.all([fetchSession(s.session_id), listQa(s.session_id).catch(() => [])]);
      setReport(done);
      const thread: Msg[] = [];
      if (s.focus) thread.push({ id: nid(), role: "user", text: s.focus });
      if (done.status === "completed") {
        thread.push({ id: nid(), role: "assistant", text: done.answer || done.summary || "Analysis complete.",
          evidence: done.answer_evidence ?? [], confidence: done.answer_confidence, verification: done.answer_verification });
      }
      for (const q of qa) {
        thread.push({ id: nid(), role: "user", text: q.question });
        thread.push({ id: nid(), role: "assistant", text: q.answer, evidence: q.evidence ?? [],
          confidence: q.confidence, verification: q.verification });
      }
      setMsgs(thread);
      fetchEvents(s.session_id).then(setEvents).catch(() => {});
      fetchEntities(s.session_id).then((r) => setEntities(r.entities)).catch(() => {});
      fetchTrace(s.session_id).then(setTrace).catch(() => {});
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not load session");
    }
  }, [push, watchSession]);

  const openSessionRef = useRef(openSession);
  openSessionRef.current = openSession;

  // Boot: session index + restore last session.
  useEffect(() => {
    (async () => {
      let list: SessionSummary[] = [];
      try {
        list = await listSessions();
        setSessions(list);
      } catch { return; }
      let last: string | null = null;
      try { last = localStorage.getItem(LS_KEY); } catch {}
      const target = list.find((s) => s.session_id === last) ?? list[0];
      if (target && target.status !== "processing") {
        // Restore newest/last session silently (processing ones resume explicitly via sidebar).
        if (target.status === "completed" || target.status === "failed") {
          void openSessionRef.current(target);
        }
      }
    })();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const handleFiles = async (files: FileList | File[] | null) => {
    const file = files?.[0];
    if (!file) return;
    if (!file.type.startsWith("video/")) { setError("Please choose a video file (MP4)."); return; }
    setUploading(true); setError(null);
    try {
      const { video_url } = await uploadClip(file);
      setVideoUrl(video_url);
      setVideoSrc((prev) => {
        if (prev.startsWith("blob:")) URL.revokeObjectURL(prev);
        return URL.createObjectURL(file);
      });
      setFileName(file.name);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Upload failed");
    } finally {
      setUploading(false);
      if (fileRef.current) fileRef.current.value = "";
    }
  };

  const newChat = () => {
    unsubRef.current?.();
    setBusy(false); setAnswering(false);
    setSessionId(""); setReport(null); setMsgs([]);
    setEvents([]); setEntities([]); setTrace([]); setProgress(null);
    setVideoSrc(""); setVideoUrl(""); setFileName(""); setError(null);
    try { localStorage.removeItem(LS_KEY); } catch {}
  };

  const startSession = useCallback(async (question: string) => {
    if (!videoUrl || busy) return;
    unsubRef.current?.();
    const payload: VideoSessionPayload = { title: fileName || "Untitled video", focus: question, video_url: videoUrl, fps: 30 };
    const playSrc = videoSrc;
    setBusy(true); setError(null);
    setReport(null); setEvents([]); setEntities([]); setTrace([]);
    setProgress({ session_id: "", stage: "queued", detail: "Queued — pipeline starting…", error: null, updated_at: Date.now() / 1000 });
    setElapsed(0); startedRef.current = Date.now();
    push({ id: nid(), role: "user", text: question });
    const statusId = nid();
    push({ id: statusId, role: "status", text: "Watching your video…" });
    try {
      const sid = await createSession(payload);
      setSessionId(sid);
      try { localStorage.setItem(LS_KEY, sid); } catch {}
      const s: SessionSummary = { session_id: sid, title: payload.title, status: "processing",
        focus: question, video_url: videoUrl, updated_at: Date.now() / 1000 };
      setSessions((p) => [s, ...p.filter((x) => x.session_id !== sid)]);
      setVideoSrc(playSrc);
      unsubRef.current = subscribeSession(sid, (e) => {
        if (e.stage || e.detail) {
          setProgress((p) => ({ session_id: sid, stage: e.stage ?? p?.stage ?? "processing",
            detail: e.detail ?? "", error: e.error ?? null, updated_at: Date.now() / 1000 }));
        }
      }, () => {});
      await watchSession(sid, statusId);
    } catch (err) {
      setMsgs((p) => p.filter((m) => m.id !== statusId));
      setError(err instanceof Error ? err.message : "Analysis failed");
    } finally {
      setBusy(false);
      unsubRef.current?.();
    }
  }, [videoUrl, videoSrc, fileName, busy, push, watchSession]);

  const sendFollowup = useCallback(async () => {
    const text = input.trim();
    if (!text || !sessionId || busy || answering) return;
    setInput("");
    push({ id: nid(), role: "user", text });
    const pendingId = nid();
    push({ id: pendingId, role: "status", text: "Thinking…" });
    setAnswering(true);
    try {
      const r = await sendChat(sessionId, text);
      setMsgs((p) => p.map((m) => m.id === pendingId
        ? { id: pendingId, role: "assistant", text: r.answer || "I could not verify an answer from this video.",
            evidence: r.evidence ?? [], confidence: r.confidence, verification: r.verification, route: r.route }
        : m));
    } catch (e) {
      setMsgs((p) => p.map((m) => m.id === pendingId
        ? { id: pendingId, role: "assistant", text: `Sorry — ${e instanceof Error ? e.message : "that failed"}. Try again.` }
        : m));
    } finally {
      setAnswering(false);
      fetchTrace(sessionId).then(setTrace).catch(() => {});
    }
  }, [input, sessionId, busy, answering, push]);

  const analyzing = busy;
  const moments: VideoMoment[] = report?.moments ?? [];

  return (
    <div className="min-h-screen bg-gradient-to-b from-slate-950 via-slate-950 to-slate-900 text-white">
      <header className="sticky top-0 z-10 border-b border-slate-800/80 bg-slate-950/80 backdrop-blur">
        <div className="mx-auto flex max-w-5xl items-center gap-3 px-4 py-3">
          <button onClick={() => setSidebarOpen((v) => !v)} title="Chat sessions"
            className="rounded-lg border border-slate-700 px-2.5 py-1.5 text-sm text-slate-300 hover:border-slate-500">☰</button>
          <div className="flex h-8 w-8 items-center justify-center rounded-full bg-red-600 text-sm font-bold">δ</div>
          <div className="min-w-0 flex-1">
            <div className="truncate text-sm font-semibold">delta · video intel</div>
            <div className="truncate text-xs text-slate-400">
              {sessionId ? (analyzing ? `analyzing · ${progress?.stage ?? "queued"}` : "ask anything — follow-ups reuse the saved analysis") : "upload a clip, ask what you want"}
            </div>
          </div>
          <label className="flex cursor-pointer items-center gap-2 rounded-full border border-slate-700 px-3 py-1.5 text-xs text-slate-300 hover:border-slate-500">
            <input type="checkbox" checked={showEngine} onChange={(e) => setShowEngine(e.target.checked)} className="accent-red-500" />
            engine
          </label>
        </div>
      </header>

      <div className="mx-auto flex max-w-5xl">
        {(sidebarOpen || sessions.length > 0) && (
          <aside className={`${sidebarOpen ? "flex" : "hidden"} w-64 shrink-0 flex-col gap-1 overflow-y-auto border-r border-slate-800/80 p-3 md:flex`}
            style={{ maxHeight: "calc(100vh - 57px)", position: "sticky", top: 57 }}>
            <button onClick={() => { newChat(); setSidebarOpen(false); }}
              className="mb-2 rounded-xl bg-red-600 p-2.5 text-sm font-semibold hover:bg-red-500">+ New chat</button>
            {sessions.map((s) => (
              <button key={s.session_id} onClick={() => void openSession(s)}
                className={`rounded-xl p-3 text-left transition ${s.session_id === sessionId ? "bg-slate-800" : "hover:bg-slate-900"}`}>
                <div className="flex items-center gap-2">
                  <span className={`h-2 w-2 shrink-0 rounded-full ${s.status === "completed" ? "bg-emerald-500" : s.status === "failed" ? "bg-red-500" : "bg-amber-400 animate-pulse"}`} />
                  <span className="truncate text-sm font-medium">{s.title || "Untitled video"}</span>
                </div>
                <div className="mt-0.5 truncate text-xs text-slate-500">{s.focus || "—"}</div>
                <div className="mt-0.5 text-[11px] text-slate-600">{ago(s.updated_at)} · {s.status}</div>
              </button>
            ))}
            {sessions.length === 0 && <div className="p-3 text-xs text-slate-500">No chats yet — analyze a video to start.</div>}
          </aside>
        )}

        <main className="min-w-0 flex-1 space-y-4 px-4 pb-40 pt-6">
          {!sessionId ? (
            <section className="rounded-3xl border border-slate-800 bg-slate-900/60 p-8 text-center shadow-2xl">
              <h1 className="text-3xl font-semibold tracking-tight">Ask anything about your video</h1>
              <p className="mx-auto mt-2 max-w-md text-sm text-slate-400">
                Drop a clip, ask in plain words. Watch the analysis happen live, then keep asking follow-ups.
              </p>
              <div onClick={() => !uploading && fileRef.current?.click()}
                className="mx-auto mt-6 cursor-pointer rounded-2xl border border-dashed border-slate-600 bg-slate-800/40 p-6 transition hover:border-slate-400">
                <input ref={fileRef} type="file" accept="video/mp4,video/*" className="hidden"
                  onChange={(e) => void handleFiles(e.target.files)} />
                <div className="text-sm text-slate-200">
                  {uploading ? "Uploading…" : videoUrl ? `✓ ${fileName || "clip ready"}` : "Drop an MP4 here or click to browse"}
                </div>
                {videoUrl && <div className="mt-1 font-mono text-[11px] text-slate-500">{videoUrl.slice(0, 48)}…</div>}
              </div>
              <div className="mt-4 flex flex-wrap justify-center gap-2">
                {PRESETS.map((p) => (
                  <button key={p} onClick={() => setFocus(p)}
                    className={`rounded-full px-3 py-1.5 text-xs transition ${focus === p ? "bg-red-600 text-white" : "border border-slate-700 text-slate-300 hover:border-slate-500"}`}>
                    {p}
                  </button>
                ))}
              </div>
              <textarea value={focus} onChange={(e) => setFocus(e.target.value)} rows={3}
                placeholder="What do you want from this video?"
                className="mt-4 w-full rounded-2xl border border-slate-700 bg-slate-950 p-4 text-sm outline-none placeholder:text-slate-600 focus:border-red-500" />
              <button onClick={() => void startSession(focus.trim() || PRESETS[0])} disabled={!videoUrl || uploading || !focus.trim()}
                className="mt-4 w-full rounded-2xl bg-red-600 p-3.5 text-sm font-semibold transition hover:bg-red-500 disabled:opacity-40">
                {uploading ? "Uploading…" : "Analyze video"}
              </button>
              {error && <div className="mt-3 text-sm text-red-300">{error}</div>}
            </section>
          ) : (
            <>
              {videoSrc && (
                <section className="overflow-hidden rounded-3xl border border-slate-800 bg-black shadow-xl">
                  <video ref={videoRef} src={videoSrc} controls preload="metadata" className="aspect-video w-full"
                    onTimeUpdate={(e) => setCurrentTime(e.currentTarget.currentTime)}
                    onLoadedMetadata={(e) => setDuration(e.currentTarget.duration)} />
                </section>
              )}
              {moments.length > 0 && (
                <MomentTimelineBar moments={moments} currentTime={currentTime}
                  duration={duration || report?.video?.duration_s || 0}
                  selectedId={null} onSeek={seekTo} onSelect={(m) => seekTo(m.t_start)} />
              )}

              <section className="space-y-3" aria-live="polite">
                {msgs.map((m) => m.role === "user" ? (
                  <div key={m.id} className="flex justify-end">
                    <div className="max-w-[85%] rounded-3xl rounded-br-lg bg-slate-100 px-4 py-3 text-sm text-slate-900">{m.text}</div>
                  </div>
                ) : m.role === "status" ? (
                  <div key={m.id} className="flex">
                    <div className="w-full rounded-3xl rounded-bl-lg border border-slate-800 bg-slate-900/80 px-4 py-3">
                      <div className="flex items-center gap-2 text-sm text-slate-200">
                        <span className="relative flex h-2.5 w-2.5">
                          <span className="absolute h-full w-full animate-ping rounded-full bg-red-500 opacity-60" />
                          <span className="h-2.5 w-2.5 rounded-full bg-red-500" />
                        </span>
                        {analyzing ? (
                          <span>{progress?.stage === "completed" ? "Finishing…" : progress?.detail || m.text}
                            <span className="ml-2 text-xs text-slate-500">{elapsed}s</span></span>
                        ) : <span>{m.text}</span>}
                      </div>
                      {analyzing && (
                        <ol className="mt-3 flex flex-wrap gap-1.5">
                          {ORDER.map((s) => {
                            const order = ORDER as readonly string[];
                            const cur = order.indexOf(progress?.stage ?? "queued");
                            const mine = order.indexOf(s);
                            const done = mine < cur, active = mine === cur;
                            return (
                              <li key={s} className={`rounded-full px-2.5 py-1 text-[11px] ${
                                done ? "bg-emerald-900 text-emerald-200" : active ? "bg-red-600 text-white" : "bg-slate-800 text-slate-500"}`}>
                                {done ? "✓ " : ""}{s}
                              </li>
                            );
                          })}
                        </ol>
                      )}
                    </div>
                  </div>
                ) : (
                  <div key={m.id} className="flex gap-2.5">
                    <div className="mt-1 flex h-7 w-7 shrink-0 items-center justify-center rounded-full bg-red-600/15 text-xs text-red-300">✦</div>
                    <div className="max-w-[90%] flex-1 rounded-3xl rounded-bl-lg border border-slate-800 bg-slate-900 px-4 py-3">
                      <p className="whitespace-pre-wrap text-sm leading-relaxed text-slate-100">{m.text}</p>
                      {(m.route || m.verification != null) && (
                        <div className="mt-2 text-[11px] text-slate-500">
                          {[m.route, m.verification, m.confidence != null ? `${Math.round(m.confidence * 100)}%` : null]
                            .filter(Boolean).join(" · ")}
                        </div>
                      )}
                      {(m.evidence ?? []).length > 0 && (
                        <div className="mt-2 flex flex-wrap gap-1.5">
                          {m.evidence!.map((s, i) => (
                            <button key={i} onClick={() => seekTo(s.start)}
                              className="rounded-full bg-slate-800 px-2.5 py-1 font-mono text-[11px] text-sky-300 hover:bg-slate-700">
                              {s.start.toFixed(1)}–{s.end.toFixed(1)}s
                            </button>
                          ))}
                        </div>
                      )}
                    </div>
                  </div>
                ))}
                <div ref={bottomRef} />
              </section>

              {error && <div className="rounded-2xl bg-red-950 p-3 text-sm text-red-200">{error}</div>}

              {showEngine && (
                <section className="space-y-3 rounded-3xl border border-slate-800 bg-slate-950/60 p-4">
                  <div className="text-xs font-semibold uppercase tracking-wider text-slate-400">Under the hood</div>
                  {trace.length > 0 && <TracePanel trace={trace} />}
                  {events.length > 0 && (
                    <details open className="rounded-2xl bg-slate-900 p-3">
                      <summary className="cursor-pointer text-xs uppercase tracking-wider text-slate-400">Event graph ({events.length})</summary>
                      <div className="mt-2"><EventList events={events} onSeek={seekTo} /></div>
                    </details>
                  )}
                  {entities.length > 0 && (
                    <details className="rounded-2xl bg-slate-900 p-3">
                      <summary className="cursor-pointer text-xs uppercase tracking-wider text-slate-400">Entities ({entities.length})</summary>
                      <div className="mt-2"><EntityList entities={entities} /></div>
                    </details>
                  )}
                  {report?.transcript ? (
                    <details className="rounded-2xl bg-slate-900 p-3">
                      <summary className="cursor-pointer text-xs uppercase tracking-wider text-slate-400">Transcript</summary>
                      <p className="mt-2 whitespace-pre-wrap text-xs text-slate-300">{report.transcript || "—"}</p>
                    </details>
                  ) : null}
                </section>
              )}
            </>
          )}
        </main>
      </div>

      {sessionId && (
        <footer className="fixed inset-x-0 bottom-0 border-t border-slate-800 bg-slate-950/90 backdrop-blur">
          <div className="mx-auto flex max-w-5xl items-end gap-2 px-4 py-3">
            <textarea value={input} rows={1} onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); void sendFollowup(); } }}
              placeholder={analyzing ? "Wait for the analysis to finish…" : "Ask a follow-up… (Enter to send)"}
              disabled={analyzing || answering}
              className="max-h-32 flex-1 resize-none rounded-2xl border border-slate-700 bg-slate-900 px-4 py-3 text-sm outline-none placeholder:text-slate-600 focus:border-red-500 disabled:opacity-50" />
            <button onClick={() => void sendFollowup()} disabled={analyzing || answering || !input.trim()}
              className="rounded-2xl bg-red-600 px-5 py-3 text-sm font-semibold transition hover:bg-red-500 disabled:opacity-40">
              {answering ? "…" : "↑"}
            </button>
          </div>
        </footer>
      )}
    </div>
  );
}
