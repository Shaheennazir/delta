"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { MomentTimelineBar, fmt } from "@/components/MomentTimelineBar";
import { MomentCards } from "@/components/MomentCards";
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
  { tag: "Story", text: "Summarize what happens and the key takeaways" },
  { tag: "People", text: "Who is in the video and what do they do?" },
  { tag: "Beats", text: "List the key moments with timestamps" },
  { tag: "Words", text: "Transcribe what is said and shown on screen" },
];
const LS_KEY = "delta-session-id";

let _id = 0;
const nid = () => `m${Date.now().toString(36)}${(_id++).toString(36)}`;
const ago = (t: number) => {
  const s = Math.max(0, Math.floor(Date.now() / 1000 - t));
  if (s < 60) return "just now";
  if (s < 3600) return `${Math.floor(s / 60)}m`;
  if (s < 86400) return `${Math.floor(s / 3600)}h`;
  return `${Math.floor(s / 86400)}d`;
};

function ConfBadge({ v }: { v?: number }) {
  if (v == null) return null;
  const p = Math.round(v * 100);
  const c = p >= 80 ? "text-[#3ddc97] border-[#3ddc97]/25 bg-[#3ddc97]/10" : p >= 55 ? "text-[#ffb224] border-[#ffb224]/25 bg-[#ffb224]/10" : "text-[#ff8d82] border-[#ff3d2e]/25 bg-[#ff3d2e]/10";
  return (
    <span className={`font-mono2 rounded-full border px-2 py-0.5 text-[10.5px] tabular-nums ${c}`}>{p}%</span>
  );
}

export default function VideoChat() {
  const [videoSrc, setVideoSrc] = useState("");
  const [sessionId, setSessionId] = useState("");
  const [sessions, setSessions] = useState<SessionSummary[]>([]);
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [report, setReport] = useState<SessionReport | null>(null);
  const [msgs, setMsgs] = useState<Msg[]>([]);
  const [input, setInput] = useState("");
  const [focus, setFocus] = useState(PRESETS[0].text);
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
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [railTab, setRailTab] = useState<"moments" | "graph" | "intel">("moments");
  const [dragOver, setDragOver] = useState(false);

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

  const openSession = useCallback(async (s: SessionSummary, playSrc?: string) => {
    unsubRef.current?.();
    setBusy(false); setAnswering(false); setError(null);
    setSessionId(s.session_id);
    try { localStorage.setItem(LS_KEY, s.session_id); } catch {}
    setSidebarOpen(false);
    setReport(null); setMsgs([]); setEvents([]); setEntities([]); setTrace([]);
    setSelectedId(null); setRailTab("moments");
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
      if (target && (target.status === "completed" || target.status === "failed")) {
        void openSessionRef.current(target);
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
    setSelectedId(null);
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
  const stageIdx = ORDER.indexOf((progress?.stage ?? "queued") as (typeof ORDER)[number]);
  const stagePct = analyzing ? Math.min(100, Math.max(6, ((Math.max(stageIdx, 0)) / ORDER.length) * 100)) : 100;

  const selectMoment = (m: VideoMoment) => {
    setSelectedId(m.moment_id);
    seekTo(m.t_start);
  };

  return (
    <div className="bg-scene noise min-h-dvh text-[#f4f1ea]">
      <div className="bg-grid pointer-events-none fixed inset-0" />

      {/* ── top bar ─────────────────────────────── */}
      <header className="glass sticky top-0 z-40 border-b border-white/[0.07]">
        <div className="mx-auto flex h-[60px] max-w-[1440px] items-center gap-3 px-4 md:px-6">
          <button
            onClick={() => setSidebarOpen((v) => !v)}
            aria-label="Toggle sessions"
            className="pressable flex h-9 w-9 items-center justify-center rounded-xl border border-white/10 bg-white/[0.04] lg:hidden"
          >
            <span className="flex flex-col gap-[5px]">
              <i className={`block h-[1.5px] w-4 bg-white transition-transform duration-300 ${sidebarOpen ? "translate-y-[6.5px] rotate-45" : ""}`} />
              <i className={`block h-[1.5px] w-4 bg-white transition-opacity duration-200 ${sidebarOpen ? "opacity-0" : ""}`} />
              <i className={`block h-[1.5px] w-4 bg-white transition-transform duration-300 ${sidebarOpen ? "-translate-y-[6.5px] -rotate-45" : ""}`} />
            </span>
          </button>

          <button onClick={newChat} className="pressable flex items-center gap-2.5" title="New analysis">
            <span className="font-display flex h-9 w-9 items-center justify-center rounded-2xl bg-[#ff3d2e] text-[17px] font-extrabold text-white shadow-[0_8px_28px_rgba(255,61,46,0.4)]">
              δ
            </span>
            <span className="hidden text-left sm:block">
              <span className="font-display block text-[15px] font-bold leading-none tracking-tight">delta</span>
              <span className="font-mono2 mt-1 block text-[10px] uppercase tracking-[0.24em] text-[#63615c]">video intel</span>
            </span>
          </button>

          <div className="mx-2 hidden h-6 w-px bg-white/10 md:block" />

          <div className="font-mono2 hidden min-w-0 flex-1 items-center gap-2 text-[11.5px] text-[#63615c] md:flex">
            {sessionId ? (
              analyzing ? (
                <>
                  <span className="live-dot h-1.5 w-1.5 rounded-full bg-[#ff3d2e]" />
                  <span className="truncate text-[#a7a5a0]">{progress?.detail ?? "analyzing"}</span>
                  <span className="tabular-nums text-[#ff8d82]">{elapsed}s</span>
                </>
              ) : (
                <>
                  <span className="h-1.5 w-1.5 rounded-full bg-[#3ddc97]" />
                  <span className="truncate">evidence locked · {moments.length} moments · {events.length} events · ask anything</span>
                </>
              )
            ) : (
              <span>drop a clip → ask in plain words → get grounded answers</span>
            )}
          </div>

          <div className="flex-1 md:hidden" />

          <label className="pressable flex cursor-pointer items-center gap-2 rounded-full border border-white/10 bg-white/[0.04] py-1.5 pl-3 pr-1.5 text-xs text-[#a7a5a0]">
            <span className="font-mono2 text-[10px] uppercase tracking-[0.18em]">engine</span>
            <span
              role="switch"
              aria-checked={showEngine}
              onClick={(e) => { e.preventDefault(); setShowEngine(!showEngine); }}
              className={`relative h-6 w-11 rounded-full transition-colors duration-300 ${showEngine ? "bg-[#ff3d2e]" : "bg-white/10"}`}
            >
              <i className={`absolute top-0.5 h-5 w-5 rounded-full bg-white shadow transition-transform duration-300 ${showEngine ? "translate-x-[22px]" : "translate-x-0.5"}`} style={{ transitionTimingFunction: "cubic-bezier(0.32,0.72,0,1)" }} />
            </span>
            <input type="checkbox" checked={showEngine} onChange={(e) => setShowEngine(e.target.checked)} className="sr-only" />
          </label>
        </div>

        {/* pipeline progress hairline */}
        {analyzing && (
          <div className="h-[2px] w-full bg-white/[0.06]">
            <div
              className="flow-bar h-full bg-[#ff3d2e] transition-[width] duration-700"
              style={{ width: `${stagePct}%`, transitionTimingFunction: "cubic-bezier(0.32,0.72,0,1)" }}
            />
          </div>
        )}
      </header>

      <div className="relative z-10 mx-auto grid max-w-[1440px] gap-4 px-4 pb-32 pt-4 md:px-6 lg:grid-cols-[272px_minmax(0,1fr)_336px]">
        {/* ── session rail ── */}
        <aside
          className={`${
            sidebarOpen ? "fixed inset-y-[60px] left-0 z-30 flex w-[300px] flex-col bg-[#0a0b0e]/95 p-4 backdrop-blur-2xl lg:static lg:z-auto lg:w-auto lg:bg-transparent lg:p-0" : "hidden lg:flex"
          } flex-col gap-2 lg:sticky lg:top-[76px] lg:max-h-[calc(100dvh-92px)]`}
        >
          <button
            onClick={() => { newChat(); setSidebarOpen(false); }}
            className="pressable btn-primary group flex items-center justify-between rounded-2xl px-4 py-3 text-sm font-semibold"
          >
            <span>+ New analysis</span>
            <span className="flex h-7 w-7 items-center justify-center rounded-full bg-black/20 text-base transition-transform duration-300 group-hover:translate-x-0.5" style={{ transitionTimingFunction: "cubic-bezier(0.32,0.72,0,1)" }}>↗</span>
          </button>

          <div className="slim-scroll -mx-1 flex-1 space-y-1.5 overflow-y-auto px-1 py-1">
            {sessions.map((s, i) => {
              const activeS = s.session_id === sessionId;
              return (
                <button
                  key={s.session_id}
                  onClick={() => void openSession(s)}
                  style={{ animationDelay: `${Math.min(i * 40, 300)}ms` }}
                  className={`rise pressable w-full rounded-2xl border p-3 text-left ${
                    activeS ? "border-[#ff3d2e]/40 bg-[#ff3d2e]/[0.07]" : "border-transparent hover:border-white/10 hover:bg-white/[0.04]"
                  }`}
                >
                  <div className="flex items-center gap-2">
                    <span className={`h-1.5 w-1.5 shrink-0 rounded-full ${s.status === "completed" ? "bg-[#3ddc97]" : s.status === "failed" ? "bg-[#ff3d2e]" : "animate-pulse bg-[#ffb224]"}`} />
                    <span className="truncate text-[13px] font-semibold">{s.title || "Untitled video"}</span>
                  </div>
                  <div className="mt-1 truncate text-xs text-[#63615c]">{s.focus || "—"}</div>
                  <div className="font-mono2 mt-1.5 text-[10.5px] uppercase tracking-wider text-[#4a4844]">{ago(s.updated_at)} · {s.status}</div>
                </button>
              );
            })}
            {sessions.length === 0 && (
              <div className="rounded-2xl border border-dashed border-white/10 p-5 text-center text-xs leading-relaxed text-[#63615c]">
                No sessions yet.<br />Your analyses live here.
              </div>
            )}
          </div>

          <div className="font-mono2 hidden rounded-2xl border border-white/[0.07] bg-white/[0.02] p-3.5 text-[10.5px] leading-relaxed text-[#63615c] lg:block">
            REAL MODELS ONLY<br />no demo data · evidence<br />or it didn&apos;t happen
          </div>
        </aside>

        {/* ── center stage ── */}
        <main className="min-w-0 space-y-4">
          {!sessionId ? (
            <section className="rise overflow-hidden">
              <div className="bezel">
                <div className="bezel-inner relative overflow-hidden px-6 py-10 text-center md:px-12 md:py-14">
                  <div className="pointer-events-none absolute inset-0 bg-[radial-gradient(600px_240px_at_50%_0%,rgba(255,61,46,0.12),transparent_70%)]" />
                  <div className="rise relative">
                    <span className="font-mono2 inline-flex items-center gap-2 rounded-full border border-white/10 bg-white/[0.04] px-3.5 py-1.5 text-[10px] uppercase tracking-[0.24em] text-[#a7a5a0]">
                      <i className="h-1.5 w-1.5 rounded-full bg-[#ff3d2e]" style={{ animation: "blink 1.6s ease infinite" }} />
                      grounded video intelligence
                    </span>
                  </div>
                  <h1 className="font-display rise-1 relative mx-auto mt-5 max-w-[16ch] text-[clamp(2.2rem,6vw,4.2rem)] font-extrabold leading-[0.95] tracking-[-0.04em]">
                    Ask anything about your <span className="text-[#ff3d2e]">video.</span>
                  </h1>
                  <p className="rise-2 relative mx-auto mt-4 max-w-md text-[15px] leading-relaxed text-[#a7a5a0]">
                    Drop a clip, ask in plain words. Watch the pipeline work live — every answer carries timestamps you can click.
                  </p>

                  <div
                    onClick={() => !uploading && fileRef.current?.click()}
                    onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
                    onDragLeave={() => setDragOver(false)}
                    onDrop={(e) => { e.preventDefault(); setDragOver(false); void handleFiles(e.dataTransfer.files); }}
                    className={`rise-2 pressable relative mx-auto mt-8 max-w-lg cursor-pointer rounded-3xl border border-dashed p-8 transition-colors duration-300 ${
                      dragOver ? "border-[#ff3d2e]/70 bg-[#ff3d2e]/[0.06]" : videoUrl ? "border-[#3ddc97]/40 bg-[#3ddc97]/[0.04]" : "border-white/15 bg-black/40 hover:border-white/30"
                    }`}
                  >
                    <input ref={fileRef} type="file" accept="video/mp4,video/*" className="hidden" onChange={(e) => void handleFiles(e.target.files)} />
                    <div className="mx-auto flex h-12 w-12 items-center justify-center rounded-2xl bg-white/[0.06] text-xl">
                      {uploading ? <span className="animate-pulse">◌</span> : videoUrl ? <span className="text-[#3ddc97]">✓</span> : "▣"}
                    </div>
                    <div className="mt-3 text-sm font-medium">
                      {uploading ? "Uploading…" : videoUrl ? (fileName || "Clip ready") : dragOver ? "Drop it — let's watch" : "Drop an MP4 here or click to browse"}
                    </div>
                    <div className="font-mono2 mt-1 text-[11px] text-[#63615c]">
                      {videoUrl ? videoUrl.slice(0, 52) + "…" : "mp4 · up to 30 min · real models, no demo data"}
                    </div>
                  </div>

                  <div className="rise-3 relative mt-6 flex flex-wrap justify-center gap-2">
                    {PRESETS.map((p) => (
                      <button
                        key={p.tag}
                        onClick={() => setFocus(p.text)}
                        className={`pressable rounded-full border px-4 py-2 text-left text-xs transition-colors duration-200 ${
                          focus === p.text ? "border-[#ff3d2e] bg-[#ff3d2e] text-white shadow-[0_8px_28px_rgba(255,61,46,0.35)]" : "border-white/10 bg-white/[0.03] text-[#a7a5a0] hover:border-white/25 hover:text-white"
                        }`}
                      >
                        <span className="font-mono2 mr-1.5 text-[10px] uppercase tracking-widest opacity-70">{p.tag}</span>
                        <span className="hidden sm:inline">{p.text.slice(0, 34)}…</span>
                      </button>
                    ))}
                  </div>

                  <div className="rise-4 relative mx-auto mt-4 max-w-lg">
                    <div className="rounded-3xl border border-white/10 bg-black/50 p-1.5 focus-within:border-[#ff3d2e]/60">
                      <textarea
                        value={focus}
                        onChange={(e) => setFocus(e.target.value)}
                        rows={3}
                        placeholder="What do you want from this video?"
                        className="w-full resize-none rounded-2xl bg-transparent px-4 py-3 text-sm leading-relaxed outline-none placeholder:text-[#4a4844]"
                      />
                      <button
                        onClick={() => void startSession(focus.trim() || PRESETS[0].text)}
                        disabled={!videoUrl || uploading || !focus.trim()}
                        className="pressable btn-primary group flex w-full items-center justify-center gap-2 rounded-2xl px-4 py-3.5 text-sm font-semibold disabled:opacity-40"
                      >
                        <span>{uploading ? "Uploading clip…" : "Analyze video"}</span>
                        <span className="flex h-6 w-6 items-center justify-center rounded-full bg-black/20 transition-transform duration-300 group-hover:translate-x-1" style={{ transitionTimingFunction: "cubic-bezier(0.32,0.72,0,1)" }}>→</span>
                      </button>
                    </div>
                  </div>
                  {error && <div className="rise relative mx-auto mt-3 max-w-lg rounded-2xl border border-[#ff3d2e]/30 bg-[#ff3d2e]/10 p-3 text-sm text-[#ff9d94]">{error}</div>}
                </div>
              </div>

              {/* strip */}
              <div className="mt-4 grid grid-cols-3 gap-2 md:gap-3">
                {[["01", "Watch", "frame-accurate playback"], ["02", "Verify", "evidence or abstain"], ["03", "Ask", "follow-ups reuse analysis"]].map((s, i) => (
                  <div key={s[1]} style={{ animationDelay: `${200 + i * 80}ms` }} className="rise rounded-2xl border border-white/[0.07] bg-white/[0.02] p-4">
                    <div className="font-mono2 text-[11px] text-[#ff3d2e]">{s[0]}</div>
                    <div className="font-display mt-1 text-[15px] font-bold">{s[1]}</div>
                    <div className="mt-0.5 text-xs text-[#63615c]">{s[2]}</div>
                  </div>
                ))}
              </div>
            </section>
          ) : (
            <>
              {videoSrc && (
                <section className="rise">
                  <div className="bezel">
                    <div className="bezel-inner overflow-hidden">
                      <div className="flex items-center gap-2 border-b border-white/[0.06] px-4 py-2.5">
                        <span className="flex gap-1.5">
                          <i className="h-2 w-2 rounded-full bg-white/15" />
                          <i className="h-2 w-2 rounded-full bg-white/15" />
                          <i className="h-2 w-2 rounded-full bg-[#ff3d2e]" />
                        </span>
                        <span className="font-mono2 truncate text-[10.5px] uppercase tracking-[0.18em] text-[#63615c]">
                          {report?.title || fileName || "screening room"}
                        </span>
                        <span className="font-mono2 ml-auto hidden text-[10.5px] tabular-nums text-[#63615c] sm:block">
                          {report?.video ? `${report.video.width}×${report.video.height} · ${report.video.fps}fps` : ""}
                        </span>
                      </div>
                      <video
                        ref={videoRef}
                        src={videoSrc}
                        controls
                        playsInline
                        preload="metadata"
                        className="aspect-video w-full bg-black"
                        onTimeUpdate={(e) => setCurrentTime(e.currentTarget.currentTime)}
                        onLoadedMetadata={(e) => setDuration(e.currentTarget.duration)}
                      />
                    </div>
                  </div>
                </section>
              )}

              {moments.length > 0 && (
                <div className="rise-1">
                  <MomentTimelineBar
                    moments={moments}
                    currentTime={currentTime}
                    duration={duration || report?.video?.duration_s || 0}
                    selectedId={selectedId}
                    onSeek={seekTo}
                    onSelect={selectMoment}
                  />
                </div>
              )}

              <section className="space-y-3" aria-live="polite">
                {msgs.map((m, idx) => m.role === "user" ? (
                  <div key={m.id} className="msg-in flex justify-end" style={{ animationDelay: `${Math.min(idx * 30, 150)}ms` }}>
                    <div className="max-w-[85%] rounded-3xl rounded-br-lg bg-[#f4f1ea] px-4 py-3 text-sm leading-relaxed text-[#141310] shadow-[0_12px_40px_rgba(0,0,0,0.35)]">
                      {m.text}
                    </div>
                  </div>
                ) : m.role === "status" ? (
                  <div key={m.id} className="fade-in flex">
                    <div className="w-full overflow-hidden rounded-3xl rounded-bl-lg border border-white/[0.08] bg-[#0e0f13]/90">
                      <div className="flex items-center gap-2.5 px-4 py-3.5">
                        <span className="live-dot h-2 w-2 shrink-0 rounded-full bg-[#ff3d2e]" />
                        <span className="min-w-0 flex-1 truncate text-sm text-[#d8d5cf]">
                          {analyzing ? (progress?.detail || m.text) : m.text}
                        </span>
                        {analyzing && <span className="font-mono2 shrink-0 text-xs tabular-nums text-[#ff8d82]">{elapsed}s</span>}
                      </div>
                      {analyzing && (
                        <div className="border-t border-white/[0.06] px-4 py-3">
                          <div className="mb-2.5 h-1 overflow-hidden rounded-full bg-white/[0.07]">
                            <div className="flow-bar h-full rounded-full bg-[#ff3d2e] transition-[width] duration-700" style={{ width: `${stagePct}%` }} />
                          </div>
                          <ol className="flex flex-wrap gap-1.5">
                            {ORDER.map((s) => {
                              const cur = ORDER.indexOf((progress?.stage ?? "queued") as (typeof ORDER)[number]);
                              const mine = ORDER.indexOf(s);
                              const done = mine < cur, active = mine === cur;
                              return (
                                <li
                                  key={s}
                                  className={`font-mono2 rounded-full px-2.5 py-1 text-[10.5px] transition-all duration-300 ${
                                    done ? "bg-[#3ddc97]/15 text-[#3ddc97]" : active ? "shimmer-line bg-[#ff3d2e] text-white" : "bg-white/[0.05] text-[#63615c]"
                                  }`}
                                >
                                  {done ? `✓ ${s}` : s}
                                </li>
                              );
                            })}
                          </ol>
                        </div>
                      )}
                    </div>
                  </div>
                ) : (
                  <div key={m.id} className="msg-in flex gap-2.5" style={{ animationDelay: `${Math.min(idx * 30, 150)}ms` }}>
                    <div className="font-display mt-1 flex h-8 w-8 shrink-0 items-center justify-center rounded-xl bg-[#ff3d2e]/15 text-sm text-[#ff8d82]">✦</div>
                    <div className="min-w-0 max-w-[92%] flex-1 rounded-3xl rounded-bl-lg border border-white/[0.08] bg-[#101116] px-4 py-3.5 shadow-[0_16px_50px_rgba(0,0,0,0.4)]">
                      <p className="whitespace-pre-wrap text-[14px] leading-[1.7] text-[#ece9e2]">{m.text}</p>
                      {(m.route || m.verification != null || m.confidence != null) && (
                        <div className="mt-2.5 flex flex-wrap items-center gap-1.5">
                          {m.verification && (
                            <span className={`font-mono2 rounded-full border px-2 py-0.5 text-[10.5px] ${m.verification === "supported" ? "border-[#3ddc97]/25 bg-[#3ddc97]/10 text-[#3ddc97]" : "border-[#ffb224]/25 bg-[#ffb224]/10 text-[#ffb224]"}`}>
                              {m.verification}
                            </span>
                          )}
                          <ConfBadge v={m.confidence} />
                          {m.route && <span className="font-mono2 text-[10.5px] text-[#4a4844]">{m.route}</span>}
                        </div>
                      )}
                      {(m.evidence ?? []).length > 0 && (
                        <div className="mt-2.5 flex flex-wrap gap-1.5 border-t border-white/[0.06] pt-2.5">
                          {m.evidence!.map((s, i) => (
                            <button
                              key={i}
                              onClick={() => seekTo(s.start)}
                              className="pressable font-mono2 rounded-full border border-white/10 bg-white/[0.05] px-2.5 py-1 text-[11px] tabular-nums text-sky-300 hover:border-sky-300/40 hover:bg-sky-300/10"
                            >
                              ▸ {s.start.toFixed(1)}–{s.end.toFixed(1)}s
                            </button>
                          ))}
                        </div>
                      )}
                    </div>
                  </div>
                ))}
                <div ref={bottomRef} />
              </section>

              {error && <div className="fade-in rounded-2xl border border-[#ff3d2e]/30 bg-[#ff3d2e]/10 p-3.5 text-sm text-[#ff9d94]">{error}</div>}

              {showEngine && (
                <section className="fade-in space-y-3 rounded-3xl border border-white/[0.08] bg-black/40 p-3.5">
                  <div className="font-mono2 px-1 text-[10px] uppercase tracking-[0.24em] text-[#63615c]">under the hood</div>
                  {trace.length > 0 && <TracePanel trace={trace} />}
                  {report?.transcript ? (
                    <details className="rounded-2xl border border-white/[0.07] bg-white/[0.02] p-4">
                      <summary className="font-mono2 cursor-pointer text-[10px] uppercase tracking-[0.2em] text-[#63615c]">transcript</summary>
                      <p className="slim-scroll mt-2 max-h-48 overflow-y-auto whitespace-pre-wrap text-xs leading-relaxed text-[#a7a5a0]">{report.transcript || "—"}</p>
                    </details>
                  ) : null}
                </section>
              )}
            </>
          )}
        </main>

        {/* ── intel rail ── */}
        <aside className="min-w-0 lg:sticky lg:top-[76px] lg:max-h-[calc(100dvh-92px)] lg:overflow-y-auto slim-scroll">
          {sessionId ? (
            <div className="rise-2 space-y-3">
              <div className="grid grid-cols-3 gap-1 rounded-2xl border border-white/[0.08] bg-white/[0.03] p-1">
                {(["moments", "graph", "intel"] as const).map((t) => (
                  <button
                    key={t}
                    onClick={() => setRailTab(t)}
                    className={`pressable font-mono2 rounded-xl px-2 py-2 text-[10.5px] uppercase tracking-[0.14em] transition-colors duration-200 ${
                      railTab === t ? "bg-white text-black" : "text-[#63615c] hover:text-white"
                    }`}
                  >
                    {t}
                  </button>
                ))}
              </div>

              {railTab === "moments" && (
                <MomentCards moments={moments} selectedId={selectedId} currentTime={currentTime} onSelect={selectMoment} />
              )}

              {railTab === "graph" && (
                <div className="fade-in space-y-3">
                  {events.length > 0 && <EventList events={events} onSeek={seekTo} />}
                  {events.length === 0 && (
                    <div className="rounded-2xl border border-dashed border-white/10 bg-white/[0.02] p-5 text-center text-[13px] text-[#63615c]">
                      Event graph lands here once verified.
                    </div>
                  )}
                </div>
              )}

              {railTab === "intel" && (
                <div className="fade-in space-y-3">
                  <div className="rounded-2xl border border-white/[0.07] bg-white/[0.02] p-4">
                    <div className="font-mono2 mb-2.5 text-[10px] uppercase tracking-[0.2em] text-[#63615c]">
                      entities · {entities.length}
                    </div>
                    <EntityList entities={entities} />
                  </div>
                  {report && (
                    <div className="rounded-2xl border border-white/[0.07] bg-white/[0.02] p-4">
                      <div className="font-mono2 mb-2 text-[10px] uppercase tracking-[0.2em] text-[#63615c]">signal</div>
                      <dl className="font-mono2 space-y-1.5 text-[11.5px]">
                        <div className="flex justify-between"><dt className="text-[#63615c]">frames</dt><dd className="tabular-nums">{report.n_frames_indexed ?? "—"}</dd></div>
                        <div className="flex justify-between"><dt className="text-[#63615c]">events</dt><dd className="tabular-nums">{report.n_events ?? events.length}</dd></div>
                        <div className="flex justify-between"><dt className="text-[#63615c]">entities</dt><dd className="tabular-nums">{report.n_entities ?? entities.length}</dd></div>
                        <div className="flex justify-between"><dt className="text-[#63615c]">ablation</dt><dd>L{report.ablation_level ?? "—"}</dd></div>
                        {report.av_offset_s != null && (
                          <div className="flex justify-between"><dt className="text-[#63615c]">av offset</dt><dd className="tabular-nums">{report.av_offset_s.toFixed(2)}s</dd></div>
                        )}
                      </dl>
                      {report.themes?.length > 0 && (
                        <div className="mt-3 flex flex-wrap gap-1.5">
                          {report.themes.map((t) => (
                            <span key={t} className="rounded-full bg-white/[0.06] px-2.5 py-1 text-[11px] text-[#a7a5a0]">{t}</span>
                          ))}
                        </div>
                      )}
                    </div>
                  )}
                  {report?.on_screen_text ? (
                    <details className="rounded-2xl border border-white/[0.07] bg-white/[0.02] p-4">
                      <summary className="font-mono2 cursor-pointer text-[10px] uppercase tracking-[0.2em] text-[#63615c]">on-screen text</summary>
                      <p className="mt-2 whitespace-pre-wrap text-xs leading-relaxed text-[#a7a5a0]">{report.on_screen_text}</p>
                    </details>
                  ) : null}
                </div>
              )}
            </div>
          ) : (
            <div className="rise-3 hidden space-y-3 lg:block">
              <div className="rounded-3xl border border-white/[0.08] bg-gradient-to-b from-white/[0.05] to-transparent p-6">
                <div className="font-mono2 text-[10px] uppercase tracking-[0.24em] text-[#ff8d82]">how it works</div>
                <ol className="mt-4 space-y-4">
                  {[
                    ["Drop", "Your clip never leaves the pipeline. Uploads stream straight to analysis."],
                    ["Ground", "Every claim is pinned to a timestamp. No timestamp, no claim."],
                    ["Interrogate", "Follow-ups reuse the saved analysis — no re-uploads."],
                  ].map((s, i) => (
                    <li key={s[0]} className="flex gap-3">
                      <span className="font-mono2 text-[11px] text-[#4a4844]">0{i + 1}</span>
                      <div>
                        <div className="font-display text-[15px] font-bold">{s[0]}</div>
                        <div className="mt-0.5 text-[13px] leading-relaxed text-[#63615c]">{s[1]}</div>
                      </div>
                    </li>
                  ))}
                </ol>
              </div>
              <div className="font-mono2 rounded-2xl border border-[#3ddc97]/20 bg-[#3ddc97]/[0.05] p-4 text-[11px] leading-relaxed text-[#3ddc97]">
                ✓ abstains when evidence is thin<br />✓ cites frame spans, not vibes
              </div>
            </div>
          )}
        </aside>
      </div>

      {/* ── floating composer ── */}
      {sessionId && (
        <footer className="fixed inset-x-0 bottom-0 z-40 px-4 pb-4 md:pb-6">
          <div className="fade-in mx-auto max-w-[720px]">
            <div className="bezel glass shadow-[0_24px_80px_rgba(0,0,0,0.6)]">
              <div className="flex items-end gap-2 p-1.5">
                <textarea
                  value={input}
                  rows={1}
                  onChange={(e) => setInput(e.target.value)}
                  onKeyDown={(e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); void sendFollowup(); } }}
                  placeholder={analyzing ? "Pipeline running — hold that thought…" : "Ask a follow-up… Enter to send"}
                  disabled={analyzing || answering}
                  className="max-h-32 min-w-0 flex-1 resize-none bg-transparent px-3.5 py-3 text-sm outline-none placeholder:text-[#4a4844] disabled:opacity-50"
                />
                <button
                  onClick={() => void sendFollowup()}
                  disabled={analyzing || answering || !input.trim()}
                  aria-label="Send"
                  className="pressable btn-primary flex h-11 w-11 shrink-0 items-center justify-center rounded-2xl text-lg disabled:opacity-40"
                >
                  {answering ? <span className="animate-pulse text-sm">◌</span> : "↑"}
                </button>
              </div>
            </div>
            <div className="font-mono2 mt-2 text-center text-[10.5px] text-[#4a4844]">
              {fmt(currentTime)} · click any timestamp chip to seek · {moments.length} moments grounded
            </div>
          </div>
        </footer>
      )}
    </div>
  );
}
