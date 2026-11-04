"use client";

import { useState } from "react";
import { askQuestion, runBenchmark } from "@/lib/api";
import type { BenchmarkTest, Entity, GraphEvent, QARecord } from "@/lib/types";
import { fmt } from "./MomentTimelineBar";

export function EventList({ events, onSeek }: { events: GraphEvent[]; onSeek: (t: number) => void }) {
  if (!events.length)
    return <div className="rounded-2xl border border-white/[0.07] bg-white/[0.02] p-4 text-[13px] text-[#63615c]">No grounded events yet.</div>;
  return (
    <div className="space-y-2">
      {events.map((e, i) => (
        <div
          key={e.event_id}
          onClick={() => onSeek(e.start_t)}
          role="button"
          tabIndex={0}
          onKeyDown={(ev) => { if (ev.key === "Enter") onSeek(e.start_t); }}
          style={{ animationDelay: `${Math.min(i * 50, 300)}ms` }}
          className="rise pressable cursor-pointer rounded-2xl border border-white/[0.07] bg-white/[0.025] p-3.5 hover:border-white/15 hover:bg-white/[0.05]"
        >
          <div className="flex items-center justify-between gap-2">
            <span className="font-mono2 truncate text-[11px] text-[#3ddc97]">{e.event_id}</span>
            <span className="font-mono2 shrink-0 rounded-full bg-white/[0.06] px-2 py-0.5 text-[10.5px] tabular-nums text-[#a7a5a0]">
              {fmt(e.start_t)} · {Math.round(e.confidence * 100)}%
            </span>
          </div>
          <div className="mt-1.5 text-[13.5px] font-medium text-[#f4f1ea]">{e.action}</div>
          <div className="mt-0.5 line-clamp-2 text-xs leading-relaxed text-[#a7a5a0]">{e.detail}</div>
          <div className="font-mono2 mt-2 flex flex-wrap items-center gap-1.5 text-[10.5px]">
            {e.source && <span className="rounded-full border border-white/10 px-2 py-0.5 text-[#a7a5a0]">{e.source}</span>}
            <button
              onClick={(ev) => { ev.stopPropagation(); onSeek(e.start_t); }}
              className="rounded-full bg-[#ff3d2e]/15 px-2 py-0.5 text-[#ff8d82] hover:bg-[#ff3d2e]/25"
            >
              seek →
            </button>
          </div>
        </div>
      ))}
    </div>
  );
}

export function EntityList({ entities }: { entities: Entity[] }) {
  if (!entities.length)
    return <div className="rounded-2xl border border-white/[0.07] bg-white/[0.02] p-4 text-[13px] text-[#63615c]">No persistent entities tracked.</div>;
  return (
    <div className="flex flex-wrap gap-1.5">
      {entities.map((e, i) => (
        <span
          key={e.entity_id}
          title={`${e.kind} · conf ${e.confidence}`}
          style={{ animationDelay: `${Math.min(i * 40, 300)}ms` }}
          className="rise rounded-full border border-white/[0.08] bg-white/[0.04] px-3 py-1.5 text-xs text-[#d8d5cf] transition-colors duration-200 hover:border-[#ff3d2e]/40 hover:text-white"
        >
          <span className="font-mono2 mr-1.5 text-[#ff8d82]">{e.entity_id}</span>
          {e.label || e.kind}
        </span>
      ))}
    </div>
  );
}

const TRACE_COLOR: Record<string, string> = {
  sent: "text-sky-300",
  waiting: "text-amber-300",
  received: "text-emerald-300",
  failed: "text-red-300",
};

export function TracePanel({ trace }: { trace: { seq: number; tool: string; event: string; detail: string }[] }) {
  if (!trace.length) return null;
  const tail = trace.slice(-40);
  return (
    <div className="overflow-hidden rounded-2xl border border-white/[0.07] bg-black/60" aria-live="polite">
      <div className="flex items-center gap-2 border-b border-white/[0.06] px-3.5 py-2.5">
        <span className="flex gap-1">
          <i className="h-2 w-2 rounded-full bg-[#ff5f57]" />
          <i className="h-2 w-2 rounded-full bg-[#febc2e]" />
          <i className="h-2 w-2 rounded-full bg-[#28c840]" />
        </span>
        <span className="font-mono2 text-[10px] uppercase tracking-[0.2em] text-[#63615c]">agent trace · live</span>
      </div>
      <ol className="slim-scroll max-h-56 space-y-0.5 overflow-y-auto p-2 font-mono2 text-[11px] leading-relaxed">
        {tail.map((t) => (
          <li key={t.seq} className="fade-in flex gap-2 rounded-lg px-2 py-1 hover:bg-white/[0.04]">
            <span className="shrink-0 text-[#4a4844]">#{t.seq}</span>
            <span className="shrink-0 text-[#d8d5cf]">{t.tool}</span>
            <span className={`shrink-0 font-semibold ${TRACE_COLOR[t.event] ?? "text-slate-300"}`}>{t.event}</span>
            <span className="break-words text-[#7a7874]">{t.detail}</span>
          </li>
        ))}
      </ol>
    </div>
  );
}

export function QaPanel({ sessionId, onSeek }: { sessionId: string; onSeek: (t: number) => void }) {
  const [q, setQ] = useState("");
  const [busy, setBusy] = useState(false);
  const [rec, setRec] = useState<(QARecord & { trace?: unknown }) | null>(null);
  const [err, setErr] = useState<string | null>(null);

  const ask = async () => {
    if (!q.trim() || busy) return;
    setBusy(true); setErr(null);
    try {
      const r = await askQuestion(sessionId, q.trim());
      setRec(r);
    } catch (e) {
      setErr(e instanceof Error ? e.message : "QA failed");
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="rounded-2xl border border-white/[0.07] bg-white/[0.02] p-4">
      <div className="font-mono2 text-[10px] uppercase tracking-[0.2em] text-[#63615c]">Investigator</div>
      <div className="mt-2.5 flex gap-2">
        <input value={q} onChange={(e) => setQ(e.target.value)} onKeyDown={(e) => e.key === "Enter" && ask()}
          placeholder="Did the person put the phone in their pocket?"
          className="min-w-0 flex-1 rounded-xl border border-white/10 bg-black/50 px-3 py-2.5 text-[13px] text-[#f4f1ea] outline-none placeholder:text-[#4a4844] focus:border-[#ff3d2e]/50" />
        <button onClick={ask} disabled={busy || !q.trim()}
          className="pressable btn-primary shrink-0 rounded-xl px-4 py-2 text-[13px] font-semibold disabled:opacity-50">
          {busy ? "…" : "Ask"}
        </button>
      </div>
      {err && <div className="mt-2 text-[13px] text-red-300">{err}</div>}
      {rec && (
        <div className="rise mt-3 space-y-2">
          {rec.abstained && (
            <div className="rounded-xl bg-amber-950/60 p-2.5 text-xs text-amber-200">
              Abstained — insufficient verified evidence ({Math.round(rec.confidence * 100)}% certainty there is none; not factual probability).
            </div>
          )}
          <p className="text-[13.5px] leading-relaxed text-[#f4f1ea]">{rec.answer}</p>
          <div className="font-mono2 text-[11px] text-[#63615c]" title={rec.abstained ? "certainty that no verified evidence exists" : "calibrated factual confidence"}>
            verification: <span className={rec.verification === "supported" ? "text-emerald-300" : "text-amber-300"}>{rec.verification}</span>
            {" · "}{(rec.confidence * 100).toFixed(0)}%{rec.calibration ? ` · ${rec.calibration}` : ""}
          </div>
          {rec.evidence.map((s, i) => (
            <button key={i} onClick={() => onSeek(s.start)} className="font-mono2 block text-[11.5px] text-sky-300 hover:underline">
              evidence [{s.start.toFixed(1)}–{s.end.toFixed(1)}s]
            </button>
          ))}
        </div>
      )}
    </div>
  );
}

export function BenchmarkPanel({ sessionId }: { sessionId: string }) {
  const [busy, setBusy] = useState(false);
  const [tests, setTests] = useState<BenchmarkTest[] | null>(null);
  const [score, setScore] = useState("");

  const run = async () => {
    setBusy(true);
    try {
      const r = await runBenchmark(sessionId);
      setTests(r.tests);
      setScore(`${r.passed}/${r.total} · ablation L${r.ablation.level}`);
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="rounded-2xl border border-white/[0.07] bg-white/[0.02] p-4">
      <div className="flex items-center justify-between">
        <div className="font-mono2 text-[10px] uppercase tracking-[0.2em] text-[#63615c]">Adversarial eval</div>
        <button onClick={run} disabled={busy}
          className="pressable rounded-full border border-white/10 bg-white/[0.05] px-3.5 py-1.5 text-xs text-white disabled:opacity-50 hover:border-white/20">
          {busy ? "running…" : score ? "Re-run" : "Run 11 tests"}
        </button>
      </div>
      {score && <div className="font-mono2 mt-2 text-xs text-[#d8d5cf]">{score}</div>}
      {tests && (
        <ul className="mt-2 space-y-1">
          {tests.map((t) => (
            <li key={t.test} className="flex gap-2 text-xs">
              <span className={t.result === "pass" ? "text-emerald-300" : t.result === "fail" ? "text-red-300" : "text-slate-500"}>
                {t.result === "pass" ? "✓" : t.result === "fail" ? "✗" : "○"}
              </span>
              <span className="text-[#d8d5cf]">{t.test.replace(/_/g, " ")}</span>
              <span className="text-[#63615c]">— {t.detail}</span>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
