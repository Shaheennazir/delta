"use client";

import { useState } from "react";
import { askQuestion, runBenchmark } from "@/lib/api";
import type { BenchmarkTest, Entity, GraphEvent, QARecord } from "@/lib/types";

export function EventList({ events, onSeek }: { events: GraphEvent[]; onSeek: (t: number) => void }) {
  if (!events.length) return <div className="rounded-lg bg-slate-900 p-3 text-sm text-slate-400">No events — proposal found nothing grounded, or tracking was empty.</div>;
  return (
    <div className="space-y-2">
      {events.map((e) => {
        const spans = e.evidence ?? [];
        const fids = spans.flatMap((s) => s.frame_ids ?? []).slice(0, 4);
        return (
          <div key={e.event_id} onClick={() => onSeek(e.start_t)} role="button" tabIndex={0}
            onKeyDown={(ev) => { if (ev.key === "Enter") onSeek(e.start_t); }}
            className="w-full cursor-pointer rounded-lg bg-slate-900 p-3 text-left text-sm hover:bg-slate-800">
            <div className="flex items-center justify-between gap-2">
              <span className="font-mono text-xs text-emerald-300">{e.event_id}</span>
              <span className="text-xs text-slate-400">[{e.start_t.toFixed(1)}–{e.end_t.toFixed(1)}s] · {(e.confidence * 100).toFixed(0)}%</span>
            </div>
            <div className="mt-1 font-medium text-slate-100">{e.action}</div>
            <div className="mt-0.5 text-xs text-slate-400">{e.detail}</div>
            {e.entity_ids.length > 0 && (
              <div className="mt-1 text-[11px] text-slate-500">entities: {e.entity_ids.join(", ")}</div>
            )}
            <div className="mt-1 flex flex-wrap items-center gap-2 text-[11px]">
              {e.source && (
                <span className="rounded-full bg-slate-800 px-2 py-0.5 text-slate-400">{e.source}</span>
              )}
              {fids.length > 0 ? (
                <span className="text-slate-500">
                  frames {fids.join(", ")}{" "}
                  <button
                    onClick={(ev) => { ev.stopPropagation(); onSeek(e.start_t); }}
                    className="text-sky-300 hover:underline">
                    seek
                  </button>
                </span>
              ) : (
                <span className="text-slate-600">no retrievable frames</span>
              )}
            </div>
          </div>
        );
      })}
    </div>
  );
}

export function EntityList({ entities }: { entities: Entity[] }) {
  if (!entities.length) return <div className="rounded-lg bg-slate-900 p-3 text-sm text-slate-400">No persistent entities tracked.</div>;
  return (
    <div className="flex flex-wrap gap-1.5">
      {entities.map((e) => (
        <span key={e.entity_id} title={`${e.kind} · conf ${e.confidence}`}
          className="rounded-full bg-slate-800 px-3 py-1 text-xs text-slate-200">
          {e.entity_id} · {e.label || e.kind}
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
  return (
    <div className="rounded-xl border border-slate-800 bg-slate-950 p-4 text-sm" aria-live="polite">
      <div className="text-xs uppercase tracking-wider text-slate-400">Agent tools — sent · waiting · received</div>
      <ol className="mt-2 max-h-64 space-y-1 overflow-y-auto font-mono text-[11px] leading-relaxed">
        {trace.map((t) => (
          <li key={t.seq} className="flex gap-2">
            <span className="shrink-0 text-slate-600">#{t.seq}</span>
            <span className="shrink-0 text-slate-200">{t.tool}</span>
            <span className={`shrink-0 font-semibold ${TRACE_COLOR[t.event] ?? "text-slate-300"}`}>{t.event}</span>
            <span className="break-words text-slate-400">{t.detail}</span>
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
    <div className="rounded-xl bg-slate-900 p-4 text-sm">
      <div className="text-xs uppercase tracking-wider text-slate-400">Investigator — ask about the video</div>
      <div className="mt-2 flex gap-2">
        <input value={q} onChange={(e) => setQ(e.target.value)} onKeyDown={(e) => e.key === "Enter" && ask()}
          placeholder="Did the person put the phone in their pocket?"
          className="flex-1 rounded-lg bg-slate-800 px-3 py-2 text-sm text-slate-100 placeholder:text-slate-500" />
        <button onClick={ask} disabled={busy || !q.trim()}
          className="rounded-lg bg-red-600 px-4 py-2 text-sm font-medium text-white disabled:opacity-50">
          {busy ? "…" : "Ask"}
        </button>
      </div>
      {err && <div className="mt-2 text-red-300">{err}</div>}
      {rec && (
        <div className="mt-3 space-y-2">
          {rec.abstained && (
            <div className="rounded-lg bg-amber-950 p-2 text-xs text-amber-200">
              Abstained — insufficient verified evidence.
            </div>
          )}
          <p className="text-slate-100">{rec.answer}</p>
          <div className="text-xs text-slate-400">
            verification: <span className={rec.verification === "supported" ? "text-emerald-300" : "text-amber-300"}>{rec.verification}</span>
            {" · "}confidence {(rec.confidence * 100).toFixed(0)}%
            {rec.calibration && ` · ${rec.calibration}`}
          </div>
          {rec.evidence.map((s, i) => (
            <button key={i} onClick={() => onSeek(s.start)} className="block text-xs text-sky-300 hover:underline">
              evidence [{s.start.toFixed(1)}–{s.end.toFixed(1)}s]{s.frame_ids.length ? ` · frames ${s.frame_ids.slice(0, 4).join(", ")}` : ""}
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
      setScore(`${r.passed}/${r.total} · ablation level ${r.ablation.level}`);
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="rounded-xl bg-slate-900 p-4 text-sm">
      <div className="flex items-center justify-between">
        <div className="text-xs uppercase tracking-wider text-slate-400">Adversarial evaluation</div>
        <button onClick={run} disabled={busy} className="rounded-lg bg-slate-700 px-3 py-1 text-xs text-white disabled:opacity-50">
          {busy ? "…" : score ? "Re-run" : "Run 11 tests"}
        </button>
      </div>
      {score && <div className="mt-1 text-xs text-slate-300">{score}</div>}
      {tests && (
        <ul className="mt-2 space-y-1">
          {tests.map((t) => (
            <li key={t.test} className="flex gap-2 text-xs">
              <span className={t.result === "pass" ? "text-emerald-300" : t.result === "fail" ? "text-red-300" : "text-slate-500"}>
                {t.result === "pass" ? "✓" : t.result === "fail" ? "✗" : "○"}
              </span>
              <span className="text-slate-200">{t.test.replace(/_/g, " ")}</span>
              <span className="text-slate-500">— {t.detail}</span>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
