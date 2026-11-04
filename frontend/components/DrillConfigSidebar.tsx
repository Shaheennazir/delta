"use client";

import { useState } from "react";
import { uploadClip } from "@/lib/api";
import type { CoachSessionPayload } from "@/lib/types";

interface Props {
  busy: boolean;
  status: string;
  onSubmit: (payload: CoachSessionPayload, playSrc: string) => void;
}

const PRESETS: { label: string; setup: string; goals: string }[] = [
  {
    label: "4v2 Rondo",
    setup: "4 attackers on the perimeter, 2 defenders pressing inside a 10x10m grid. Two-touch maximum.",
    goals: "Flag whenever an attacker holds the ball longer than 1.0 second, and detect when a pass splits the two defenders.",
  },
  {
    label: "Shooting",
    setup: "Single striker 18m from goal, feeder wide. First-time finish after a cutback.",
    goals: "Flag plant foot distance over 0.2m and torso lean over 20 degrees at ball strike.",
  },
  {
    label: "1v1 Sprint",
    setup: "Attacker vs defender over 20m channel. Attacker attacks space on the whistle.",
    goals: "Flag attacker sprint speed over 7 m/s and defender separation under 1.5m.",
  },
];

export function DrillConfigSidebar({ busy, status, onSubmit }: Props) {
  const [form, setForm] = useState<CoachSessionPayload>({
    drill_title: "4v2 High-Tempo Rondo",
    drill_setup: PRESETS[0].setup,
    analysis_goals: PRESETS[0].goals,
    video_url: "",
    fps: 30,
  });
  const [playSrc, setPlaySrc] = useState("");
  const [uploading, setUploading] = useState(false);
  const [uploadError, setUploadError] = useState<string | null>(null);

  const set = (k: keyof CoachSessionPayload) => (e: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement>) => {
    const v = e.target.value;
    setForm((f) => ({ ...f, [k]: k === "fps" ? Number(v) || 30 : v }));
    if (k === "video_url") setPlaySrc(v);
  };

  const handleFile = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    setUploading(true);
    setUploadError(null);
    try {
      const { video_url } = await uploadClip(file);
      setForm((f) => ({ ...f, video_url }));
      setPlaySrc(URL.createObjectURL(file));
    } catch (err) {
      setUploadError(err instanceof Error ? err.message : "Upload failed");
    } finally {
      setUploading(false);
    }
  };

  const ready = form.video_url.trim().length > 0 && !uploading;

  return (
    <aside className="space-y-4 rounded-xl border border-slate-800 bg-slate-900 p-5">
      <h2 className="text-sm font-semibold uppercase tracking-wider text-slate-400">Drill builder</h2>
      <div className="flex flex-wrap gap-2">
        {PRESETS.map((p) => (
          <button
            key={p.label}
            type="button"
            onClick={() => setForm((f) => ({ ...f, drill_title: p.label, drill_setup: p.setup, analysis_goals: p.goals }))}
            className="rounded-full border border-slate-700 px-3 py-1 text-xs text-slate-300 hover:bg-slate-800"
          >
            {p.label}
          </button>
        ))}
      </div>
      <label className="block text-xs text-slate-400">
        Drill title
        <input value={form.drill_title} onChange={set("drill_title")} className="mt-1 w-full rounded bg-slate-800 p-2 text-sm text-white" />
      </label>
      <label className="block text-xs text-slate-400">
        Drill setup
        <textarea value={form.drill_setup} onChange={set("drill_setup")} rows={3} className="mt-1 w-full rounded bg-slate-800 p-2 text-sm text-white" />
      </label>
      <label className="block text-xs text-slate-400">
        Coaching focus (analysis goals)
        <textarea value={form.analysis_goals} onChange={set("analysis_goals")} rows={3} className="mt-1 w-full rounded bg-slate-800 p-2 text-sm text-white" />
      </label>
      <label className="block text-xs text-slate-400">
        Clip upload (MP4, max 60s, max 1080p)
        <input type="file" accept="video/mp4" onChange={handleFile} disabled={busy} className="mt-1 w-full text-xs text-slate-300" />
      </label>
      {uploading && <div className="text-xs text-slate-400">Uploading…</div>}
      {uploadError && <div className="text-xs text-red-300">{uploadError}</div>}
      <label className="block text-xs text-slate-400">
        Video URL <span className="text-slate-500">(or paste a remote MP4 URL)</span>
        <input value={form.video_url} onChange={set("video_url")} placeholder="upload://… or https://…" className="mt-1 w-full rounded bg-slate-800 p-2 text-sm text-white" />
      </label>
      <label className="block text-xs text-slate-400">
        FPS
        <input type="number" value={form.fps} onChange={set("fps")} min={1} max={120} className="mt-1 w-full rounded bg-slate-800 p-2 text-sm text-white" />
      </label>
      <button
        onClick={() => ready && onSubmit(form, playSrc || form.video_url)}
        disabled={busy || !ready}
        className="w-full rounded-lg bg-red-600 p-3 text-sm font-semibold hover:bg-red-500 disabled:opacity-50"
      >
        {busy ? `Analyzing… (${status})` : "Analyze drill"}
      </button>
    </aside>
  );
}
