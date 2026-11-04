"use client";

import { useRef, useState } from "react";
import { uploadClip } from "@/lib/api";
import type { VideoSessionPayload } from "@/lib/types";

interface Props {
  busy: boolean;
  status: string;
  onSubmit: (payload: VideoSessionPayload, playSrc: string) => void;
}

const PRESETS: { label: string; focus: string }[] = [
  { label: "Summary", focus: "Summarize what happens, who is present, and the key takeaways." },
  { label: "Meeting", focus: "Who spoke, what was decided, and what are the action items with timestamps?" },
  { label: "Lecture", focus: "Extract the main concepts, definitions, and any on-screen text or slides." },
  { label: "Detailed", focus: "Dense timestamped breakdown of every scene change, action, and spoken line." },
];

export function VideoConfigSidebar({ busy, status, onSubmit }: Props) {
  const [form, setForm] = useState<VideoSessionPayload>({
    title: "Untitled video",
    focus: PRESETS[0].focus,
    video_url: "",
    fps: 30,
  });
  const [playSrc, setPlaySrc] = useState("");
  const [fileName, setFileName] = useState("");
  const [uploading, setUploading] = useState(false);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement | null>(null);

  const set = (k: keyof VideoSessionPayload) => (e: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement>) => {
    const v = e.target.value;
    setForm((f) => ({ ...f, [k]: k === "fps" ? Number(v) || 30 : v }));
  };

  const handleFiles = async (files: FileList | File[] | null) => {
    const file = files?.[0];
    if (!file) return;
    if (!file.type.startsWith("video/")) {
      setUploadError("Please choose a video file (MP4).");
      return;
    }
    setUploading(true);
    setUploadError(null);
    try {
      const { video_url } = await uploadClip(file);
      setForm((f) => ({ ...f, video_url }));
      setPlaySrc((prev) => {
        if (prev.startsWith("blob:")) URL.revokeObjectURL(prev);
        return URL.createObjectURL(file);
      });
      setFileName(file.name);
    } catch (err) {
      setUploadError(err instanceof Error ? err.message : "Upload failed");
    } finally {
      setUploading(false);
      if (inputRef.current) inputRef.current.value = "";
    }
  };

  const ready = form.video_url.trim().length > 0 && !uploading;

  return (
    <aside className="space-y-4 rounded-xl border border-slate-800 bg-slate-900 p-5">
      <h2 className="text-sm font-semibold uppercase tracking-wider text-slate-400">Video intel</h2>
      <div className="flex flex-wrap gap-2">
        {PRESETS.map((p) => (
          <button
            key={p.label}
            type="button"
            onClick={() => setForm((f) => ({ ...f, focus: p.focus }))}
            className="rounded-full border border-slate-700 px-3 py-1 text-xs text-slate-300 hover:bg-slate-800"
          >
            {p.label}
          </button>
        ))}
      </div>
      <label className="block text-xs text-slate-400">
        Title
        <input value={form.title} onChange={set("title")} className="mt-1 w-full rounded bg-slate-800 p-2 text-sm text-white" />
      </label>
      <label className="block text-xs text-slate-400">
        Question or focus
        <textarea value={form.focus} onChange={set("focus")} rows={4} className="mt-1 w-full rounded bg-slate-800 p-2 text-sm text-white" />
      </label>
      <div className="block text-xs text-slate-400">
        <span>Video clip (MP4 · max 30min)</span>
        <div
          onClick={() => !busy && !uploading && inputRef.current?.click()}
          className="mt-1 cursor-pointer rounded-lg border border-dashed border-slate-700 bg-slate-800/50 p-4 text-center hover:border-slate-500"
        >
          <input ref={inputRef} type="file" accept="video/mp4,video/*"
            onChange={(e) => void handleFiles(e.target.files)} disabled={busy || uploading} className="hidden" />
          {form.video_url ? (
            <div className="space-y-2" onClick={(e) => e.stopPropagation()}>
              <div className="truncate text-sm text-white">{fileName || "Clip ready"}</div>
              <button type="button" onClick={() => { setForm((f) => ({ ...f, video_url: "" })); setPlaySrc(""); setFileName(""); }}
                className="rounded-full border border-slate-600 px-3 py-1 text-xs text-slate-300">Remove</button>
            </div>
          ) : (
            <div className="text-sm text-slate-200">{uploading ? "Uploading…" : "Drop MP4 or browse"}</div>
          )}
        </div>
      </div>
      {uploadError && <div className="text-xs text-red-300">{uploadError}</div>}
      <button onClick={() => ready && onSubmit(form, playSrc || form.video_url)} disabled={busy || !ready}
        className="w-full rounded-lg bg-red-600 p-3 text-sm font-semibold hover:bg-red-500 disabled:opacity-50">
        {busy ? `Analyzing… (${status})` : "Analyze video"}
      </button>
      {!busy && !ready && <div className="text-xs text-slate-500">Upload a clip to enable analysis. Real models only — no demo data.</div>}
    </aside>
  );
}
