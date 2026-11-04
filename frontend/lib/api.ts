import type { CoachSessionPayload, SegmentationPayload, SessionReport } from "./types";

const API_BASE =
  process.env.NEXT_PUBLIC_API_BASE?.replace(/\/$/, "") ?? "http://localhost:8000";

export async function uploadClip(file: File): Promise<{ video_url: string; meta: unknown }> {
  const form = new FormData();
  form.append("file", file);
  const res = await fetch(`${API_BASE}/api/v1/uploads`, { method: "POST", body: form });
  if (!res.ok) throw new Error(`Upload failed: ${res.status} ${await res.text()}`);
  const body = await res.json();
  return { video_url: body.video_url as string, meta: body.meta };
}

export async function fetchSegmentation(sessionId: string): Promise<SegmentationPayload> {
  const res = await fetch(`${API_BASE}/api/v1/sessions/${sessionId}/segmentation`, { cache: "no-store" });
  if (!res.ok) throw new Error(`Segmentation not ready: ${res.status}`);
  return (await res.json()) as SegmentationPayload;
}

/** Subscribe to the session SSE stream. Returns an unsubscribe function. */
export function subscribeSession(
  sessionId: string,
  onEvent: (e: { status: string; roles?: string[]; done?: boolean }) => void,
  onError?: (err: unknown) => void,
): () => void {
  const src = new EventSource(`${API_BASE}/api/v1/sessions/${sessionId}/stream`);
  src.onmessage = (msg) => {
    try {
      onEvent(JSON.parse(msg.data));
    } catch (err) {
      onError?.(err);
    }
  };
  src.onerror = (err) => onError?.(err);
  return () => src.close();
}

export async function createSession(payload: CoachSessionPayload): Promise<string> {
  const res = await fetch(`${API_BASE}/api/v1/sessions`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!res.ok) throw new Error(`POST /sessions failed: ${res.status}`);
  const body = await res.json();
  return body.session_id as string;
}

export async function fetchSession(sessionId: string): Promise<SessionReport> {
  const res = await fetch(`${API_BASE}/api/v1/sessions/${sessionId}`, { cache: "no-store" });
  if (!res.ok) throw new Error(`GET /sessions failed: ${res.status}`);
  return (await res.json()) as SessionReport;
}

/** Poll until completed/failed or timeout. Backend uses BackgroundTasks (no SSE yet). */
export async function pollSession(
  sessionId: string,
  opts: { intervalMs?: number; timeoutMs?: number; onUpdate?: (r: SessionReport) => void } = {},
): Promise<SessionReport> {
  const { intervalMs = 1500, timeoutMs = 90000, onUpdate } = opts;
  const deadline = Date.now() + timeoutMs;
  for (;;) {
    const report = await fetchSession(sessionId);
    onUpdate?.(report);
    if (report.status === "completed" || report.status === "failed") return report;
    if (Date.now() > deadline) throw new Error("Timed out waiting for session report");
    await new Promise((r) => setTimeout(r, intervalMs));
  }
}
