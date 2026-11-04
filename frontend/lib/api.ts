import type { BenchmarkTest, Entity, EvidenceSpan, GraphEvent, Observation, QARecord, SessionProgress, SessionReport, VideoSessionPayload } from "./types";

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

/** Subscribe to the session SSE stream. Returns an unsubscribe function. */
export function subscribeSession(
  sessionId: string,
  onEvent: (e: { status: string; stage?: string; detail?: string; error?: string | null; done?: boolean }) => void,
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

export async function createSession(payload: VideoSessionPayload): Promise<string> {
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

export async function fetchProgress(sessionId: string): Promise<SessionProgress | null> {
  const res = await fetch(`${API_BASE}/api/v1/sessions/${sessionId}/progress`, { cache: "no-store" });
  if (res.status === 404) return null;
  if (!res.ok) throw new Error(`GET /progress failed: ${res.status}`);
  return (await res.json()) as SessionProgress;
}

/** Poll until completed/failed or timeout. Backend uses BackgroundTasks (no SSE yet). */
export async function pollSession(
  sessionId: string,
  opts: { intervalMs?: number; timeoutMs?: number; onUpdate?: (r: SessionReport) => void } = {},
): Promise<SessionReport> {
  const { intervalMs = 1500, timeoutMs = 900000, onUpdate } = opts;
  const deadline = Date.now() + timeoutMs;
  for (;;) {
    const report = await fetchSession(sessionId);
    onUpdate?.(report);
    if (report.status === "completed" || report.status === "failed") return report;
    if (Date.now() > deadline) throw new Error("Timed out waiting for session report");
    await new Promise((r) => setTimeout(r, intervalMs));
  }
}

export async function fetchEvents(sessionId: string): Promise<GraphEvent[]> {
  const res = await fetch(`${API_BASE}/api/v1/sessions/${sessionId}/events`, { cache: "no-store" });
  if (!res.ok) throw new Error(`GET /events failed: ${res.status}`);
  return ((await res.json()).events ?? []) as GraphEvent[];
}

export async function fetchEntities(sessionId: string): Promise<{ entities: Entity[]; observations: Observation[] }> {
  const res = await fetch(`${API_BASE}/api/v1/sessions/${sessionId}/entities`, { cache: "no-store" });
  if (!res.ok) throw new Error(`GET /entities failed: ${res.status}`);
  const body = await res.json();
  return { entities: body.entities ?? [], observations: body.observations ?? [] };
}

export interface TraceRow {
  seq: number;
  tool: string;
  event: string;
  detail: string;
  created_at: number;
}

export async function fetchTrace(sessionId: string): Promise<TraceRow[]> {
  const res = await fetch(`${API_BASE}/api/v1/sessions/${sessionId}/trace`, { cache: "no-store" });
  if (res.status === 404) return [];
  if (!res.ok) throw new Error(`GET /trace failed: ${res.status}`);
  return ((await res.json()).trace ?? []) as TraceRow[];
}

export async function askQuestion(sessionId: string, question: string): Promise<QARecord & { trace?: unknown }> {
  const res = await fetch(`${API_BASE}/api/v1/sessions/${sessionId}/qa`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question }),
  });
  if (!res.ok) throw new Error(`POST /qa failed: ${res.status} ${await res.text()}`);
  return (await res.json()) as QARecord & { trace?: unknown };
}

export async function listQa(sessionId: string): Promise<QARecord[]> {
  const res = await fetch(`${API_BASE}/api/v1/sessions/${sessionId}/qa`, { cache: "no-store" });
  if (!res.ok) throw new Error(`GET /qa failed: ${res.status}`);
  return ((await res.json()).qa ?? []) as QARecord[];
}

export async function runBenchmark(sessionId: string): Promise<{ passed: number; total: number; tests: BenchmarkTest[]; ablation: { level: number } }> {
  const res = await fetch(`${API_BASE}/api/v1/sessions/${sessionId}/benchmark`, { method: "POST" });
  if (!res.ok) throw new Error(`POST /benchmark failed: ${res.status}`);
  return (await res.json()) as { passed: number; total: number; tests: BenchmarkTest[]; ablation: { level: number } };
}
