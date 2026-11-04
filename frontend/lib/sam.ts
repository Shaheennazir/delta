import { formats } from "@meta-sam/parser";
import type { VideoSegmentationResult } from "@meta-sam/parser";

/** Parse one raw output-text lane (from /segmentation) into a result. */
export function parseLane(laneText: string): VideoSegmentationResult | null {
  if (!laneText) return null;
  const parser = formats.segmentation.video().createParser();
  parser.push(laneText);
  const { result } = parser.finish({ status: "completed" });
  return result.records.length > 0 ? (result as VideoSegmentationResult) : null;
}

/**
 * Merge per-role results into one renderable result. Object ids are
 * namespaced per role ("ball:0") because the same wire id in two streams
 * names different objects; the renderer keys overlays by id.
 */
export function mergeResults(byRole: Record<string, VideoSegmentationResult>): VideoSegmentationResult | null {
  const roles = Object.keys(byRole);
  if (roles.length === 0) return null;
  if (roles.length === 1) return byRole[roles[0]];
  const base = byRole[roles[0]];
  const records = roles.flatMap((role) =>
    byRole[role].records.map((rec: any) => ({ ...rec, object_id: `${role}:${rec.object_id}` })),
  );
  return { ...base, records, revision: Math.max(...roles.map((r) => byRole[r].revision)) } as VideoSegmentationResult;
}

/** Parse all lanes of a /segmentation payload and merge for <Video result>. */
export function resultsFromSegmentation(lanes: Record<string, string>): VideoSegmentationResult | null {
  const byRole: Record<string, VideoSegmentationResult> = {};
  for (const [role, lane] of Object.entries(lanes)) {
    try {
      const parsed = parseLane(lane);
      if (parsed) byRole[role] = parsed;
    } catch {
      /* keep other roles on a corrupt lane */
    }
  }
  return mergeResults(byRole);
}
