"""Persistent object identity: instance-level descriptors + cross-session gallery.

Each SAM object (not each concept stream) becomes one candidate. Appearance is
summarised by a spatial colour + hue + gradient-orientation descriptor computed
from the stored frame crop. Candidates are matched to a persistent global_id by
cosine similarity within the same kind; matches update a count-weighted running
descriptor in sqlite, so identity survives restarts and new clips.

No identity is claimed without a descriptor: if frames/boxes are missing the
candidate gets a fresh global_id and no gallery assertion.
"""
from __future__ import annotations

import os
import uuid

import numpy as np

GRID = 4
HUE_BINS = 12
ORIENT_BINS = 8
MATCH_THRESHOLD = 0.86
# Gap-bridging (occlusion recovery): looser than MATCH_THRESHOLD because
# lighting/pose shifts across an occlusion, but still well above chance.
BRIDGE_APPEARANCE_THR = 0.75
BRIDGE_MAX_GAP_S = 5.0
# Long gaps need stronger appearance evidence; scene-cut crossings need more.
LONG_GAP_S = 2.5
LONG_GAP_APPEARANCE_THR = 0.85
CUT_APPEARANCE_THR = 0.90


def descriptor(path: str, bbox: list[float], media_w: float = 0.0) -> list[float]:
    """Unit-norm appearance descriptor for one box crop. [] when unusable."""
    try:
        from PIL import Image
        img = Image.open(path).convert("RGB")
    except Exception:
        return []
    if img.width < 4 or img.height < 4:
        return []
    scale = (img.width / float(media_w)) if media_w else 1.0
    x1, y1, x2, y2 = (float(v) * scale for v in bbox[:4])
    pad_x, pad_y = 0.1 * (x2 - x1), 0.1 * (y2 - y1)
    x1 = int(max(0, min(img.width - 2, x1 - pad_x)))
    y1 = int(max(0, min(img.height - 2, y1 - pad_y)))
    x2 = int(max(x1 + 2, min(img.width, x2 + pad_x)))
    y2 = int(max(y1 + 2, min(img.height, y2 + pad_y)))
    crop = img.crop((x1, y1, x2, y2))
    if crop.width < 2 or crop.height < 2:
        return []
    rgb = np.asarray(crop.resize((32, 32)), dtype=np.float32) / 255.0
    hsv = np.asarray(crop.convert("HSV").resize((32, 32)), dtype=np.float32) / 255.0

    # 4x4 spatial colour means.
    cells = []
    step = 32 // GRID
    for gy in range(GRID):
        for gx in range(GRID):
            cells.extend(rgb[gy * step:(gy + 1) * step, gx * step:(gx + 1) * step]
                         .reshape(-1, 3).mean(axis=0).tolist())
    # Saturation-weighted hue histogram (ignores washed-out background).
    hue = np.minimum((hsv[..., 0] * HUE_BINS).astype(int), HUE_BINS - 1)
    weights = hsv[..., 1].reshape(-1)
    hist = np.bincount(hue.reshape(-1), weights=weights, minlength=HUE_BINS)
    hist = hist / max(1e-6, hist.sum())
    # Gradient orientation histogram (shape cue).
    gray = rgb.mean(axis=2)
    gx = np.diff(gray, axis=1)
    gy = np.diff(gray, axis=0)
    mag = np.hypot(gx[:-1, :], gy[:, :-1])
    ang = (np.arctan2(gy[:, :-1], gx[:-1, :]) + np.pi) / (2 * np.pi)
    obin = np.minimum((ang * ORIENT_BINS).astype(int), ORIENT_BINS - 1)
    ohist = np.bincount(obin.reshape(-1), weights=mag.reshape(-1), minlength=ORIENT_BINS)
    ohist = ohist / max(1e-6, ohist.sum())

    vec = np.concatenate([np.asarray(cells, dtype=np.float64), hist, ohist * 0.5])
    norm = float(np.linalg.norm(vec))
    return (vec / norm).round(6).tolist() if norm > 1e-9 else []


def cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    x, y = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    den = float(np.linalg.norm(x) * np.linalg.norm(y))
    return float(x @ y / den) if den > 1e-9 else 0.0


def match(desc: list[float], kind: str, gallery: list[dict],
          threshold: float = MATCH_THRESHOLD) -> dict | None:
    """Best same-kind gallery match above threshold."""
    if not desc:
        return None
    best, best_score = None, 0.0
    for g in gallery:
        if (g.get("kind") or "object") != (kind or "object"):
            continue
        score = cosine(desc, g.get("descriptor") or [])
        if score > best_score:
            best, best_score = g, score
    if best is None or best_score < threshold:
        return None
    return {**best, "score": round(best_score, 4)}


def candidate_descriptor(observations: list[dict], frame_rows: list[dict],
                         media_w: float) -> list[float]:
    """Average descriptor over the largest visible boxes (up to 4 frames)."""
    by_frame = {int(r["frame_index"]): r.get("path", "") for r in frame_rows}
    ranked = sorted(observations, key=lambda o: -(
        (o["bbox"][2] - o["bbox"][0]) * (o["bbox"][3] - o["bbox"][1])))[:4]
    vecs = []
    for o in ranked:
        path = by_frame.get(int(o.get("frame_index", -1)))
        if path and os.path.exists(path):
            d = descriptor(path, o["bbox"], media_w)
            if d:
                vecs.append(np.asarray(d, dtype=np.float64))
    if not vecs:
        return []
    mean = np.mean(vecs, axis=0)
    norm = float(np.linalg.norm(mean))
    return (mean / norm).round(6).tolist() if norm > 1e-9 else []


def assign(candidates: list[dict], frame_rows: list[dict], media_wh: tuple,
           session_id: str) -> list[dict]:
    """Attach persistent global_id to each instance candidate; update gallery.

    SAM often fragments one physical object into several tracklets across an
    occlusion gap. After gallery matching, tracklets with compatible
    appearance + spatio-temporal continuity are bridged to one global_id,
    so reappearance resumes the same identity instead of forking a new one.
    """
    import store
    gallery = store.get_gallery()
    mw = float(media_wh[0] or 0)
    for c in candidates:
        c["_desc"] = candidate_descriptor(c["observations"], frame_rows, mw)
        hit = match(c["_desc"], c.get("kind", "object"), gallery)
        if hit:
            gid = hit["global_id"]
            c["identity_score"] = hit["score"]
        else:
            gid = f"g_{uuid.uuid4().hex[:10]}"
            c["identity_score"] = 0.0
        c["entity_id"] = gid
        c["global_id"] = gid
    # Conflicting simultaneous tracks must never share one identity: if two
    # candidates overlap in time and matched the same gallery id, only the
    # strongest keeps it; the rest fork to unknown ids rather than forcing.
    def _tspan(c: dict) -> tuple[float, float]:
        ts = [float(o["t"]) for o in c.get("observations", [])]
        return (min(ts), max(ts)) if ts else (0.0, 0.0)

    def _overlaps(a: dict, b: dict) -> bool:
        sa, ea = _tspan(a)
        sb, eb = _tspan(b)
        fa = {int(o.get("frame_index", -1)) for o in a.get("observations", [])}
        fb = {int(o.get("frame_index", -1)) for o in b.get("observations", [])}
        if fa & fb:
            return True
        return max(sa, sb) <= min(ea, eb) + 1e-6

    seen_gid: dict[str, dict] = {}
    for c in sorted(candidates, key=lambda x: -float(x.get("identity_score", 0.0))):
        gid = c.get("global_id", "")
        if gid in seen_gid and _overlaps(seen_gid[gid], c):
            gid = f"g_{uuid.uuid4().hex[:10]}"
            c["identity_score"] = 0.0
            c["entity_id"] = gid
            c["global_id"] = gid
        seen_gid.setdefault(c.get("global_id", ""), c)
    cuts = [float(r.get("t", 0)) for r in (frame_rows or [])
            if r.get("scene_cut") or r.get("tier") == "transition"]
    candidates = stitch_tracklets(candidates, scene_cut_times=cuts or None)
    for c in candidates:
        for o in c["observations"]:
            o["entity_id"] = c["global_id"]
        desc = c.pop("_desc", [])
        if desc:
            store.upsert_gallery(c["global_id"], c.get("label", ""), c.get("kind", "object"),
                                 desc, float(c.get("confidence", 0.0)))
        else:
            # No visual descriptor: keep a kind-scoped row so the id is durable,
            # but do not let an empty vector match anything.
            store.upsert_gallery(c["global_id"], c.get("label", ""), c.get("kind", "object"),
                                 [], float(c.get("confidence", 0.0)))
    return candidates


def _box_iou(a: list[float], b: list[float]) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    aa = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    bb = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return inter / max(1e-9, aa + bb - inter)


def _center(b: list[float]) -> tuple[float, float]:
    return ((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0)


def _diag(b: list[float]) -> float:
    return max(1.0, ((b[2] - b[0]) ** 2 + (b[3] - b[1]) ** 2) ** 0.5)


def resume_plausible(before: list[float], after: list[float], dt: float,
                     max_gap: float = BRIDGE_MAX_GAP_S) -> bool:
    """Spatial continuity across a track gap: same scale, reachable position."""
    if dt <= 0 or dt > max_gap:
        return False
    cbx, cby = _center(before)
    cax, cay = _center(after)
    avg_diag = (_diag(before) + _diag(after)) / 2.0
    drift = (((cax - cbx) ** 2 + (cay - cby) ** 2) ** 0.5) / max(1e-9, avg_diag)
    # Allow ~1 diag of drift per second plus 2 diags of slack for re-detection.
    if drift > 2.0 + 1.0 * dt:
        return False
    size_ratio = _diag(after) / max(1e-9, _diag(before))
    return 0.33 <= size_ratio <= 3.0


def stitch_tracklets(candidates: list[dict],
                     max_gap: float = BRIDGE_MAX_GAP_S,
                     appearance_thr: float = BRIDGE_APPEARANCE_THR,
                     scene_cut_times: list[float] | None = None) -> list[dict]:
    """Merge fragmented tracklets of one physical object across occlusion gaps.

    Constrained association: a merge is allowed only when ALL hold —
    temporally disjoint (no shared frame, no time overlap), spatially
    reachable resume, compatible appearance, no conflicting simultaneous
    track, no scene-cut crossing without strong independent evidence, and no
    excessively long gap without strong evidence. Otherwise the fragment keeps
    its own (unknown) identity. Deterministic: earliest-first, greedy.
    """
    for c in candidates:
        c["observations"] = sorted(c.get("observations", []), key=lambda o: float(o["t"]))

    def _span(c: dict) -> tuple[float, float]:
        ts = [float(o["t"]) for o in c.get("observations", [])]
        return (min(ts), max(ts)) if ts else (0.0, 0.0)

    def _frames(c: dict) -> set[int]:
        return {int(o.get("frame_index", -1)) for o in c.get("observations", [])}

    def _overlaps(a: dict, b: dict) -> bool:
        sa, ea = _span(a)
        sb, eb = _span(b)
        if _frames(a) & _frames(b):
            return True
        return max(sa, sb) <= min(ea, eb) + 1e-6

    def _cut_in_gap(pe: float, cs: float) -> bool:
        if not scene_cut_times:
            return False
        return any(pe < float(t) <= cs for t in scene_cut_times)

    def _would_conflict(prev: dict, cur: dict) -> bool:
        # One identity in two places at once: merged track would hold two
        # different boxes on the same frame.
        pf = {int(o.get("frame_index", -1)): tuple(o["bbox"])
              for o in prev.get("observations", [])}
        for o in cur.get("observations", []):
            fi = int(o.get("frame_index", -1))
            if fi in pf and tuple(o["bbox"]) != pf[fi]:
                return True
        return False

    def _can_merge(prev: dict, cur: dict) -> bool:
        if (prev.get("kind", "object") or "object") != (cur.get("kind", "object") or "object"):
            return False
        if _overlaps(prev, cur):
            return False  # conflicting simultaneous tracks: never the same identity
        if _would_conflict(prev, cur):
            return False
        pe = float(prev["observations"][-1]["t"])
        cs = float(cur["observations"][0]["t"])
        gap = cs - pe
        if gap <= 0 or gap > max_gap:
            return False  # excessively long gap without supporting evidence
        da, db = prev.get("_desc") or [], cur.get("_desc") or []
        app = cosine(da, db) if (da and db) else 0.0
        spatial = resume_plausible(prev["observations"][-1]["bbox"],
                                   cur["observations"][0]["bbox"], gap, max_gap)
        if not spatial:
            return False  # impossible displacement
        iou = _box_iou(prev["observations"][-1]["bbox"], cur["observations"][0]["bbox"])
        thr = appearance_thr
        if gap > LONG_GAP_S:
            thr = max(thr, LONG_GAP_APPEARANCE_THR)
        if _cut_in_gap(pe, cs):
            thr = max(thr, CUT_APPEARANCE_THR)
            if iou <= 0.1 and not (da and db and app >= thr):
                return False  # scene-cut boundary without independent evidence
        if da and db:
            if app < thr:
                return False  # incompatible appearance
            return True
        return iou > 0.1  # descriptorless: near-overlap only

    # Stage 1: same global_id merges ONLY when continuity also allows it.
    # Gallery agreement is necessary but not sufficient — this closes the
    # implausible-resume path where lookalikes shared one id across a jump.
    by_gid: dict[str, dict] = {}
    for c in candidates:
        gid = c.get("global_id", "")
        if gid in by_gid and _can_merge(by_gid[gid], c):
            prev = by_gid[gid]
            seen = {(o.get("frame_index"), tuple(o["bbox"])) for o in prev["observations"]}
            for o in c["observations"]:
                if (o.get("frame_index"), tuple(o["bbox"])) not in seen:
                    prev["observations"].append(o)
            prev["observations"].sort(key=lambda o: float(o["t"]))
            prev["confidence"] = max(float(prev.get("confidence", 0)),
                                     float(c.get("confidence", 0)))
        elif gid in by_gid:
            # Same gallery id but continuity rejected: fork an unknown identity
            # rather than forcing a bad match.
            c["global_id"] = f"g_{uuid.uuid4().hex[:10]}"
            c["identity_score"] = 0.0
            by_gid[c["global_id"]] = c
        else:
            by_gid[gid] = c
    merged = list(by_gid.values())
    # Stage 2: bridge unmatched fragments across gaps (earliest first, greedy).
    merged.sort(key=lambda c: float(c["observations"][0]["t"]) if c["observations"] else 0.0)
    out: list[dict] = []
    for cur in merged:
        if not cur["observations"]:
            out.append(cur)
            continue
        best = None
        for prev in out:
            if _can_merge(prev, cur):
                best = prev
                break
        if best is None:
            out.append(cur)
        else:
            seen = {(o.get("frame_index"), tuple(o["bbox"])) for o in best["observations"]}
            for o in cur["observations"]:
                if (o.get("frame_index"), tuple(o["bbox"])) not in seen:
                    best["observations"].append(o)
            best["observations"].sort(key=lambda o: float(o["t"]))
            best["confidence"] = max(float(best.get("confidence", 0)),
                                     float(cur.get("confidence", 0)))
            # Keep the stronger descriptor for the gallery update.
            if len(cur.get("_desc") or []) and not best.get("_desc"):
                best["_desc"] = cur["_desc"]
    return out
