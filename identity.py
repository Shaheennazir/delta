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
    """Attach persistent global_id to each instance candidate; update gallery."""
    import store
    gallery = store.get_gallery()
    mw = float(media_wh[0] or 0)
    for c in candidates:
        desc = candidate_descriptor(c["observations"], frame_rows, mw)
        hit = match(desc, c.get("kind", "object"), gallery)
        if hit:
            gid = hit["global_id"]
            c["identity_score"] = hit["score"]
        else:
            gid = f"g_{uuid.uuid4().hex[:10]}"
            c["identity_score"] = 0.0
        c["entity_id"] = gid
        c["global_id"] = gid
        for o in c["observations"]:
            o["entity_id"] = gid
        if desc:
            store.upsert_gallery(gid, c.get("label", ""), c.get("kind", "object"),
                                 desc, float(c.get("confidence", 0.0)))
            gallery = store.get_gallery()
        else:
            # No visual descriptor: keep a kind-scoped row so the id is durable,
            # but do not let an empty vector match anything.
            store.upsert_gallery(gid, c.get("label", ""), c.get("kind", "object"),
                                 [], float(c.get("confidence", 0.0)))
    return candidates
