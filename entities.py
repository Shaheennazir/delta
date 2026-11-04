"""Persistent object identity (entity registry) on top of SAM 3.1.

Segmentation identity vs tracking identity vs real-world identity are kept
separate: a tracker gives continuity of an observed region; it never claims
who a person is. Real-world identity is always 'unknown' unless the question
provides it.

Flow: Spark proposes ≤4 trackable concept prompts from keyframes (real model,
still images) -> one SAM 3.1 stream per concept (real perception) ->
tracks_from_result -> deterministic IoU association -> entity registry with
observations [{t, bbox, track_conf}]. SAM errors yield empty (honest), never
invented boxes.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os

log = logging.getLogger("delta.entities")


def _iou(a: tuple, b: tuple) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    aa = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    bb = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    return inter / max(1.0, aa + bb - inter)


async def propose_concepts(keyframe_paths: list[str], focus: str = "") -> list[dict]:
    """Ask Spark (still images, NOT the video model) which ≤4 concepts to track."""
    from spark import extract_text, jpeg_data_url, post_unary, SPARK_MODEL
    from media import read_frame  # noqa (kept for symmetry; paths already JPEGs)
    content: list[dict] = [{"type": "input_text",
                            "text": f"Viewer focus: {focus}\nList up to 4 visually trackable subjects "
                                    "(people, balls, phones, cups, vehicles, animals). Return JSON list "
                                    "of {{label, kind, prompt}} where prompt is a short SAM referring "
                                    "expression like 'the person in red' or 'the glass on the table'. "
                                    "If nothing trackable, return []."}]
    for p in (keyframe_paths or [])[:3]:
        try:
            with open(p, "rb") as f:
                import base64
                content.append({"type": "input_image",
                                "image_url": f"data:image/jpeg;base64,{base64.b64encode(f.read()).decode()}"})
        except OSError:
            continue
    body = await post_unary({
        "model": SPARK_MODEL, "instructions": "Propose tracking concepts as strict JSON.",
        "input": [{"type": "message", "role": "user", "content": content}],
        "text": {"format": {"type": "json_schema", "name": "Concepts",
                 "schema": {"type": "object",
                            "properties": {"concepts": {"type": "array", "items": {"type": "object",
                                "properties": {"label": {"type": "string"}, "kind": {"type": "string"},
                                               "prompt": {"type": "string"}},
                                "required": ["label", "kind", "prompt"],
                                "additionalProperties": False}}},
                            "required": ["concepts"], "additionalProperties": False}, "strict": True}},
    })
    try:
        data = json.loads(extract_text(body))
        concepts = data.get("concepts", []) or []
    except Exception:
        return []
    return [{"label": str(c.get("label", ""))[:60], "kind": str(c.get("kind", "object"))[:30],
             "prompt": str(c.get("prompt", ""))[:200]}
            for c in concepts if c.get("prompt")][:4]


def _objects_to_candidates(tracks: dict, fps: float, concept: dict, stream_i: int) -> list[dict]:
    """One candidate per SAM object id — never per concept stream.

    A stream may contain several instances; collapsing them would merge two
    people into one identity, so object_id stays the unit here.
    """
    out = []
    for j, (oid, obj) in enumerate((tracks.get("objects") or {}).items()):
        boxes = obj.get("boxes") or {}
        if not boxes:
            continue
        n = int(obj.get("n_frames", 0))
        coverage = n / max(1, len(boxes))
        conf = min(0.99, 0.5 + 0.5 * coverage ** 0.5)
        obs = [{"entity_id": "", "t": round(int(fi) / max(1.0, fps), 3),
                "frame_index": int(fi), "bbox": [float(b[0]), float(b[1]),
                                                 float(b[2]), float(b[3])],
                "track_conf": round(float(conf), 3)} for fi, b in boxes.items()]
        out.append({"candidate_id": f"c{stream_i}_{j}_{oid}",
                    "label": concept.get("label", concept.get("prompt", "")),
                    "kind": concept.get("kind", "object"),
                    "confidence": round(min(0.95, 0.4 + 0.3 * coverage), 3),
                    "observations": obs})
    return out


def _merge(cands: list[dict]) -> list[dict]:
    """Merge duplicate SAM objects by shared-frame IoU; keeps instance separation."""
    merged: list[dict] = []
    for c in cands:
        target = None
        for m in merged:
            if _same_track(m["observations"], c["observations"], iou_thr=0.65):
                target = m
                break
        if target:
            seen = {(o["frame_index"], tuple(o["bbox"])) for o in target["observations"]}
            for o in c["observations"]:
                if (o["frame_index"], tuple(o["bbox"])) not in seen:
                    target["observations"].append(o)
            target["confidence"] = max(target["confidence"], c["confidence"])
        else:
            merged.append({**c, "observations": list(c["observations"])})
    for m in merged:
        m["observations"].sort(key=lambda o: o["t"])
    return merged


async def track_session(file_id: str, concepts: list[dict], fps: float,
                        n_frames: int, media_wh: tuple,
                        timeout_per_concept: float = 240.0,
                        frame_rows: list[dict] | None = None,
                        session_id: str = "") -> tuple[list[dict], list[dict]]:
    """One SAM stream per concept; one persistent identity per SAM object.

    Streams run concurrently, each budgeted. Duplicate tracks are merged only
    when their boxes actually overlap on shared frames. Appearance descriptors
    then bind each instance to a cross-session global_id via identity.py.
    """
    from sam_client import stream_segmentation, tracks_from_result
    import identity
    entities: list[dict] = []
    all_obs: list[dict] = []
    if not concepts:
        return entities, all_obs
    concepts = concepts[:3]

    async def one(i: int, c: dict):
        try:
            result = await asyncio.wait_for(
                stream_segmentation(file_id, c["prompt"]), timeout_per_concept)
            tracks = tracks_from_result(result, n_frames, media=media_wh)
            return i, c, tracks, None
        except (asyncio.TimeoutError, Exception) as e:  # honest empty, never fake boxes
            log.warning("SAM stream failed for %r: %s", c.get("prompt"), e)
            return i, c, None, str(e)[:200]

    results = await asyncio.gather(*(one(i, c) for i, c in enumerate(concepts)))
    cands: list[dict] = []
    for i, c, tracks, _err in results:
        if tracks and tracks.get("objects"):
            cands.extend(_objects_to_candidates(tracks, fps, c, i))
    cands = [c for c in _merge(cands) if c["observations"]]
    if not cands:
        return entities, all_obs

    identity.assign(cands, frame_rows or [], media_wh, session_id)
    for c in cands:
        n = len(c["observations"])
        mean_conf = sum(o["track_conf"] for o in c["observations"]) / max(1, n)
        entities.append({
            "entity_id": c["global_id"], "global_id": c["global_id"],
            "label": c["label"], "kind": c["kind"],
            "confidence": round(min(0.99, 0.5 * mean_conf + 0.5 * c["confidence"]), 3),
        })
        all_obs.extend(c["observations"])
    all_obs.sort(key=lambda o: (o["t"], o["entity_id"]))
    return entities, all_obs


def _same_track(a: list[dict], b: list[dict], iou_thr: float = 0.7) -> bool:
    by_fa = {o["frame_index"]: o["bbox"] for o in a}
    by_fb = {o["frame_index"]: o["bbox"] for o in b}
    common = set(by_fa) & set(by_fb)
    if len(common) < 3:
        return False
    ious = [_iou(tuple(by_fa[f]), tuple(by_fb[f])) for f in common]
    return sum(ious) / len(ious) > iou_thr
