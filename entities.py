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


def _split_implausible_tracklets(obs: list[dict]) -> list[list[dict]]:
    """Split one SAM object into tracklets at resume-implausible gaps.

    A SAM-native track can jump 6.8-23s or teleport across the frame; keeping
    it whole contaminates the event graph. A boundary is cut wherever
    identity.resume_plausible rejects the hop (temporal gap >MAX or drift /
    size change impossible). Fragments are preserved as observations — never
    discarded — with separate identities; later re-identification happens only
    through identity.stitch_tracklets' independent appearance+spatial gate.
    """
    if len(obs) <= 1:
        return [obs]
    from identity import resume_plausible
    ordered = sorted(obs, key=lambda o: float(o["t"]))
    out: list[list[dict]] = [[ordered[0]]]
    for prev, cur in zip(ordered, ordered[1:]):
        dt = float(cur["t"]) - float(prev["t"])
        try:
            ok = resume_plausible(prev["bbox"], cur["bbox"], dt)
        except Exception:
            ok = True
        if not ok:
            out.append([])
        out[-1].append(cur)
    return [frag for frag in out if frag]


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
                "track_conf": round(float(conf), 3),
                "sam_track_id": str(oid)} for fi, b in boxes.items()]
        obs.sort(key=lambda o: o["t"])
        # Split implausible jumps; keep every fragment as observations.
        frags = _split_implausible_tracklets(obs)
        for k, frag in enumerate(frags):
            cid = f"c{stream_i}_{j}_{oid}" if len(frags) == 1 else f"c{stream_i}_{j}_{oid}_p{k}"
            out.append({"candidate_id": cid,
                        "sam_track_id": str(oid), "sam_stream": stream_i,
                        "label": concept.get("label", concept.get("prompt", "")),
                        "kind": concept.get("kind", "object"),
                        "confidence": round(min(0.95, 0.4 + 0.3 * coverage), 3),
                        "observations": frag})
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

    Streams run concurrently, each budgeted. Failed streams are retried once
    (transient transport aborts were the observed zero-entity cause); permanent
    errors (missing key, bad handle) fail fast. Duplicate tracks are merged only
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
    n_retried = 0
    n_retry_ok = 0
    stream_errors: list[str] = []

    async def one(i: int, c: dict):
        nonlocal n_retried, n_retry_ok
        try:
            from sam_client import NoApiKey as _NoKey
        except Exception:  # pragma: no cover - import always succeeds
            _NoKey = ()
        for attempt in (1, 2):
            try:
                result = await asyncio.wait_for(
                    stream_segmentation(file_id, c["prompt"]), timeout_per_concept)
                tracks = tracks_from_result(result, n_frames, media=media_wh)
                if attempt == 2:
                    n_retry_ok += 1
                    log.info("SAM retry succeeded for %r", c.get("prompt"))
                return i, c, tracks, None
            except asyncio.CancelledError:
                raise
            except Exception as e:  # honest empty, never fake boxes
                if isinstance(e, _NoKey) or isinstance(e, ValueError):
                    log.warning("SAM stream failed fast for %r: %s", c.get("prompt"), e)
                    return i, c, None, str(e)[:200]
                log.warning("SAM stream failed for %r (attempt %d/2): %s",
                            c.get("prompt"), attempt, e)
                if attempt == 1:
                    n_retried += 1
                    try:
                        await asyncio.sleep(2.0)
                    except asyncio.CancelledError:
                        raise
                    continue
                return i, c, None, str(e)[:200]
        return i, c, None, "unreachable"  # pragma: no cover

    results = await asyncio.gather(*(one(i, c) for i, c in enumerate(concepts)))
    n_streams = len(concepts)
    n_ok = sum(1 for _, _, t, e in results if t and t.get("objects"))
    n_objects = sum(len((t or {}).get("objects") or {}) for _, _, t, _e in results)
    n_boxes = sum(len((o or {}).get("boxes") or {})
                  for _, _, t, _e in results for o in ((t or {}).get("objects") or {}).values())
    for _, c, tracks, err in results:
        if not (tracks and tracks.get("objects")) and err:
            stream_errors.append(f"{str(c.get('prompt', ''))[:40]}: {err}"[:120])
    if stream_errors:
        log.warning("SAM streams failed: %d/%d (%s)",
                    n_streams - n_ok, n_streams, "; ".join(stream_errors)[:300])
    cands: list[dict] = []
    for i, c, tracks, _err in results:
        if tracks and tracks.get("objects"):
            cands.extend(_objects_to_candidates(tracks, fps, c, i))
    n_pre_merge, n_empty_dropped = len(cands), 0
    merged = _merge(cands)
    n_merged_away = n_pre_merge - len(merged)
    cands = [c for c in merged if c["observations"]]
    n_empty_dropped = len(merged) - len(cands)
    n_split_extra = max(0, n_pre_merge - n_objects)
    if not cands:
        _coverage_log(session_id, {"concepts": [c.get("prompt", "") for c in concepts],
                                   "streams_launched": n_streams, "streams_ok": n_ok,
                                   "sam_objects": n_objects, "sam_boxes": n_boxes,
                                   "candidates": n_pre_merge, "merged_away": n_merged_away,
                                   "empty_dropped": n_empty_dropped,
                                   "split_frags": n_split_extra,
                                   "streams_retried": n_retried, "retry_ok": n_retry_ok,
                                   "stream_errors": list(stream_errors),
                                   "entities": 0, "observations": 0})
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
    _coverage_log(session_id, {"concepts": [c.get("prompt", "") for c in concepts],
                               "streams_launched": n_streams, "streams_ok": n_ok,
                               "sam_objects": n_objects, "sam_boxes": n_boxes,
                               "candidates": n_pre_merge, "merged_away": n_merged_away,
                               "empty_dropped": n_empty_dropped,
                               "split_frags": n_split_extra,
                               "streams_retried": n_retried, "retry_ok": n_retry_ok,
                               "stream_errors": list(stream_errors),
                               "entities": len(entities), "observations": len(all_obs)})
    return entities, all_obs


def _coverage_log(session_id: str, diag: dict) -> None:
    """Honest coverage accounting: why zero-entity clips have zero entities.

    Thresholds are NOT lowered here — that would trade false negatives for
    false tracks/identity switches (P1). The trace line records concepts,
    streams, SAM status, masks/tracks, and filter/merge/deletion counts so a
    0-entity result is diagnosable instead of silent.
    """
    msg = ("coverage concepts=%d streams=%d/%d ok objects=%d boxes=%d "
           "cands=%d merged_away=%d empty_dropped=%d split_frags=%d "
           "retried=%d retry_ok=%d -> entities=%d obs=%d" % (
               len(diag.get("concepts", [])), diag.get("streams_ok", 0),
               diag.get("streams_launched", 0), diag.get("sam_objects", 0),
               diag.get("sam_boxes", 0), diag.get("candidates", 0),
               diag.get("merged_away", 0), diag.get("empty_dropped", 0),
               diag.get("split_frags", 0), diag.get("streams_retried", 0),
               diag.get("retry_ok", 0),
               diag.get("entities", 0), diag.get("observations", 0)))
    log.info("entity %s", msg)
    if session_id:
        try:
            import store
            detail = msg[:240]
            errs = diag.get("stream_errors") or []
            if errs:
                detail = (detail + " | err: " + errs[0])[:300]
            store.append_trace(session_id, "segment_track_sam", "received", detail)
        except Exception:
            pass


def _same_track(a: list[dict], b: list[dict], iou_thr: float = 0.7) -> bool:
    by_fa = {o["frame_index"]: o["bbox"] for o in a}
    by_fb = {o["frame_index"]: o["bbox"] for o in b}
    common = set(by_fa) & set(by_fb)
    if len(common) < 3:
        return False
    ious = [_iou(tuple(by_fa[f]), tuple(by_fb[f])) for f in common]
    return sum(ious) / len(ious) > iou_thr
