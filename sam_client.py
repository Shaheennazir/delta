"""Real SAM 3.1 perception over the Meta Model API (spec sections 3.2, 5.1, 6).

Contract (from meta-models/meta-sam + dev.meta.ai):
- Auth:  Bearer $META_API_KEY (aka MODEL_API_KEY), base https://api.meta.ai/v1
- Files: POST /files multipart (purpose=user_data, file=MP4) -> {"id": "file-..."}
- Video: POST /responses {model:"sam-3.1", stream:true,
          input:[{message, content:[{input_text}, {input_video:{file_id}}]}]}
  SSE events feed meta_sam_parser.parse_responses_stream(..., video format).
- One stream per concept prompt (asyncio.gather); every object in a stream
  shares that stream's role. Object-multiplex (build_object_prompt) is the
  fallback when Frame 0 already gives boxes (<=16 objects, one pass).

Without a key every entry point raises NoApiKey and callers use mock fallbacks.
"""
from __future__ import annotations

import json
import os
import re
from collections.abc import AsyncIterator

import httpx
import numpy as np
from meta_sam_parser import (
    decode_mask_to_raster,
    parse_responses_stream,
    video_segmentation_format,
)

from extractor import centroid, foot_contact_point, stance_width_m, torso_axis_deg

BASE_URL = os.getenv("META_API_BASE_URL", "https://api.meta.ai/v1").rstrip("/")
SAM_MODEL = os.getenv("SAM_MODEL", "sam-3.1")
FILE_ID_PATTERN = re.compile(r"^file-[A-Za-z0-9_-]{1,120}$")
FILE_TIMEOUT = httpx.Timeout(connect=30.0, read=600.0, write=120.0, pool=30.0)


class NoApiKey(RuntimeError):
    pass


_dotenv_cache: dict | None = None


def _dotenv() -> dict:
    """Values from the repo-root .env (no extra dependency). Env vars win."""
    global _dotenv_cache
    if _dotenv_cache is None:
        vals: dict = {}
        try:
            with open(os.path.join(os.path.dirname(__file__), ".env")) as f:
                for line in f:
                    s = line.strip()
                    if s and not s.startswith("#") and "=" in s:
                        k, v = s.split("=", 1)
                        vals[k.strip()] = v.strip().strip("\"").strip("'")
        except OSError:
            pass
        _dotenv_cache = vals
    return _dotenv_cache


def api_key() -> str:
    key = (os.getenv("META_API_KEY") or os.getenv("MODEL_API_KEY") or os.getenv("SAM_API_KEY")
           or _dotenv().get("META_API_KEY") or _dotenv().get("MODEL_API_KEY")
           or _dotenv().get("SAM_API_KEY"))
    if not key:
        raise NoApiKey("set META_API_KEY (Model API key from dev.meta.ai)")
    return key


async def list_models() -> list[str]:
    """GET /models — confirms model ids (sam-3.1, muse-spark-1.3) live."""
    async with httpx.AsyncClient(base_url=BASE_URL, timeout=30.0) as c:
        r = await c.get("/models", headers={"Authorization": f"Bearer {api_key()}"})
        r.raise_for_status()
        data = r.json()
    return [e["id"] for e in data.get("data", []) if isinstance(e.get("id"), str)]


async def upload_file(data: bytes, filename: str = "clip.mp4") -> str:
    """Upload MP4 bytes to the Files API. Returns the opaque file_id handle."""
    async with httpx.AsyncClient(base_url=BASE_URL, timeout=FILE_TIMEOUT) as c:
        r = await c.post(
            "/files",
            headers={"Authorization": f"Bearer {api_key()}"},
            data={"purpose": "user_data"},
            files={"file": (filename, data, "video/mp4")},
        )
        r.raise_for_status()
        file_id = r.json().get("id", "")
    if not FILE_ID_PATTERN.match(file_id):
        raise ValueError(f"Files API returned an invalid handle: {file_id!r}")
    return file_id


def local_video_bytes(video_url: str) -> bytes | None:
    """Bytes already on disk for upload:// handles and local paths; None otherwise."""
    if FILE_ID_PATTERN.match(video_url) or video_url.startswith(("http://", "https://")):
        return None
    from media import UPLOAD_DIR, validate_upload

    if video_url.startswith("upload://"):
        path = os.path.join(UPLOAD_DIR, os.path.basename(video_url[len("upload://"):]))
    else:
        path = video_url
    with open(path, "rb") as f:
        data = f.read()
    validate_upload(data)
    return data


async def resolve_video(video_url: str) -> tuple[str, bytes | None]:
    """video_url may be a file_id, upload:// handle, http(s) URL, or local path.

    Returns (file_id, raw_bytes|None). Bytes are kept when available for
    Frame-0 grounding and keyframe crops; remote URLs are downloaded once.
    """
    from media import UPLOAD_DIR, save_upload, validate_upload

    if FILE_ID_PATTERN.match(video_url):
        return video_url, None
    if video_url.startswith("upload://"):
        path = os.path.join(UPLOAD_DIR, os.path.basename(video_url[len("upload://"):]))
        with open(path, "rb") as f:
            data = f.read()
        validate_upload(data)
        return await upload_file(data, os.path.basename(path)), data
    if video_url.startswith(("http://", "https://")):
        async with httpx.AsyncClient(timeout=FILE_TIMEOUT, follow_redirects=True) as c:
            r = await c.get(video_url)
            r.raise_for_status()
            data = r.content
    else:
        with open(video_url, "rb") as f:
            data = f.read()
    validate_upload(data)
    save_upload(data, video_url.rsplit("/", 1)[-1] or "clip.mp4")
    return await upload_file(data), data


def _responses_body(file_id: str, prompt: str) -> dict:
    return {
        "model": SAM_MODEL,
        "stream": True,
        "input": [{
            "type": "message", "role": "user",
            "content": [
                {"type": "input_text", "text": prompt},
                {"type": "input_video", "file_id": file_id},
            ],
        }],
        "metadata": {"mask_encoding": "one_bit", "include_confidence": "true"},
    }


async def _sse_events(response: httpx.Response) -> AsyncIterator[dict]:
    """Yield wire event mappings from an SSE stream."""
    async for line in response.aiter_lines():
        line = line.strip()
        if not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if payload in ("[DONE]", ""):
            continue
        try:
            yield json.loads(payload)
        except json.JSONDecodeError:
            continue


async def stream_segmentation(file_id: str, prompt: str):
    """Parsed snapshot stream + final result for one concept prompt.

    Returns (role-agnostic) VideoSegmentationResult. Raises on refusal/failure.
    """
    async with httpx.AsyncClient(base_url=BASE_URL, timeout=FILE_TIMEOUT) as c:
        async with c.stream(
            "POST", "/responses",
            headers={"Authorization": f"Bearer {api_key()}", "Accept": "text/event-stream"},
            json=_responses_body(file_id, prompt),
        ) as r:
            r.raise_for_status()
            parsed = parse_responses_stream(_sse_events(r), video_segmentation_format())
            result = await parsed.final_result()
    if result.outcome.status != "completed":
        raise RuntimeError(f"SAM stream ended {result.outcome.status}")
    return result


def _raster_2d(mask) -> np.ndarray:
    h, w = mask.height, mask.width
    arr = np.frombuffer(decode_mask_to_raster(mask), dtype=np.uint8).reshape(h, w)
    return arr.astype(bool)


def tracks_from_result(result, n_frames: int, media: tuple | None = None) -> dict:
    """Per-object tracks from a parsed result: foot/center/lean/boxes per frame.

    Mask rasters are box-sized crops at media resolution; box left/top offsets
    them back to media pixels. Missing frames stay NaN (ball gaps are filled
    downstream by kalman_fill_2d). Pass media=(w,h) from the video probe;
    otherwise max box extents are used as a fallback.
    """
    per_obj: dict[str, dict[int, tuple[np.ndarray, tuple]]] = {}
    media_w = media_h = 0
    for rec in result.records:
        if rec.kind != "mask":
            continue
        frame = rec.frame.frame_index if rec.frame is not None else 0
        raster = _raster_2d(rec.mask)
        b = rec.bounds
        foot = foot_contact_point(raster)
        ctr = centroid(raster)
        per_obj.setdefault(rec.object_id, {})[frame] = (raster, b, foot, ctr)
        media_w = max(media_w, b.right)
        media_h = max(media_h, b.bottom)

    objects: dict[str, dict] = {}
    for oid, frames in per_obj.items():
        T = max(n_frames, max(frames) + 1)
        foot = np.full((T, 2), np.nan)
        center = np.full((T, 2), np.nan)
        lean = np.full(T, np.nan)
        width_px = np.full(T, np.nan)
        area_px = np.full(T, np.nan)
        boxes = {}
        for f, (raster, b, ft, ct) in frames.items():
            if f >= T:
                continue
            foot[f] = (b.left + ft[0], b.top + ft[1])
            center[f] = (b.left + ct[0], b.top + ct[1])
            lean[f] = torso_axis_deg(raster)
            ys, xs = np.nonzero(raster)
            width_px[f] = float(xs.max() - xs.min()) if len(xs) else np.nan
            area_px[f] = float(raster.sum())
            boxes[f] = (b.left, b.top, b.right, b.bottom)
        objects[oid] = {
            "foot_px": foot, "center_px": center, "lean_deg": lean,
            "box_width_px": width_px, "area_px": area_px, "boxes": boxes,
            "n_frames": len(frames),
            "media": (media_w, media_h),
        }
    primary = max(objects, key=lambda o: objects[o]["n_frames"]) if objects else None
    mw, mh = media or (media_w, media_h)
    return {"objects": objects, "primary_id": primary, "media": (mw, mh)}
