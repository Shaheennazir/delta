"""Local media handling: validation, probing, frame extraction (spec section 8).

Uses imageio/imageio-ffmpeg (pip-only, no system ffmpeg needed). Videos are
never committed: validated uploads live under ./uploads/ (gitignored).
"""
from __future__ import annotations

import io
import os
import struct

import imageio.v3 as iio
import numpy as np

UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "uploads")
MAX_BYTES = 60 * 1024 * 1024
MAX_DURATION_S = 60.0
MAX_DIM = 1080

os.makedirs(UPLOAD_DIR, exist_ok=True)


def sniff_mp4(data: bytes) -> bool:
    """True if bytes look like MP4/ISO BMFF (ftyp box in first 32 bytes)."""
    if len(data) < 12:
        return False
    return data[4:8] == b"ftyp" and data[8:12] in (
        b"isom", b"iso2", b"iso4", b"iso5", b"iso6",
        b"mp41", b"mp42", b"avc1", b"dash", b"mmp4",
    )


def probe(data: bytes) -> dict:
    """Width, height, fps, duration, n_frames. Raises ValueError if unreadable."""
    try:
        meta = iio.immeta(io.BytesIO(data), extension=".mp4")
        fps = float(meta.get("fps", 30.0) or 30.0)
        size = meta.get("size") or (0, 0)
        nframes = meta.get("nframes")
        if nframes in (None, float("inf")):
            nframes = None
        duration = meta.get("duration")
        if isinstance(duration, (tuple, list)):
            duration = duration[0] if duration else 0.0
        return {
            "width": int(size[0]), "height": int(size[1]), "fps": fps,
            "duration_s": float(duration or (nframes / fps if nframes else 0.0)),
            "n_frames": int(nframes) if nframes else 0,
        }
    except Exception as e:
        raise ValueError(f"unreadable video: {e}") from e


def validate_upload(data: bytes, filename: str = "clip.mp4") -> dict:
    """Spec section 8 guards: MP4 only, <=60s, <=1080p. Returns probe meta."""
    if len(data) > MAX_BYTES:
        raise ValueError(f"clip too large: {len(data)} bytes (max {MAX_BYTES})")
    if not sniff_mp4(data):
        raise ValueError("only MP4/ISO BMFF clips are accepted")
    meta = probe(data)
    if meta["duration_s"] > MAX_DURATION_S:
        raise ValueError(f"clip too long: {meta['duration_s']:.1f}s (max {MAX_DURATION_S}s)")
    if max(meta["width"], meta["height"]) > MAX_DIM:
        raise ValueError(f"resolution too large: {meta['width']}x{meta['height']} (max dim {MAX_DIM})")
    return meta


def read_frame(data: bytes, index: int) -> np.ndarray:
    """Decode frame N as HxWx3 uint8."""
    frames = iio.imread(io.BytesIO(data), index=index, extension=".mp4")
    return np.asarray(frames)


def frame_to_jpeg(frame: np.ndarray, max_width: int = 960) -> bytes:
    """Encode (optionally downscaled) frame as JPEG bytes."""
    from PIL import Image

    img = Image.fromarray(np.asarray(frame, dtype=np.uint8))
    if img.width > max_width:
        img = img.resize((max_width, int(img.height * max_width / img.width)))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return buf.getvalue()


def crop_around(frame: np.ndarray, cx: float, cy: float, size_px: int = 320) -> np.ndarray:
    """Square crop centered on (cx, cy) media pixels, clamped to frame bounds."""
    h, w = frame.shape[:2]
    half = size_px // 2
    x0, y0 = int(max(0, cx - half)), int(max(0, cy - half))
    x1, y1 = int(min(w, x0 + size_px)), int(min(h, y0 + size_px))
    return frame[y0:y1, x0:x1]


def save_upload(data: bytes, filename: str) -> str:
    safe = "".join(c for c in os.path.basename(filename) if c.isalnum() or c in "._-") or "clip.mp4"
    path = os.path.join(UPLOAD_DIR, safe)
    with open(path, "wb") as f:
        f.write(data)
    return path
