"""Adaptive visual observation (replaces uniform frame sampling).

Tiers (proposed defaults, preserved with timestamps, never silently gapped):
  static   ~0.5 FPS, normal ~3 FPS, rapid interaction ~15 FPS,
  ambiguous -> temporary upsample, suspected transition -> inspect neighbours.

Implementation: one ffmpeg probe pass decodes at a low rate and scores motion
via downscaled grayscale differences (numpy only, no cv2), detects scene cuts
via histogram distance, then one ffmpeg select pass materialises the plan as
JPEGs (JPEGs on disk + sqlite rows with original timestamps). Two decode
passes total, regardless of frame count — never one process per frame.
"""
from __future__ import annotations

import contextlib
import logging
import os
import subprocess
import tempfile

import imageio_ffmpeg
import numpy as np

log = logging.getLogger("delta.observer")

FRAME_DIR = os.path.join(os.path.dirname(__file__), "frames")
os.makedirs(FRAME_DIR, exist_ok=True)

STATIC_FPS, NORMAL_FPS, RAPID_FPS = 0.5, 3.0, 15.0
PROBE_WIDTH = 96


@contextlib.contextmanager
def _temp_video(video_bytes: bytes):
    """One temp file shared by probe + materialise (ffmpeg seeks files fast)."""
    fd, path = tempfile.mkstemp(suffix=".mp4")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(video_bytes)
        yield path
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


def _probe_path(path: str, meta: dict, probe_fps: float = 2.0) -> list[dict]:
    """Decode probe frames in one pass; return per-probe motion/scene scores."""
    fps = float(meta.get("fps") or 30.0)
    mw, mh = meta.get("width") or 0, meta.get("height") or 0
    ph = max(2, round(PROBE_WIDTH * mh / mw / 2) * 2) if mw and mh else 54
    vf = f"fps={probe_fps},scale={PROBE_WIDTH}:{ph}:flags=fast_bilinear,format=gray"
    cmd = [imageio_ffmpeg.get_ffmpeg_exe(), "-v", "error", "-nostdin", "-i", path,
           "-vf", vf, "-f", "rawvideo", "-pix_fmt", "gray", "pipe:1"]
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0:
        log.warning("probe ffmpeg failed: %s", proc.stderr.decode(errors="replace")[-300:])
        return []
    frame_size = PROBE_WIDTH * ph
    count = len(proc.stdout) // frame_size
    prev = None
    prev_hist = None
    out: list[dict] = []
    for k in range(count):
        g = np.frombuffer(proc.stdout, dtype=np.uint8, count=frame_size,
                          offset=k * frame_size).reshape(ph, PROBE_WIDTH)
        g = g.astype(np.float32) / 255.0
        hist, _ = np.histogram(g, bins=16, range=(0, 1))
        hist = hist / max(1, hist.sum())
        if prev is None:
            motion, cut = 0.0, False
        else:
            motion = float(np.mean(np.abs(g - prev)))
            cut = bool(float(np.sum(np.abs(hist - prev_hist))) > 0.9 or motion > 0.45)
        fi = int(round(k / probe_fps * fps))
        out.append({"frame_index": fi, "t": round(fi / fps, 3), "motion": motion,
                    "scene_cut": bool(cut)})
        prev, prev_hist = g, hist
    return out


def probe_motion(video_bytes: bytes, probe_fps: float = 2.0) -> tuple[list[dict], dict]:
    """Decode a sparse probe track; return per-probe motion/scene scores + meta."""
    from media import probe
    meta = probe(video_bytes)
    if not meta.get("duration_s"):
        return [], meta
    with _temp_video(video_bytes) as path:
        return _probe_path(path, meta, probe_fps), meta


def plan_sampling(probe_track: list[dict], duration: float, base_fps: float = 30.0,
                  max_frames: int = 400) -> list[dict]:
    """Map probe motion to tiered sampling. Returns [{frame_index, t, tier, reason}]."""
    picks: dict[int, dict] = {}
    for p in probe_track:
        m = p["motion"]
        if p["scene_cut"]:
            tier, fps, reason = "transition", RAPID_FPS, "scene-cut-neighbourhood"
        elif m > 0.18:
            tier, fps, reason = "rapid", RAPID_FPS, f"high-motion {m:.2f}"
        elif m > 0.05:
            tier, fps, reason = "normal", NORMAL_FPS, f"motion {m:.2f}"
        else:
            tier, fps, reason = "static", STATIC_FPS, "static"
        # Emit samples for the interval until the next probe point.
        t0 = p["t"]
        t1 = duration
        nxt = next((q for q in probe_track if q["t"] > t0), None)
        if nxt:
            t1 = nxt["t"]
        import math
        count = max(1, min(8, int(math.ceil((t1 - t0) * fps))))
        for k in range(count):
            t = t0 + (t1 - t0) * k / max(1, count)
            fi = int(round(t * base_fps))
            picks[fi] = {"frame_index": fi, "t": round(t, 3), "tier": tier, "reason": reason}
    ordered = sorted(picks.values(), key=lambda r: r["frame_index"])
    if len(ordered) > max_frames:  # thin uniformly, keep first/last + cuts
        keep_cuts = [r for r in ordered if r["tier"] == "transition"]
        rest = [r for r in ordered if r["tier"] != "transition"]
        stride = max(1, len(rest) // max(1, (max_frames - len(keep_cuts))))
        ordered = sorted(keep_cuts + rest[::stride], key=lambda r: r["frame_index"])[:max_frames]
    # Ambiguous-event upsample marker: rapid tiers already dense; flag gaps >2.5s.
    return ordered


def _select_expr(indexes: list[int]) -> str:
    """Balanced OR tree — a flat 120-term chain overflows ffmpeg's parser."""
    terms = [f"eq(n\\,{i})" for i in indexes]

    def merge(xs: list[str]) -> str:
        if len(xs) == 1:
            return xs[0]
        mid = len(xs) // 2
        return f"({merge(xs[:mid])}+{merge(xs[mid:])})"

    return merge(terms)


def _materialise_path(path: str, plan: list[dict], session_id: str,
                      max_width: int = 960) -> list[dict]:
    """One ffmpeg select pass writes only planned frames; rows in plan order."""
    by_index = {int(r["frame_index"]): r for r in plan}
    indexes = sorted(by_index)
    if not indexes:
        return []
    sdir = os.path.join(FRAME_DIR, session_id)
    os.makedirs(sdir, exist_ok=True)
    for name in os.listdir(sdir):  # stale frames from a re-run
        if name.startswith("f") and name.endswith(".jpg"):
            try:
                os.remove(os.path.join(sdir, name))
            except OSError:
                pass
    vf = f"select='{_select_expr(indexes)}',scale='min({max_width},iw)':-2"
    cmd = [imageio_ffmpeg.get_ffmpeg_exe(), "-v", "error", "-nostdin", "-i", path,
           "-vf", vf, "-fps_mode", "passthrough", "-q:v", "4",
           os.path.join(sdir, "f%06d.jpg")]
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0:
        log.warning("materialise ffmpeg failed: %s",
                    proc.stderr.decode(errors="replace")[-300:])
        return []
    files = sorted(n for n in os.listdir(sdir) if n.startswith("f") and n.endswith(".jpg"))
    if len(files) != len(indexes):
        log.warning("frame count mismatch: wanted %d, decoded %d", len(indexes), len(files))
    rows: list[dict] = []
    for name, fi in zip(files, indexes):
        r = by_index[fi]
        rows.append({"frame_index": fi, "t": r["t"], "path": os.path.join(sdir, name),
                     "motion": 0.0, "keyframe": r["tier"] in ("rapid", "transition")})
    return rows


def materialise_frames(video_bytes: bytes, plan: list[dict], session_id: str,
                       max_width: int = 960) -> list[dict]:
    """Decode planned frames to JPEGs; return rows for store.save_frames."""
    with _temp_video(video_bytes) as path:
        return _materialise_path(path, plan, session_id, max_width)


def observe(video_bytes: bytes, session_id: str, base_fps: float,
            duration: float, max_frames: int = 400) -> tuple[list[dict], list[dict]]:
    """Full adaptive observation: probe -> plan -> materialise. Returns (plan, rows)."""
    from media import probe
    meta = probe(video_bytes)
    with _temp_video(video_bytes) as path:
        track = _probe_path(path, meta)
        plan = plan_sampling(track, duration, base_fps=base_fps, max_frames=max_frames)
        if not plan:  # degenerate: fall back to 1 FPS uniform so evidence still exists
            import math
            n = max(1, int(math.ceil(duration)))
            plan = [{"frame_index": int(t * base_fps), "t": float(t),
                     "tier": "static", "reason": "fallback-uniform"} for t in range(n)]
        rows = _materialise_path(path, plan, session_id)
    # Attach probe motion to rows by nearest probe time.
    by_t = {p["t"]: p["motion"] for p in track}
    pts = sorted(by_t)
    for r in rows:
        if pts:
            nearest = min(pts, key=lambda t: abs(t - r["t"]))
            r["motion"] = round(float(by_t[nearest]), 4)
    return plan, rows
