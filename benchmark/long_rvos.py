#!/usr/bin/env python3
"""Long-RVOS mini-benchmark: 15 small + difficult clips -> engine -> results.

Usage: python benchmark/long_rvos.py select|fetch|build|run|all [--limit N]
"""
from __future__ import annotations

import argparse
import asyncio
import io
import json
import os
import sys
import tarfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

BENCH = os.path.join(REPO, "benchmark")
CACHE = os.path.join(BENCH, "cache")
SRC = os.path.join(BENCH, "clips_src")
CLIPS = os.path.join(BENCH, "clips")
RESULTS = os.environ.get("BENCH_RESULTS_DIR", os.path.join(BENCH, "results"))
SEL_PATH = os.path.join(BENCH, "selection.json")

REPO_ID = "iSEE-Laboratory/Long-RVOS"
SPLIT = "valid"
MAX_FRAMES = 300  # "small vids": annotated-frame cap
N_CLIPS = 15
CLIP_WIDTH = 640
CLIP_FPS_IF_DENSE = 30.0
CLIP_FPS_IF_SPARSE = 6.0
MAX_IMAGES_PER_CLIP = 900  # ~30 s cap


def _exp_priority(e: dict) -> tuple:
    order = {"dynamic": 0, "hybrid": 1, "static": 2}
    return (order.get(e.get("type", "static"), 2), -len(e.get("exp", "")))


def select(limit: int = N_CLIPS) -> list[dict]:
    from huggingface_hub import hf_hub_download

    meta_path = hf_hub_download(REPO_ID, f"{SPLIT}/meta_expressions.json",
                                repo_type="dataset", local_dir=CACHE)
    with open(meta_path) as f:
        videos = json.load(f)["videos"]
    rows = []
    for vid, info in videos.items():
        exprs = list(info["expressions"].values())
        nobjs = len({e["obj_id"] for e in exprs})
        ndyn = sum(1 for e in exprs if e.get("type") in ("dynamic", "hybrid"))
        nf = len(info["frames"])
        if nf > MAX_FRAMES:
            continue
        score = 3 * nobjs + 0.3 * len(exprs) + 0.2 * ndyn - 0.005 * nf
        best = sorted(info["expressions"].items(),
                      key=lambda kv: _exp_priority(kv[1]))[0]
        rows.append({"video_id": vid, "score": round(score, 2),
                     "n_frames": nf, "n_expressions": len(exprs),
                     "n_objects": nobjs, "n_dyn_hybrid": ndyn,
                     "exp_id": best[0], "obj_id": best[1]["obj_id"],
                     "exp_type": best[1]["type"], "focus": best[1]["exp"]})
    rows.sort(key=lambda r: -r["score"])
    sel = rows[:limit]
    with open(SEL_PATH, "w") as f:
        json.dump({"split": SPLIT, "clips": sel}, f, indent=2)
    print(f"selected {len(sel)} clips -> {SEL_PATH}")
    for r in sel:
        print(f"  {r['video_id']} score={r['score']} frames={r['n_frames']} "
              f"objs={r['n_objects']} expr={r['n_expressions']} [{r['exp_type']}]")
    return sel


def _load_selection() -> list[dict]:
    with open(SEL_PATH) as f:
        return json.load(f)["clips"]


def fetch() -> None:
    """Download annotations tar (small) + extract only selected JPEG videos."""
    from huggingface_hub import hf_hub_download

    sel = _load_selection()
    want = {c["video_id"] for c in sel}
    os.makedirs(SRC, exist_ok=True)

    ann_tar = hf_hub_download(REPO_ID, f"{SPLIT}/Annotations.tar.gz",
                              repo_type="dataset", local_dir=CACHE)
    print(f"annotations: {ann_tar} ({os.path.getsize(ann_tar) / 1e6:.0f} MB)")
    _gt_stats(sel, ann_tar)
    with open(SEL_PATH, "w") as f:
        json.dump({"split": SPLIT, "clips": sel}, f, indent=2)

    img_tar = hf_hub_download(REPO_ID, f"{SPLIT}/JPEGImages.tar.gz",
                              repo_type="dataset", local_dir=CACHE)
    print(f"images tar: {img_tar} ({os.path.getsize(img_tar) / 1e6:.0f} MB), "
          f"extracting {len(want)} videos…")
    found: dict[str, int] = {}
    with tarfile.open(img_tar, "r:gz") as tf:
        for m in tf:
            if not m.isfile():
                continue
            parts = m.name.split("/")
            vid = next((p for p in parts if p in want), None)
            if vid is None:
                continue
            out = os.path.join(SRC, vid, os.path.basename(m.name))
            if os.path.exists(out):
                found[vid] = found.get(vid, 0) + 1
                continue
            os.makedirs(os.path.join(SRC, vid), exist_ok=True)
            with open(out, "wb") as f:
                f.write(tf.extractfile(m).read())
            found[vid] = found.get(vid, 0) + 1
    missing = want - set(found)
    print(f"extracted: {len(found)} videos, "
          f"images={[f'{v}:{found[v]}' for v in sorted(found)]}")
    if missing:
        print(f"WARNING missing videos in tar: {sorted(missing)}")


def _gt_stats(sel: list[dict], ann_tar: str) -> None:
    """Real difficulty stats from GT masks: disappearance + mask size."""
    import numpy as np
    from PIL import Image

    by_vid: dict[str, dict] = {}
    with tarfile.open(ann_tar, "r:gz") as tf:
        for m in tf:
            if not m.isfile() or not m.name.endswith(".png"):
                continue
            parts = m.name.split("/")
            try:
                ai = parts.index("Annotations")
                vid, obj = parts[ai + 1], parts[ai + 2]
            except (ValueError, IndexError):
                continue
            d = by_vid.setdefault(vid, {"frames": set(), "areas": [], "n": 0})
            d["frames"].add(parts[-1].split(".")[0])
            d["n"] += 1
    for c in sel:
        d = by_vid.get(c["video_id"], {"frames": set(), "areas": [], "n": 0})
        c["gt"] = {"annotated_frames_with_masks": len(d["frames"]),
                   "masks_total": d["n"]}
        # mean mask area from a sample (cheap: first 20 masks of this video)
        areas = []
        with tarfile.open(ann_tar, "r:gz") as tf:
            for m in tf:
                if not m.isfile() or c["video_id"] not in m.name:
                    continue
                if not m.name.endswith(".png") or len(areas) >= 20:
                    continue
                try:
                    img = np.asarray(Image.open(io.BytesIO(
                        tf.extractfile(m).read())).convert("L"))
                    areas.append(float((img > 127).mean()))
                except Exception:
                    pass
        c["gt"]["mean_mask_fill_20"] = round(float(np.mean(areas)), 4) if areas else 0.0


def build() -> None:
    import imageio.v3 as iio

    sel = _load_selection()
    os.makedirs(CLIPS, exist_ok=True)
    for c in sel:
        srcdir = os.path.join(SRC, c["video_id"])
        jpgs = sorted(f for f in os.listdir(srcdir)
                      if f.lower().endswith((".jpg", ".jpeg"))) if os.path.isdir(srcdir) else []
        if not jpgs:
            print(f"  {c['video_id']}: NO IMAGES, skipped")
            continue
        jpgs = jpgs[:MAX_IMAGES_PER_CLIP]
        dense = len(jpgs) > 2 * c["n_frames"]
        fps = CLIP_FPS_IF_DENSE if dense else CLIP_FPS_IF_SPARSE
        out = os.path.join(CLIPS, f"{c['video_id']}.mp4")
        frames = []
        for j in jpgs:
            img = iio.imread(os.path.join(srcdir, j))
            if img.shape[1] > CLIP_WIDTH:
                h = int(img.shape[0] * CLIP_WIDTH / img.shape[1])
                from PIL import Image as _I
                img = __import__("numpy").asarray(
                    _I.fromarray(img).resize((CLIP_WIDTH, h)))
            frames.append(img)
        iio.imwrite(out, frames, fps=fps)
        c["clip"] = out
        c["clip_images"] = len(frames)
        c["clip_fps"] = fps
        c["clip_duration_s"] = round(len(frames) / fps, 1)
        print(f"  {c['video_id']}: {len(frames)} imgs @ {fps}fps = "
              f"{c['clip_duration_s']}s -> {os.path.basename(out)}")
    with open(SEL_PATH, "w") as f:
        json.dump({"split": SPLIT, "clips": sel}, f, indent=2)


async def _run_one(c: dict) -> dict:
    from schemas import VideoSessionInput
    from orchestrator import run_pipeline_async
    import benchmark as benchmod

    clip = os.path.join(CLIPS, f"{c['video_id']}.mp4")
    inp = VideoSessionInput(title=f"longrvos-{c['video_id']}",
                            focus=c["focus"], video_url=clip)
    try:
        report = await run_pipeline_async(inp)
        res = benchmod.run_all(report.session_id)
        return {"video_id": c["video_id"], "status": report.status,
                "answer": report.answer[:500],
                "answer_verification": report.answer_verification,
                "n_frames_indexed": report.n_frames_indexed,
                "n_entities": report.n_entities, "n_events": report.n_events,
                "ablation_level": report.ablation_level,
                "benchmark": res}
    except Exception as e:
        return {"video_id": c["video_id"], "status": "failed",
                "error": f"{type(e).__name__}: {e}"[:300]}


def run(limit: int | None = None) -> None:
    sel = _load_selection()
    if limit:
        sel = sel[:limit]
    os.makedirs(RESULTS, exist_ok=True)

    async def _all():
        out = []
        for i, c in enumerate(sel):
            print(f"[{i + 1}/{len(sel)}] {c['video_id']} running…", flush=True)
            r = await _run_one(c)
            r["focus"] = c["focus"]
            r["gt"] = c.get("gt", {})
            with open(os.path.join(RESULTS, f"{c['video_id']}.json"), "w") as f:
                json.dump(r, f, indent=2)
            out.append(r)
            print(f"[{i + 1}/{len(sel)}] {c['video_id']} -> {r['status']}", flush=True)
        return out

    out = asyncio.run(_all())
    _summarize(out)


def _summarize(out: list[dict]) -> None:
    ok = [r for r in out if r["status"] == "completed"]
    agg: dict = {"n": len(out), "completed": len(ok),
                 "benchmark_totals": {},
                 "clips": [{k: r.get(k) for k in
                             ("video_id", "status", "n_frames_indexed",
                              "n_entities", "n_events", "ablation_level",
                              "answer_verification")}
                            for r in out]}
    for r in ok:
        b = r.get("benchmark", {})
        for t in b.get("tests", []):
            d = agg["benchmark_totals"].setdefault(
                t["test"], {"pass": 0, "fail": 0, "skip": 0})
            d[t["result"]] = d.get(t["result"], 0) + 1
    with open(os.path.join(RESULTS, "results_summary.json"), "w") as f:
        json.dump(agg, f, indent=2)
    print(json.dumps(agg, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["select", "fetch", "build", "run", "all"])
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    if args.cmd in ("select", "all"):
        select(args.limit or N_CLIPS)
    if args.cmd in ("fetch", "all"):
        fetch()
    if args.cmd in ("build", "all"):
        build()
    if args.cmd in ("run", "all"):
        run(args.limit)


if __name__ == "__main__":
    main()
