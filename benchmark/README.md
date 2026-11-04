# Long-RVOS mini-benchmark (15 small + difficult clips)

Subset of [Long-RVOS](https://huggingface.co/datasets/iSEE-Laboratory/Long-RVOS)
(CVPR 2026, long-term referring video object segmentation), `valid` split only.

## Selection (how the 15 were picked)

`selection.json` holds the ranked list. Score favours difficulty, filter keeps
clips small:

```
score = 3*n_objects + 0.3*n_expressions + 0.2*n_dynamic_or_hybrid - 0.005*n_frames
keep only videos with n_annotated_frames <= 300   # "small vids"
take top 15 by score                              # "really difficult"
```

Difficulty signals: many referred objects (crowding/similar objects),
many expressions, dynamic+hybrid expressions (motion-dependent reference —
harder than static appearance). `gt` stats (disappearance rate, mean mask
area) are computed from the real `Annotations.tar.gz` after fetch.

Each clip's `focus` is its longest dynamic expression (fallback hybrid, then
static) — the hardest referring expression for one target object.

## Layout

```
benchmark/
  long_rvos.py      # harness: select | fetch | build | run | all
  README.md         # this file
  selection.json    # 15 videos + expressions + difficulty stats
  cache/            # HF tarballs (gitignored, deleted after extract)
  clips_src/        # extracted JPEGs (gitignored, deleted after mp4 build)
  clips/            # 15 mp4s, 640px wide (gitignored via *.mp4)
  results/          # per-video {video_id}.json + results_summary.json
```

## Reproduce

```bash
python benchmark/long_rvos.py select   # rank valid split -> selection.json
python benchmark/long_rvos.py fetch    # download annotations + stream JPEGs for the 15
python benchmark/long_rvos.py build    # JPEGs -> clips/*.mp4 (640px, <=30s)
python benchmark/long_rvos.py run      # full engine + benchmark.run_all per clip
python benchmark/long_rvos.py all --limit 3   # quick validation on first 3
```

`run` needs `META_API_KEY` (reads `.env`, same as the engine). One clip costs
~8 Spark calls + SAM streams; 15 clips take a while — runs sequentially and
records honest `failed` rows instead of stopping.

## Scoring

Per clip, `results/{video_id}.json` stores:

- pipeline stats: `status`, `n_frames_indexed`, `n_entities`, `n_events`,
  `ablation_level`, `answer` + `answer_verification`
- `benchmark`: the engine's 11 adversarial checks (`benchmark.run_all`)
  derived from stored data (shuffle, deletion, desync, occlusion…)

`shortcut`: no mask-IoU (J&F) scoring — the engine outputs entity boxes/event
spans, not per-pixel masks, so classic RVOS J&F needs a mask head. Upgrade when
a mask output lands. GT difficulty stats are reported instead of invented IoU.
