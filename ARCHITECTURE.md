# ARCHITECTURE.md — delta

## Overview

Single-process FastAPI service (`app.py`) + `BackgroundTasks` worker. All
durable state in sqlite (`store.py`, `data/delta.db`, WAL) so restarts keep
sessions. No Redis/queue; single worker only.

```
upload (media.py) → prepare (wav + frames) → perceive concurrently ─┐
  transcribe.py (speech) │ entities.py+sam_client.py (tracking)      │
  ocr.py (text) │ audio_events.py (non-speech)                       │
→ events.py (candidates) → temporal.py (order/contradictions)        │
→ agent.py fuses → intel.py (Spark understand) → evidence.py (verify)
→ orchestrator.py persists SessionReport → investigator.py/chat.py answer QA
```

`orchestrator.run_pipeline_async` owns stage order and progress writes.
`agent.run_perception` / `run_fusion_understanding` own concurrency and budgets.

## Modules

| Module | Responsibility | Depends on |
|---|---|---|
| `app.py` | HTTP routes, validation, 404/409/422 mapping, SSE | orchestrator, store, schemas, investigator, chat, benchmark |
| `orchestrator.py` | Stage order, progress/report persistence, degrade-vs-fail policy | schemas, store, agent, intel, events, evidence |
| `agent.py` | Tool registry, concurrent perception, fusion, trace logging | store, transcribe, entities, ocr, audio_events, events, intel, evidence |
| `schemas.py` | Pydantic v2 API + domain contracts (`extra=forbid` on input) | — |
| `store.py` | sqlite persistence: sessions/progress/frames/entities/events/ocr/audio/qa/trace/gallery/outcomes | stdlib only |
| `observer.py` | Adaptive sampling + JPEG frame store `frames/<id>/` (two ffmpeg passes) | media |
| `media.py` | Upload validation/sandbox (`upload://` traversal guard) | — |
| `entities.py` | SAM concept streams → candidates, implausible-split, IoU merge | identity, sam_client, store(trace) |
| `identity.py` | Appearance descriptors, `resume_plausible`, `stitch_tracklets`, `global_id` gallery | store |
| `sam_client.py` | Auth (env + repo `.env`), Files upload, SAM SSE streams | extractor (geometry) |
| `transcribe.py` | WAV extract + `/asr/transcribe` DIARIZATION, TimeoutError on hang | sam_client (auth/base) |
| `ocr.py` | Sharp-ranked still passes + windowed voting, frame provenance | spark/intel |
| `audio_events.py` | Non-speech detection, `rms_envelope`, `shift_events` (audio→video clock) | — |
| `av_align.py` | RMS-vs-motion cross-correlation ±2 s, 50 ms grid; confidence 0 when flat | audio_events, observer data |
| `events.py` | Deterministic candidates (presence/motion/speech/OCR/audio/interaction), `state_timeline` | temporal, identity |
| `temporal.py` | Allen algebra on ms-normalized timestamps; `order_events` pure key sort | — |
| `evidence.py` | Span/frame/time gate, isotonic calibration routing, selective abstain | confidence, store |
| `confidence.py` | Isotonic (PAVA) fit, Brier/ECE, selective threshold (≤10% risk) | store |
| `retrieval.py` | Uniform IDF evidence index over stored rows; spans from index, never model | store |
| `investigator.py` | Plan → retrieve → inspect → gate; per-claim verify + calibrate + abstain | retrieval, evidence, intel, store |
| `chat.py` | Intent router (followup/investigate/reanalyze) + repeat-abstention guard | investigator, intel, evidence, store |
| `intel.py` / `spark.py` | Spark video understanding / transport (`muse-spark-1.3-contributor`) | sam_client |
| `benchmark.py` | 11 adversarial tests over stored data; skip-when-unevaluable + coverage/metrics | temporal, evidence, store |
| `confidence.py` | see above | |
| `calibration.py`/`extractor.py`/`primitives.py` | Pure geometry (homography, centroids, stance) used by `sam_client.py` | numpy/scipy |
| `cli.py` | Standalone geometry self-check (`print` JSON); not in live path | calibration/extractor/primitives |
| `frontend/` | Chat-first UI: landing → live status → answer + follow-ups; `engine` toggle for trace/events/entities | backend REST + SSE |

## Data flow

Upload bytes → `media.save_upload` (`uploads/`, traversal-guarded) →
`sam_client.resolve_video` (file_id) + WAV + frame rows once → perception rows
in sqlite → event candidates with `evidence_frames` mapped to **stored** frame
ids (unmapped runs dropped, never faked) → Spark labels by id → verified answer
(`answer` + `answer_evidence/confidence/verification/calibration`) → QA/benchmark
read the same stored rows.

## Persistence

`store.init_db` creates tables + backfills (`sessions.answer`, `entity_gallery`,
`calibration_points`, `outcomes`, …). `save_*` uses REPLACE semantics;
`get_*` returns plain dicts/lists. Trace (`append_trace/get_trace`) records every
tool sent/waiting/received/failed event; `_degraded_suffix` surfaces failed
tools in completed progress.

## Failure policy (load-bearing)

- Core (upload/probe/Spark understand, missing key): session `failed` + real
  error in `/progress`. Never silent empty.
- Perception (SAM track, OCR, event proposal, transcribe timeout): honest empty
  + trace `failed` + `degraded:` suffix on completed detail. Thresholds are not
  lowered to hide gaps (`entities._coverage_log`).
- Evidence: unverified/low-confidence claims abstain with reason
  (`verification=unknown`, `calibration=abstained`, status-certainty ≈0.9).
- API: unknown session → 404; QA/chat before `completed` → 409; bad upload →
  422; SSE caps at ~30 min.

## Known coupling / deferred

- Flat root layout is intentional: imports are `import store`-style with no
  cycles (`store.py` has zero internal imports). A `api/core/pipeline/…`
  repackage was deliberately **not** done — it would churn every import and
  test for no behavior gain.
- `orchestrator._store/_progress/_lanes` dual-write dicts + `save_lanes` /
  `get_lanes` are back-compat shims (sqlite is source of truth). `app.py` no
  longer imports `get_lanes`. Remove the shims only after confirming no
  external caller.
- `frontend/app/page.tsx` (~830 lines) is the one oversized component
  (landing + status + thread + engine panels). Split into route panels only
  when touching that UI — not done here to avoid visual regression.
- Broad `except Exception` in `investigator.py` inspection paths is intentional
  per-row degradation (bounded re-decode); each logs to trace. Do not convert
  to silent global fallbacks.
