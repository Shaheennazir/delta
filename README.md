# delta — Video Intelligence Engine (real models only)

Evidence-grounded video analysis over real Meta models. Upload an MP4, ask a
question; every factual claim carries timestamps + stored frame refs or the
system abstains.

## What it does

1. Speech: `muse-voice-transcribe-1.0` transcript + speaker turns; deterministic
   energy/onset analysis adds non-speech audio events.
2. Adaptive visual observation (`observer.py`): motion/scene-cut sampling
   (0.5/3/15 FPS tiers) into timestamped JPEG frame store `frames/<session>/`.
3. Entities (`entities.py` + `identity.py`): `sam-3.1`, one stream per concept,
   IoU merge, implausible-tracklet split, cross-session `global_id` gallery.
4. OCR (`ocr.py`): sharp-frame still-image passes + bounded-window voting; every
   row cites its detection frame.
5. Event graph (`events.py` + `temporal.py`): deterministic candidates from
   signals; Spark labels by `candidate_id` only; Allen-interval ordering and
   contradiction checks are pure functions of normalized timestamps.
6. Understanding (`intel.py`, model `muse-spark-1.3-contributor`): answers the
   viewer prompt first, then summary/themes/moments; timestamps clamped.
7. Verification (`evidence.py`): span bounds + frame-id + frame-time gate,
   isotonic calibration, selective abstention. Unsupported → abstain with reason.
8. QA (`investigator.py`, `chat.py`): plan → retrieve (`retrieval.py`) → inspect
   → gate. Chat router: followup (stored analysis) | investigate | reanalyze.
9. Benchmark (`benchmark.py`): 11 adversarial tests over stored data; unevaluable
   → `skip`, never pass. `benchmark/` holds the Long-RVOS harness + frozen
   `results_v0N/` reruns.

## Stack

- Backend: FastAPI + Pydantic v2, sqlite (`data/delta.db`, WAL) via `store.py`,
  numpy/scipy/PIL/imageio. No mocks in live path.
- Frontend: Next.js 15 + React 19 + Tailwind (`frontend/`). Chat-first UI.
- Models: `sam-3.1`, `muse-spark-1.3-contributor` via `sam_client.py`/`spark.py`.

## Prerequisites

- Python 3.10+, Node 18+, ffmpeg on PATH (via `imageio-ffmpeg`).
- Model API key from https://dev.meta.ai (Dashboard → API keys).

## Setup

```bash
python -m pip install -r requirements.txt
cp .env.example .env   # fill META_API_KEY, never commit .env
python -m uvicorn app:app --port 8000
cd frontend && npm install && npm run dev  # http://localhost:3000
```

Env (`sam_client.api_key()` order): `META_API_KEY` → `MODEL_API_KEY` →
`SAM_API_KEY` → same keys in repo-root `.env`. Optional:
`META_API_BASE_URL` (default `https://api.meta.ai/v1`), `SAM_MODEL`
(default `sam-3.1`).

## Use

`POST /api/v1/uploads` (multipart MP4) → `POST /api/v1/sessions`
`{title, focus, video_url}` → poll `GET /api/v1/sessions/{id}` + `/progress`
or SSE `/stream`. Stages: `queued → uploading → transcribing → observing →
tracking → ocr → events → understanding → verifying → completed/failed`.
Then `GET …/events|entities|frames|ocr|audio-events|trace|state|alignment`,
`POST …/qa {question}`, `POST …/chat {question}`, `POST …/benchmark`.

Failures return `failed` with the real error in `/progress` — never fake data.
Perception sub-steps degrade to honest empty and are named in the completed
detail (`· degraded: …`); core (upload/probe/understand) failure marks `failed`.

## Testing

```bash
python -m pytest -q            # 29 tests: offline capabilities + live API
python -m pytest test_api.py -q
cd frontend && npm run build   # Next.js production build
```

Live pipeline tests need `META_API_KEY`. Benchmark: `POST
/api/v1/sessions/{id}/benchmark`, or `benchmark/long_rvos.py` (see
`benchmark/README.md`). Frozen reruns live in `benchmark/results_v0N/`.

## Layout

Flat modules at repo root, one responsibility each (see `ARCHITECTURE.md`).
`calibration.py`/`extractor.py`/`primitives.py` are live helpers (geometry used
by `sam_client.py`), not dead code; `cli.py` is a standalone geometry
self-check. `MAP.md` is the detailed change log.
