# delta
Autonomous Football Drill & Coaching Analytics Engine.

## Phase 1 — Kinematics CLI
```bash
pip install numpy scipy pytest
python cli.py --self-check
pytest test_engine.py -q
```
Engine: `calibration.py` (ball-scale + homography), `extractor.py` (mask features), `primitives.py` (5 spatial primitives).

## Phase 2 — Planner/Reviewer + API
```bash
python -m uvicorn app:app --reload  # POST /api/v1/sessions, GET /api/v1/sessions/{id}
pytest test_engine.py test_api.py -q
```
No `META_API_KEY` needed: planner, SAM tracker, and reviewer fall back to deterministic mocks. Set `META_API_KEY` to hit `muse-spark-1.3` / `sam-3.1`.

## Phase 3 — Coaching Dashboard
```bash
cd frontend && npm install && npm run dev  # http://localhost:3000
```
Needs the API on :8000 (`python -m uvicorn app:app --port 8000`). Set `NEXT_PUBLIC_API_BASE` (see `frontend/.env.example`).

## Real Meta integration (key required)
```bash
python -m pip install -r requirements.txt
export META_API_KEY=...        # Model API key from dev.meta.ai (never commit)
python -m uvicorn app:app --port 8000
```
Upload an MP4 in the dashboard (or POST /api/v1/uploads). The backend uploads to the Files API, streams sam-3.1 per role, runs real Spark planning/review, and the dashboard replays true masks via @meta-sam/react + @meta-sam/parser. Without a key everything falls back to deterministic mocks.
