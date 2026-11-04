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
