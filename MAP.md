# MAP.md — delta (Phase 1)

- `calibration.py` → scale: `ball_alphas`, `fit_alpha_vs_y`, `calibrate_from_ball`, `find_homography`/`apply_homography`. Used by `cli.py`, tested in `test_engine.py`.
- `extractor.py` → per-mask features: `foot_contact_point`, `centroid`, `stance_width_m`, `torso_axis_deg`, `extract_features`. Used by `cli.py`.
- `primitives.py` → 5 primitives: `proximity_timer`, `tether_distance_series`/`nearest_tether_series`, `line_intersection`, `mask_kinematics`/`strike_frame`, `mask_pose_axis`. Used by `cli.py`.
- `cli.py` → `--self-check` synthetic Rondo demo (dwell + split) asserting all modules.
- `test_engine.py` → 5 pytest cases covering hysteresis, split dedupe, speed, calibration.
- `schemas.py` → Pydantic v2 contracts: CoachSessionInput, DrillAnalysisPlan (1-4 rules), SessionReport. extra=forbid.
- `planner.py` → mock_plan (keyword->rules) + plan_with_spark (real call, mock fallback).
- `reviewer.py` → mock_feedback (2-sentence cue) + review_with_spark.
- `orchestrator.py` → demo_tracks, evaluate_plan (rules->primitives), run_pipeline (+async), SAM stubs, in-memory store.
- `app.py` → FastAPI: POST/GET /api/v1/sessions, BackgroundTasks (stand-in for Celery+Redis).
- `test_api.py` → 5 pytest cases: plan caps, pipeline events, API round-trip, 422 sandboxing.
- `app.py` → + CORS; POST /uploads (MP4 validation), GET /segmentation (raw lanes), GET /stream (SSE).
- `frontend/` → Next.js 15 dashboard: app/page.tsx (CoachingWorkspace), components (DrillConfigSidebar, InteractiveStage, SamVideoPlayer w/ SDK fallback, TacticalSvgOverlay, ViolationTimelineBar, CoachDiagnosticCards), lib/api.ts + types.ts. `npm run build` clean.
