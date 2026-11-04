# MAP.md — delta (Video Intelligence, real-only)

- `schemas.py` → Pydantic v2 contracts: VideoSessionInput, SessionReport (answer + answer_evidence/answer_confidence/answer_verification/answer_calibration + av_offset_s/av_sync_confidence), EvidenceSpan (source_id/kind/verification), AnswerClaim, GraphEvent (candidate_id/source), QAOutcome/QAResponse(abstained).
- `identity.py` → NEW: instance-level appearance descriptors (4x4 spatial colour + saturation-weighted hue + gradient orientation, unit-norm) from stored frame crops; cosine match within kind to persistent `global_id` via `entity_gallery`; running-mean descriptor update.
- `confidence.py` → NEW: isotonic (PAVA) calibration from labelled outcomes, Brier/ECE, selective-prediction threshold at max 10% risk; `apply_saved` returns (score, "isotonic"|"uncalibrated"); no calibrated claim without outcomes.
- `av_align.py` → NEW: audio-RMS vs visual-motion cross-correlation on 50 ms grid, ±2 s lag search; `estimate_offset` + `events_offset`; flat/insufficient signals report confidence 0 with method flag, never fake sync.
- `retrieval.py` → NEW: uniform evidence index (events/entities/speech/OCR/audio) with IDF lexical score + time-hint overlap fraction; stored confidence is tiebreak only; spans + frame ids come from the index, never the model.
- `transcribe.py` → `extract_wav` (imageio-ffmpeg binary → mono 24kHz WAV, None when no audio) + `transcribe_wav` (POST `/asr/transcribe` DIARIZATION, 240s budget, TimeoutError instead of hang) + `transcribe_video` (accepts shared wav_bytes, no double ffmpeg). No mocks; transcribe timeout degrades to empty, never fails the session.
- `transcribe.py` → `extract_wav` (imageio-ffmpeg binary → mono 24kHz WAV, None when no audio) + `transcribe_wav` (POST `/asr/transcribe` DIARIZATION, 240s budget, TimeoutError instead of hang) + `transcribe_video` (accepts shared wav_bytes, no double ffmpeg). No mocks; transcribe timeout degrades to empty, never fails the session.
- `agent.py` → muse-spark contributor agent tool registry (transcribe_full_audio | segment_track_sam | ocr_frames | detect_audio_events | extract_event_graph | understand_verify); `run_perception` gathers the 4 perception tools concurrently; `_understand_tool` starts Spark video understanding as soon as upload + transcript are ready (often during frame decode) and `run_fusion_understanding` awaits that `intel_task`; `run_fusion_understanding` fuses -> event graph -> Spark -> evidence verify. Every call logs sent/waiting/received/failed to the trace store.
- `intel.py` → `analyze_video` (muse-spark-1.3-contributor Responses + `input_file` file_id + transcript, strict json_schema, clamps `t_start/t_end` to duration). The viewer prompt is answered first (`answer` field) before summary/themes/moments. No fallback.
- `spark.py` → `SPARK_MODEL = "muse-spark-1.3-contributor"`, `post_unary`/`extract_text` transport. Errors propagate.
- `sam_client.py` → `api_key()` (env + repo `.env`) + `local_video_bytes` (validated local/upload:// bytes) + Files `upload_file`/`resolve_video`. SAM streaming helpers remain but unused by pipeline.
- `orchestrator.py` → agent pipeline: upload (concurrent with prepare for local files) -> prepare (WAV+frames once) -> 4 perception tools concurrently (transcribe | SAM track | OCR | audio events; transcribe/audio pre-started) -> event graph -> Spark understand -> evidence verify. Carries Spark's direct `answer` into `SessionReport` + sqlite. Every tool budgeted, honest-empty on timeout; only upload/probe failure marks failed.
- `store.py` → sqlite persistence (`data/delta.db`): sessions (incl. direct `answer`)/progress/frames/entities/observations/events/ocr/audio_events/qa_runs/trace; old DBs backfill the `answer` column on init.
- `observer.py` → adaptive sampling (motion + scene cuts, 0.5/3/15 FPS tiers) + JPEG frame store `frames/<session>/`; two ffmpeg passes total (raw probe pipe + one balanced-select materialise), never one process per frame.
- `ocr.py` → independent still-image OCR passes (concurrent gather, 90s per-image budget) + multi-frame voting with frame refs.
- `entities.py` → instance-level registry: one candidate per SAM object id (never per concept stream), IoU merge at 0.65, then `identity.assign` binds each instance to cross-session `global_id` via appearance gallery.
- `audio_events.py` → deterministic non-speech detection + `rms_envelope` (50 ms hop) and `shift_events` (audio-clock → video-clock) used by av_align.
- `temporal.py` → Allen interval algebra + ordering/contradiction/clamp; `contradictions(events, duration)` now also flags out-of-range spans and mutually-exclusive same-entity state values over overlapping time.
- `events.py` → deterministic candidates from signals (entity presence/motion runs with direction, speech turns, OCR rows, audio events) with typed state_before/after; Spark only labels by candidate_id; `state_timeline` merges contiguous same-value intervals.
- `evidence.py` → verify gate checks span bounds + frame-id existence + frame-time-inside-span (tol 1.0 s); `calibrate_detail` routes through persisted isotonic curve; `decision` uses selective threshold from outcomes (0.5 prior when none); `abstain` carries reason.
- `investigator.py` → plan → retrieve (retrieval.py over full index) → inspect (must cite evidence ids) → gate (resolve ids to stored spans, frame-time verify, calibrate, selective abstain). No model-authored timestamps.
- `benchmark.py` → 10 adversarial tests, all derived from stored data; unevaluable tests report `skip`, never pass. Covers shuffle, deletion, audio-missing honesty, measured +2 s desync shift, cross-identity IoU, occlusion resume, OCR time attribution, reversal, empty-evidence, high-confidence abstention.
- `store.py` → adds `entity_gallery`, `calibration_points`, `outcomes` tables; sessions gains answer_evidence/confidence/verification/calibration + av_offset/confidence; entities gains global_id; qa_runs gains raw_confidence.
- `agent.py` → tracking passes frame_rows/session_id for descriptors; fusion attaches frames actually inside each span, verifies answer claims per-claim, calibrates, abstains the direct answer when ungated.
- `orchestrator.py` → measured A/V offset applied only when av_align reports a valid measurement; transcript turns + audio rows shifted to video clock; answer evidence/confidence/verification persisted.
- `app.py` → FastAPI + engine endpoints (`/events /entities /frames /ocr /audio-events /trace /tools /qa /benchmark` plus NEW `/state /alignment /calibration`, POST `/qa/{qa_id}/outcome`, POST `/outcome`); `/segmentation` returns real entity lanes; SSE carries the last 20 trace rows; unknown sessions 404.
- `test_api.py` → real-pipeline tests: 422 sandboxing, upload validation, `test_create_session_real_pipeline_completes` (tiny silent clip → completed, non-empty summary, `has_audio is False`), SSE completes.
- `frontend/` → Next.js 15: app/page.tsx (VideoWorkspace: native `<video>`, Answer card (the direct reply to the viewer prompt), summary/themes/transcript/turns, progress checklist `queued/uploading/preparing/observing/perceiving/events/understanding/verifying/completed`, live agent TracePanel sent/waiting/received), components (VideoConfigSidebar, MomentCards, MomentTimelineBar), lib/api.ts + types.ts (VideoMoment/TranscriptTurn/SessionReport). `npm run build` clean. Deleted: DrillConfigSidebar, CoachDiagnosticCards, InteractiveStage, SamVideoPlayer, TacticalSvgOverlay, ViolationTimelineBar, lib/sam.ts.
- Legacy (unit-test only, not in live path): `calibration.py`, `extractor.py`, `primitives.py`, `cli.py`, `test_engine.py`. Deleted: `planner.py`, `reviewer.py` (football + mocks).

## Change Log

### [2026-11-04] — Full capability pass: identity, alignment, retrieval, calibrated abstention, honest benchmark

**What changed:**
- `identity.py` (new) + `entities.py` + `store.py` (`entity_gallery`): one identity per SAM object id with appearance descriptors; cross-session `global_id` via cosine match within kind; running-mean gallery update in sqlite.
- `av_align.py` (new) + `audio_events.py` (`rms_envelope`/`shift_events`) + `orchestrator.py`: measured RMS-vs-motion cross-correlation (±2 s); audio rows + transcript turns shifted to video clock only on valid measurement; offset + confidence persisted and served at `/alignment`.
- `events.py` rewritten: deterministic signal-derived candidates (presence segments, motion runs with direction, speech/OCR/audio) with typed state transitions; Spark labels by candidate_id only; `temporal.contradictions` enforces state exclusivity + range.
- `retrieval.py` (new) + `investigator.py` rewritten: uniform evidence index with IDF + time-hint retrieval; inspector must cite ids; spans resolved from the index; frame-time verification; calibrated confidence; selective abstention.
- `intel.py` + `agent.py` + `schemas.py` + `orchestrator.py` + `store.py`: answer is now per-claim timestamped (`answer_claims`), each claim verified against frames inside its span; unverified/low-confidence answers abstain with reason; evidence/confidence/verification/calibration persisted and returned.
- `confidence.py` (new) + `evidence.py` rewritten + `app.py` (`/calibration`, `/outcome`, `/qa/{qa_id}/outcome`): isotonic PAVA calibration from human outcomes, Brier/ECE, selective threshold at ≤10% risk; no calibrated claim without outcomes.
- `benchmark.py` rewritten: all 10 tests derive verdicts from stored data; skips instead of fake passes; desync test measures a real +2 s shift; abstention test uses 0.99 confidence with unknown grounding.
- `test_capabilities.py` (new, 8 tests) + `test_engine2.py` (updated): instance identity, measured lag, signal-derived events, state contradictions, retrieval spans, frame-time verify, isotonic fit, benchmark honesty. 14/14 pass offline.
- `frontend/lib/types.ts` + `frontend/app/page.tsx`: Answer card shows confidence/verification/calibration + evidence spans; types mirror new EvidenceSpan/GraphEvent/QARecord/SessionReport fields.

**Why:**
- The previous engine had concept-level identities, assumed A/V sync, LLM-authored timestamps, an uncited direct answer, fixed 0.35 abstention, and three always-pass benchmark checks. Each listed capability is now backed by a measured or derived mechanism instead.

**Impact:**
- sqlite schema migrates on `init_db` (new tables + columns, old DBs backfilled). `GET /sessions/{id}` now returns answer evidence + A/V offset. New endpoints: `/state`, `/alignment`, `/calibration`, outcome feedback. Long clips cost unchanged (~8 Spark calls + ≤3 SAM streams); calibration improves only after outcomes are recorded.

### [2026-11-04] — Viewer prompt is answered directly (prompt → analysis → answer)

**What changed:**
- `intel.py`: `SYSTEM` now treats the viewer prompt as the primary question; `SCHEMA` gains required `answer`; `analyze_video` prompt asks Spark to answer the prompt first, then produce summary/themes/moments; return includes `answer`.
- `schemas.py`: `SessionReport` gains `answer: str = ""`.
- `store.py`: `sessions.answer` column + `ALTER TABLE` backfill for existing DBs.
- `orchestrator.py`: `_row_to_report` reads `answer`; completed report + `upsert_session` persist `intel["answer"]`.
- `investigator.py`: new `_analysis_context(sess)` feeds the stored `answer`/summary/themes/moments/transcript into both `_plan` and `_inspect`; verification now accepts `inferred` grounding from that prior analysis instead of requiring the answer to be visible in a still.
- `frontend/lib/types.ts`: `SessionReport.answer?: string`.
- `frontend/app/page.tsx`: red-accented "Answer" card above Summary.
- `frontend/components/VideoConfigSidebar.tsx`: label "What to look for" → "Question or focus".

**Why:** a focus like "which side played better" was only used as a lens for summary/themes, so the result never answered it. The pipeline had the full video understanding but threw the question away.

**Impact:** the completed report now leads with a direct answer to the viewer prompt. Follow-up QA in the investigator reuses the same stored analysis and can answer judgment questions that are supported by the full-video pass but not visible in six stills. Existing DBs migrate on next `init_db()`; old sessions return `answer=""` until re-run. Live tiny-clip e2e passed and returned a grounded answer; offline tests 6/6; frontend build clean.

### [2026-11-04] — Fast path: single-pass frame decode + pipelined perception

**What changed:**
- `observer.py`: replaced per-frame `iio.imread(index=…)` (one ffmpeg process per frame, ~800 ms each) with two ffmpeg passes — probe (`fps=2, scale=96:gray` raw pipe) and materialise (one balanced `select='eq(n,i)+…'` pass writing only planned JPEGs). Both passes share one temp file; probe height follows source aspect. Removed per-frame decode helper.
- `sam_client.py`: new `local_video_bytes()` — reads + validates `upload://` and local files without uploading.
- `orchestrator.py`: Files API upload now runs concurrently with WAV extract + frame observation for local inputs; transcribe + audio-event tasks start before frame decoding; `run_perception` receives the pre-started tasks + duration.
- `agent.py`: new `_understand_tool` starts `intel.analyze_video` as soon as upload + transcript are both ready (instead of after SAM tracking, often during frame decode) and `run_perception` returns that `intel_task`; `run_fusion_understanding` awaits it, with a direct-call fallback; cancels it if perception fails.

**Why:** local observation dominated wall clock — 804 ms/frame × (190 probe + 287 materialise) ≈ 380 s for a 95 s clip, and ~33 min for a 17-min clip, because every indexed read spawned ffmpeg and re-decoded from a keyframe.

**Impact:** measured on real uploads: 95 s clip 14.4 s (was ~380 s), 173 s clip 23.7 s, 1039 s clip 97.5 s (~20×). Upload and video understanding now overlap local prep/tracking; the tiny-clip live e2e dropped 50.9 s → 19.6 s. No new dependencies; 19 tests pass.

### [2026-11-04] — Full Video Intelligence Engine (adaptive observation → event graph → investigator)

**What changed:**
- New `store.py`: sqlite persistence (`data/delta.db`, WAL) for sessions/progress/frames/entities/observations/events/ocr/audio_events/qa_runs. Replaces in-memory dicts; restarts keep sessions.
- New `observer.py`: adaptive visual observation (numpy motion + histogram scene cuts; 0.5/3/15 FPS tiers, transition densification, gap-preserving) + JPEG frame store under `frames/<session>/`.
- New `entities.py`: persistent object identity — Spark proposes ≤4 trackable concepts from keyframes, one `sam-3.1` stream per concept, IoU merge, track-vs-real-world identity kept separate. SAM failure → honest empty.
- New `ocr.py`: independent OCR pipeline — sharpness-ranked keyframe still-image Spark passes + difflib multi-frame voting, frame refs preserved.
- New `audio_events.py`: deterministic non-speech event detection (impacts/loud-events/silence) from extracted WAV energy; speech still from voice-transcribe.
- New `temporal.py`: Allen interval algebra (13 relations, before/after/during/contains/overlap, gaps, ordering, contradictions, clamp). LLMs never do temporal arithmetic.
- New `events.py`: explicit event graph — Spark proposes actions/entities/evidence-frames, `temporal.validate_clamp` + contradiction report enforce determinism.
- New `evidence.py`: evidence contracts `{claim, evidence[{start,end,frame_ids}], confidence, verification}` + deterministic `verify()` gate + calibration + abstention.
- New `investigator.py`: plan-observe-reflect agent — Spark plans evidence needs, deterministic graph queries + bounded re-decode inspection, `evidence.verify` gates the answer, low-confidence → abstains.
- New `benchmark.py`: 10 offline adversarial tests (shuffle, deletion, audio removal, AV desync, similar objects, occlusion, rapid text, reversal, counterfactual, impossible) + ablation level 1–4 reporting.
- `orchestrator.py`: full 10-stage real-only pipeline (uploading→indexing→transcribing→observing→tracking→ocr→events→understanding→verifying→completed/failed); perception sub-failures degrade to empty, core failures mark `failed` with real error.
- `schemas.py`: added `EvidenceSpan/GraphEvent/Entity/Observation/QARequest/QAResponse`; `SessionReport` gains `n_frames_indexed/n_entities/n_events/ablation_level`.
- `app.py`: new endpoints `/events /entities /frames /ocr /audio-events /qa (GET+POST) /benchmark`, frame JPEG serving, 30-min SSE cap, unknown-session 404 preserved.
- `frontend/`: `EnginePanels.tsx` (event graph, entities, investigator QA, benchmark runner) wired into `page.tsx`; `types.ts/api.ts` extended; `npm run build` clean.
- `test_engine2.py`: 6 offline tests (Allen algebra, evidence gate, adaptive observer, audio detection, store roundtrip + benchmark). No key needed.

**Why:** the `transcribe+spark` shortcut was exactly the "LLM watches video and describes" design the research memo argues against — timestamps were model-estimated, no object identity, no temporal computation, no evidence gate.

**Impact:** 19 tests pass (incl. live full-engine e2e on tiny clips in ~75s); frontend builds clean. No new dependencies — numpy/PIL/imageio/scipy(stdlib sqlite) only. Long clips cost ~8+ Spark calls + ≤4 SAM streams per session by design.

### [2026-11-04] — Pivot to real-only video intelligence (transcribe + spark-contributor)

**What changed:**
- New `transcribe.py`: ffmpeg-binary WAV extract + real `POST /asr/transcribe` (DIARIZATION, WAV mono 24kHz); silent clips → empty transcript + `has_audio=False`, not an error.
- New `intel.py`: real `muse-spark-1.3-contributor` video understanding (`input_file` file_id + transcript/turns, strict json_schema, timestamp clamping, invalid-moment drop).
- `spark.py`: `SPARK_MODEL` → `muse-spark-1.3-contributor`.
- `schemas.py`: replaced CoachSessionInput/DrillAnalysisPlan/CoachingEvent with VideoSessionInput/SessionReport (summary, themes, transcript, turns, moments, on_screen_text, has_audio).
- `orchestrator.py`: rewrote `run_pipeline_async` to uploading → transcribing → understanding → completed/failed; deleted demo_tracks/evaluate_plan/SAM/kinematics path; missing key → failed (never mock).
- `app.py`: VideoSessionInput payload, moments-count completion message, `/segmentation` compat stub, SSE without lanes.
- Deleted `planner.py`, `reviewer.py` (football prompts + mock fallbacks); deleted 6 football frontend components + `lib/sam.ts`; added VideoConfigSidebar/MomentCards/MomentTimelineBar; rewrote `page.tsx` (native video + summary/transcript UI) and `lib/types.ts`/`api.ts`.
- `test_api.py`: real-pipeline tests (tiny silent MP4 → completed, summary non-empty, `has_audio is False`); `test_sam_client.py`: dropped `_build_events` test (function deleted).
- `README.md`: real-models-only quickstart.

**Why:** pivot from football drill coaching (SAM + kinematics + mock fallbacks) to general video intelligence using the two real models; user forbade all mock data. Spark reads native video + embedded audio; separate voice-transcribe gives speaker turns; grounded timestamped moments counter language-prior/shuffle shortcuts.

**Impact:** all 13 pytest pass (incl. live 12s tiny-clip pipeline); `npm run build` clean; sessions take minutes on long clips; legacy kinematics files remain test-only.

### [2026-11-04] — Live progress + failure surfacing (fix "long wait, no output, no visibility")

**What changed:**
- `orchestrator.py`: added `_progress` store + `save_progress`/`get_progress`; `run_pipeline_async` marks planning/uploading/segmenting (per-role via progress cb)/evaluating/reviewing/completed in both mock and real paths.
- `app.py`: `create_session` inits `queued` progress; `_run` logs exception and stores `failed` stage + error text (was bare except → failed report with no reason); new `GET /progress`; SSE payload adds stage/detail/error.
- `reviewer.py`: real-path per-violation Spark calls now concurrent via `asyncio.gather` (was sequential).
- `frontend/lib/types.ts` + `api.ts`: `SessionProgress` type, `fetchProgress()`, SSE event gains stage/detail/error.
- `frontend/app/page.tsx`: progress panel (stage checklist, detail, elapsed s, error), poll loop fetches report+progress together, failed surfaces server error, completed-empty shows explicit "no violations matched" note.

**Why:** with a real key, sessions run upload → plan → 4x SAM streams → reviewer (minutes), during which UI showed only `Analyzing… (Status: processing…)` and failures rendered as empty report with no reason.

**Impact:** mock-path tests unaffected (additive fields). Backend must restart (no --reload) + frontend dev server to pick up UI. `shortcut`: progress is in-memory like store/lanes — lost on restart, single worker only; upgrade when Redis lands.

### [2026-11-04] — Backend reads .env key + strict-schema fix: real SAM/Spark analysis instead of silent demo mode

**What changed:**
- `sam_client.py`: `api_key()` falls back to repo-root `.env` (`_dotenv()` cached parser, env vars still win). Previously `os.getenv`-only, so a key in `.env` was invisible to uvicorn and every session silently ran the mock/demo pipeline.
- `planner.py` / `reviewer.py`: key gating now goes through `sam_client.api_key()` (try/except `NoApiKey`) instead of raw `os.getenv` (removed now-unused `os` imports); added `_strict_schema()` (recursive `required` = all properties, incl. `$defs`) and used it for the planner's `json_schema` payload — the API 400'd (`'required' ... Missing 'calibration'`) because pydantic omits defaulted fields from `required`.
- `frontend/components/DrillConfigSidebar.tsx`: hint under the Analyze button when no clip is uploaded ("Upload a match clip above to enable analysis." / "Uploading clip…") + tooltip — the disabled button looked active, so clicks silently did nothing.
- `test_api.py`: `_force_mock_path` autouse fixture keeps mock-path tests hermetic with a real key present.

**Why:**
- The dashboard showed instant generic cards (e.g. fixed 2.07s dwell) for any clip: backend had no key in its process env, so `run_pipeline_async` always took the demo_tracks path. After wiring the key, the planner still fell back to mock on a 400 until the schema fix.

**Impact:**
- Sessions on real clips now take the full pipeline (Files upload → Frame-0 Spark plan → per-role SAM streams → Spark reviewer): minutes, not a second. Verified live: real plan grounded on Frame 0, 4 SAM lanes, 18/18 tests pass.

### [2026-11-04] — Fix 404 on /segmentation in mock path + missing favicon

**What changed:**
- `app.py` `read_segmentation`: 404 only when `get_lanes()` returns None (unknown session); empty-lanes sessions now return 200 with `{"lanes": {}}` instead of 404 "no segmentation yet".
- `frontend/public/favicon.svg`: added (dark rounded square + red ball ring); `frontend/app/layout.tsx` sets `icons: { icon: "/favicon.svg" }`.
- `frontend/next.config.ts`: `outputFileTracingRoot` set + permanent redirect `/favicon.ico` → `/favicon.svg`.

**Why:**
- Mock/demo pipeline (`run_pipeline_async` without API key) never populates lanes, so every completed demo session 404'd on `/segmentation`. Frontend catches it but Chrome still logs "Failed to load resource: 404".
- No favicon file existed, so every page load 404'd on `/favicon.ico`. (Note: `app/icon.svg` route was tried first but broke `next build` on 15.5.29 with `no direct app page entry found`; static `public/` favicon + redirect is the version-proof fix.)

**Why:**
- Mock/demo pipeline (`run_pipeline_async` without API key) never populates lanes, so every completed demo session 404'd on `/segmentation`. Frontend catches it but Chrome still logs "Failed to load resource: 404".
- No favicon file existed, so every page load 404'd on `/favicon.ico`.

**Impact:**
- `frontend/lib/api.ts` `fetchSegmentation` no longer throws in demo mode; `resultsFromSegmentation({})` returns null → video renders without masks, as intended.
- True unknown session IDs still 404 (`unknown session_id`), preserving the not-found contract `test_api.py` relies on for sessions.
