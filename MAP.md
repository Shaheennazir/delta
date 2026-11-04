# MAP.md — delta (Video Intelligence, real-only)

- `chat.py` → NEW: session chat router — Spark classifier picks followup (answer from stored analysis only) | investigate (`investigator.answer_question`, retrieval-gated) | reanalyze (fresh `intel.analyze_video` on the same upload + frame-verified spans); every route persists to `qa_runs` and logs `chat_router`/`reanalyze` to the trace store; classifier failure defaults to investigate.
- `frontend/app/page.tsx` → VideoChat (replaces VideoWorkspace): landing hero (upload + question + presets) → live status bubble (stage pills + elapsed) → answer bubble with seekable evidence chips → follow-up composer; header `engine` toggle reveals TracePanel + event graph + entities + transcript; `QaPanel`/`BenchmarkPanel` no longer rendered (APIs unchanged).
- `frontend/components/EnginePanels.tsx` → `EventList` (event cards with source badge + stored evidence frame ids + seek; empty-frame state reads "no retrievable frames"), `QaPanel` (investigator input + abstained banner + verification/confidence/calibration + clickable evidence spans), `BenchmarkPanel` (runs 11 adversarial tests), `TracePanel`, `EntityList`.
- `frontend/lib/api.ts` → adds `sendChat(sessionId, message)` (POST `/chat`), `listSessions()` (GET `/sessions`), `uploadUrlToHttp()` (`upload://name` → `/api/v1/uploads/name`); rest unchanged.

- `schemas.py` → Pydantic v2 contracts: VideoSessionInput, SessionReport (answer + answer_evidence/answer_confidence/answer_verification/answer_calibration + av_offset_s/av_sync_confidence), EvidenceSpan (source_id/kind/verification), AnswerClaim, GraphEvent (candidate_id/source), QAOutcome/QAResponse(abstained).
- `identity.py` → instance-level appearance descriptors (4x4 spatial colour + saturation-weighted hue + gradient orientation, unit-norm) from stored frame crops; cosine match within kind to persistent `global_id` via `entity_gallery`; running-mean descriptor update. CONSTRAINED stitching: `resume_plausible` (drift ≤2+dt diags, size 0.33–3) + `stitch_tracklets` with explicit rejections (temporal overlap/shared frame, impossible displacement, appearance <thr, simultaneous-track conflict, scene-cut without IoU>0.1+appearance ≥0.90, gap>5s or gap>2.5s without appearance ≥0.85); stage-1 same-gid merges ONLY when `_can_merge` passes else forks unknown id; `assign` resolves simultaneous same-gid conflicts (strongest keeps id) and forwards scene-cut times from frame_rows.
- `events.py` → deterministic candidates from signals (entity presence/motion runs with direction, speech turns, OCR rows, audio events) with typed state_before/after; Spark labels by candidate_id only; `state_timeline` merges contiguous same-value intervals. PLUS `build_interactions(entities, observations, frame_rows=None)` (grounded: same-frame co-presence, contact requires IoU>0.02 — proximity alone is near with dist<1.2; runs linked ≤1s, ≥2 co-present frames required) with explicit SAM->stored mapping `_map_sam_to_stored` (exact index match only, no nearest-neighbour fallback; runs with <2 mapped frames rejected); emits `interaction`-source candidates with entity_ids=[a,b], `evidence_frames`=stored ids + `sam_evidence_frames`=raw SAM fis; wired into `build_candidates(..., frame_rows)` + preserved through `extract_events`.
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
- `ocr.py` → independent still-image OCR passes (concurrent gather, 90s per-image budget) + bounded-window voting (similarity ≥0.8 AND span ≤OCR_GROUP_WINDOW_S=2.0s); output keeps best row's own t+frame_index so every citation points at its detection frame.
- `entities.py` → instance-level registry: one candidate per SAM object id (never per concept stream) split by `_split_implausible_tracklets` at resume-implausible gaps (via `identity.resume_plausible`; preserves `sam_track_id`/`sam_stream` provenance, keeps all fragments, separate identities re-identified only via `stitch_tracklets`), IoU merge at 0.65, then `identity.assign` binds each instance to cross-session `global_id` via appearance gallery. `track_session` runs one SAM stream per concept concurrently with one safe retry (transient failures retried once after 2s; `NoApiKey`/`ValueError` fail fast; `CancelledError` propagates) and honest-empty on final failure. PLUS honest coverage diagnostics: `_coverage_log` records concepts/streams-ok/objects/boxes/candidates/merged-away/empty-dropped/split_frags/retried/retry_ok/stream_errors/entities/obs to log + trace (thresholds NOT lowered).
- `audio_events.py` → deterministic non-speech detection + `rms_envelope` (50 ms hop) and `shift_events` (audio-clock → video-clock) used by av_align.
- `temporal.py` → Allen interval algebra (timestamps normalised to ms via `_norm_ts` before every relation) + deterministic ordering `order_events` as pure function of (norm start, norm end, event_id/candidate_id) — permutation-invariant, never input position; `contradictions(events, duration)` flags out-of-range spans and mutually-exclusive same-entity state values over overlapping time.
- `events.py` → deterministic candidates from signals (entity presence/motion runs with direction, speech turns, OCR rows, audio events) with typed state_before/after; Spark only labels by candidate_id; `state_timeline` merges contiguous same-value intervals.
- `evidence.py` → verify gate checks span bounds + frame-id existence + frame-time-inside-span (tol 1.0 s); `calibrate_detail` routes through persisted isotonic curve; `decision` uses selective threshold from outcomes (0.5 prior when none); `abstain` carries reason.
- `investigator.py` → plan → retrieve (retrieval.py over full index) → inspect (must cite evidence ids) → gate (resolve ids to stored spans, frame-time verify, calibrate, selective abstain). No model-authored timestamps.
- `benchmark/` → Long-RVOS mini-benchmark harness (`long_rvos.py`: select/fetch/build/run over `valid` split; 15 clips ≤300 annotated frames ranked by objects+expressions+dynamic; 640px mp4s in `clips/`; per-video engine+`benchmark.run_all` JSON in `results/`, overridable via `BENCH_RESULTS_DIR` so v0.2 reruns land in `results_v02/` with v0.1 frozen in `results_v01/`). `shortcut`: no mask-IoU scoring (engine outputs boxes/spans, not masks).
- `benchmark.py` → 11 adversarial tests, all derived from stored data; unevaluable tests report `skip`, never pass. Occlusion triple-checked (collision + independent descriptor cosine ≥0.50 from stored crops + `resume_plausible`); `interaction` test (precision: every ≥2-entity event needs a co-present close frame; recall: sustained ≤1s-linked ≥0.3s proximity needs a covering event). frame_shuffle checks 5 permutations for identical full ordering. Report adds `applicable`/`evaluated`/`coverage[]` (per-test reason) + `metrics` (track gaps, implausible resumes, appearance switches, identity_precision_proxy, identity_recall=None with note, events_checkable/unsupported + unsupported_event_rate). Covers shuffle, deletion, audio-missing honesty, measured +2 s desync shift, cross-identity IoU, occlusion resume, OCR time attribution, reversal, interaction precision/recall, empty-evidence, high-confidence abstention.
- `test_capabilities.py` → 10 tests (was 8): NEW `test_occlusion_bridges_fragmented_tracklets` (same-appearance fragments merge, far jump does not) + `test_interactions_detect_contact_and_no_hallucination` (contact emitted, far pair silent); benchmark total 11 asserted.
- `store.py` → adds `entity_gallery`, `calibration_points`, `outcomes` tables; sessions gains answer_evidence/confidence/verification/calibration + av_offset/confidence; entities gains global_id; qa_runs gains raw_confidence.
- `agent.py` → tracking passes frame_rows/session_id for descriptors; fusion attaches frames actually inside each span, verifies answer claims per-claim, calibrates, abstains the direct answer when ungated.
- `orchestrator.py` → measured A/V offset applied only when av_align reports a valid measurement; transcript turns + audio rows shifted to video clock; answer evidence/confidence/verification persisted.
- `app.py` → FastAPI + engine endpoints (`/events /entities /frames /ocr /audio-events /trace /tools /qa /benchmark` plus NEW `/state /alignment /calibration`, POST `/qa/{qa_id}/outcome`, POST `/outcome`, POST `/chat` → `chat.answer_chat`, 409 until completed); `/segmentation` returns real entity lanes; SSE carries the last 20 trace rows; unknown sessions 404.
- `test_api.py` → real-pipeline tests: 422 sandboxing, upload validation, `test_create_session_real_pipeline_completes` (tiny silent clip → completed, non-empty summary, `has_audio is False`), SSE completes.
- `frontend/` → Next.js 15: app/page.tsx (VideoWorkspace: native `<video>`, Answer card (the direct reply to the viewer prompt), summary/themes/transcript/turns, progress checklist `queued/uploading/preparing/observing/perceiving/events/understanding/verifying/completed`, live agent TracePanel sent/waiting/received), components (VideoConfigSidebar, MomentCards, MomentTimelineBar), lib/api.ts + types.ts (VideoMoment/TranscriptTurn/SessionReport). `npm run build` clean. Deleted: DrillConfigSidebar, CoachDiagnosticCards, InteractiveStage, SamVideoPlayer, TacticalSvgOverlay, ViolationTimelineBar, lib/sam.ts.
- Legacy (unit-test only, not in live path): `calibration.py`, `extractor.py`, `primitives.py`, `cli.py`, `test_engine.py`. Deleted: `planner.py`, `reviewer.py` (football + mocks).

## Change Log

### [2026-11-05] — Chat-first UI + intent-routed follow-ups (chat box, live session, engine toggle)

**What changed:**
- `chat.py` (new): `answer_chat(session_id, message)` — Spark intent classifier → followup (stored analysis only) / investigate (`investigator.answer_question`) / reanalyze (fresh `intel.analyze_video` + span verify + calibrate); all persist to `qa_runs`, all log to trace.
- `app.py`: new `POST /api/v1/sessions/{session_id}/chat` (`QARequest` body, 409 until completed).
- `frontend/lib/api.ts`: new `sendChat()`.
- `frontend/app/page.tsx` (rewritten): landing hero → live status bubble → answer + follow-up thread; header `engine` toggle for trace/events/entities/transcript; evidence chips seek the video.

**Why:**
- The old workspace was a dashboard (sidebar + panels), not a chat box: no clean upload→ask→watch→follow-up flow, follow-ups needed the buried QaPanel, and engine internals were always on screen.

**Impact:**
- Analysis is saved per session in sqlite (unchanged) and reused by all three chat routes; `QaPanel`/`BenchmarkPanel` unmounted but their APIs still work; `npm run build` clean, backend route verified registered.

### [2026-11-05] — Multi-session chats + persistent state (sidebar, restore, replayable video)

**What changed:**
- `store.py`: new `list_sessions()` (newest-first id/title/status/focus/video_url; excludes `bench_*` harness rows via `NOT GLOB 'bench_*'`).
- `app.py`: new `GET /api/v1/sessions` (chat index) + `GET /api/v1/uploads/{name}` (serves `uploads/` MP4s with the same traversal guard as frames, so old sessions stay playable).
- `frontend/app/page.tsx`: ☰ sidebar (New chat + per-session status dot/title/ago), `openSession()` hydrates report + full QA thread + engine data, still-running sessions resume the poll loop, last session id in localStorage and auto-restored on reload.
- `frontend/lib/api.ts`: `listSessions()`, `uploadUrlToHttp()`; non-`upload://` videos get no player src instead of a broken path.

**Why:**
- State already survived in sqlite but the UI was single-session and died on reload: no session index, no history view, and blob-URL video was unplayable after refresh.

**Impact:**
- Sidebar lists user chats only (benchmark/`bench_*` rows hidden, APIs unchanged); videos uploaded via the app replay forever from `uploads/`; `npm run build` clean, both endpoints verified live.

### [2026-11-05] — v0.3 live regression: track split + evidence mapping + retry

**What changed:**
- `benchmark/results_v03/`: 15 fresh sessions (latest code: split + mapping + safe retry; same 15 clips/focus/models as v0.2); v0.2 frozen in `results_v02/`, v0.1 in `results_v01/`.

**Why:**
- Offline correctness was verified per change; live impact was explicitly outstanding. Same-conditions rerun isolates the pipeline effect (model-run stochasticity + SAM variance remain as noise confounds).

**Impact:**
- Occlusion 1/7/7 → 7/0/8: implausible resumes 14 → 0 across all clips; 235ff86d37 gaps 8 → 3, collisions 2 → 0. Tracklet splitting fixed the SAM-native jumps live, not just offline.
- Evidence resolution: unsupported_event_rate 0.0 on every checkable clip (235ff86d37 0.1875(15) → 0.0(0); 24b7b95ca4 0.1(4) → 0.0; 5d262b025c, 6fb7a04167, bda2f20019, fb8caa28e4 all → 0.0). The 15/80 flock failures are gone.
- Entity counts rose on split clips (0ddb 1 → 8, 235ff 9 → 14, 5d26 3 → 11, 6fb 4 → 22, b41 2 → 8, bda 2 → 5, fb8 2 → 7); fell on SAM-variance clips (24b 8 → 1, 6da 2 → 0). Zero-entity transport clips unchanged (18b160d3a7, d33c8343bc, c4638cc87e, 0d0ebe00cc).
- Interaction 6/1/8 → 3/2/10 with 0 hallucinated in every evaluated clip (precision held). Recall regressed on volume: 235ff 5 → 6 missed, 6fb 0 → 12 missed over 23 multi-events; evaluated set churned with SAM variance (24b/6da/fb8 pass → skip, 5d2c skip → pass). Fragmentation buys occlusion/evidence wins at interaction-recall cost — next research question, not a silent win.
- SAM retry: 10 final (attempt 2/2) failures in the v0.3 log, 0 retry conversions. Retry is honest (bounded, fail-fast on permanent, empty on final failure) but persistent 3/3 transport aborts are not rescued by one retry. Transport reliability stays outstanding.
- Answers: completed 15/15; answer_verification supported 14 → 12 (unknown 1 → 3: 0ddb35f585, 1088228079, 6da16715cd). No outcomes recorded, so calibration stays "uncalibrated" and the abstention gate is unchanged — not claimed as calibrated.

### [2026-11-05] — Investigation workspace evidence polish (frontend)

**What changed:**
- `frontend/components/EnginePanels.tsx`: `EventList` outer button → div+role/button (valid nested evidence buttons), added source badge + stored frame ids + seek / "no retrievable frames" empty state; `QaPanel` added abstained banner + calibration suffix; `BenchmarkPanel` label fixed 10 → 11 tests.
- `frontend/app/page.tsx`: answer evidence text → clickable seek buttons with span start/end + first frame ids.

**Why:**
- Interaction evidence is now stored-frame-resolved or explicitly unmapped; the workspace must show retrievability instead of hiding it, and every evidence citation must seek.

**Impact:**
- No API change; existing `GraphEvent.evidence` / `answer_evidence` shapes reused. `npm run build` clean; 16 offline tests pass.

### [2026-11-05] — Safe retry for failed SAM streams (entities)

**What changed:**
- `entities.py`: `track_session.one` retries each concept stream once after a 2s pause; `NoApiKey`/`ValueError` fail fast without retry, `CancelledError` propagates; counts `streams_retried`/`retry_ok` and collects per-stream `stream_errors`; `_coverage_log` message + trace now carry `retried/retry_ok` and the first stream error.

**Why:**
- v0.2 zero-entity clips (18b160d3a7, d33c8343bc, c4638cc87e, 0d0ebe00cc, 1088228079) all showed concepts proposed but 0/1–3 SAM streams ok — transport aborts, not concept/threshold failures. A single transient abort wiped the whole registry.

**Impact:**
- One transient abort no longer permanently loses a concept; permanent errors still fail fast to honest-empty (never invented boxes). Worst-case wall clock grows by one extra per-concept budget +2s, streams stay concurrent. 16 offline tests pass; mocked retry-then-ok / permanent-no-retry / persistent-fail self-checks pass.

### [2026-11-05] — Map interaction evidence to stored frames (events)

**What changed:**
- `events.py`: `build_interactions(entities, observations, frame_rows=None)` + `_map_sam_to_stored` (exact SAM fi → stored frame_index match, no nearest fallback); runs with <2 mapped frames rejected; output carries `evidence_frames` (stored) + `sam_evidence_frames` (raw provenance). `build_candidates` passes `frame_rows` through and preserves both fields on interaction candidates.

**Why:**
- 15/80 interaction events on the flock clip were geometrically grounded but cited raw SAM frame indices absent from the sparse observer frame store, failing `evidence.verify`. Nearby-arbitrary substitution would fake retrievability.

**Impact:**
- Mapped interaction evidence now verifies against the frame store; unmapped runs are dropped (recall cost) instead of emitting unverifiable evidence (precision gain). `frame_rows=None/[]` preserves raw ids (offline tests unchanged). 16 offline tests pass.

### [2026-11-05] — Split implausible SAM tracklets (entities)

**What changed:**
- `entities.py`: new `_split_implausible_tracklets` (cut wherever `identity.resume_plausible` rejects the hop: gap >5s or drift/size impossible); `_objects_to_candidates` emits one candidate per fragment (`c{stream}_{j}_{oid}_p{k}` when split) with `sam_track_id`/`sam_stream` + per-observation `sam_track_id` provenance; `_coverage_log` adds `split_frags` count.

**Why:**
- SAM-native tracks contained 6.8–23s jumps / spatially impossible teleports that contaminated the event graph. Stitching (`appearance ≥0.75` + spatial plausibility) could not have created them.

**Impact:**
- Bad SAM tracks no longer span implausible gaps; fragments keep separate identities and re-merge only via independent appearance+spatial evidence. All fragments preserved as observations (never discarded). 16 offline tests pass.

### [2026-11-05] — v0.2 live regression: 15/15 rerun, per-fix verdicts + GT identity rescoring

**What changed:**
- `benchmark/results_v02/`: 15 fresh sessions (new code, same clips/focus/models) + `results_summary.json`; v0.1 frozen in `results_v01/`.
- No code changes for the rerun except the earlier `BENCH_RESULTS_DIR` harness override (same `_run_one` path, same 11 tests).

**Why:**
- Candidate v0.2 needed live-video proof, not just offline tests. Identical conditions except model-run stochasticity (noted as noise confound below).

**Impact:**
- Shuffle 2/7/6 → 11/0/4 (P0 fixed; rescore of old sessions already showed 9/0/6 from the tie-break alone). Rapid-text 2/2/11 → 5/0/10 (P3 fixed, 0 mismatches). Interaction 1/1/13 → 6/1/8 with 0 hallucinated in all 15 clips (P2 precision fixed; the 1 fail is recall-only: 5 missed pairs on fragmented 235ff86d37). Occlusion 0/5/10 → 1/7/7: all 14 remaining implausible gaps are SAM-native track jumps (gaps >5 s or spatially impossible — the stitcher requires spatial plausibility + appearance ≥0.75, so it could not have created them); 0 appearance switches and 0 collisions outside 235ff86d37. GT target coverage (IoU≥0.5): fb8caa28e4 0→248/249, 6fb7a04167 0→200/213, bda2f20019 → 235/235 with switches 1→0, b4192dbf2b held 58/66; regressions: 0ddb35f585 234/241→0/241 (SAM returned 1/3 streams, 1 object — stitcher had 1 candidate, nothing to merge/split), 5d2c735adb 182→94/235 (single-object SAM variance), 235ff86d37 31→103/104 but across 3 ids with 4 switches (strictness trades single-id precision for coverage). Still zero-entity: 18b160d3a7, d33c8343bc, c4638cc87e, 0d0ebe00cc (all: concepts proposed, 0/1–3 SAM streams ok — transport aborts, not concepts/thresholds), 1088228079 (1 concept, 0/1 streams). Similar_objects 2/0/13 → 7/1/7: the 1 fail is a flock-crossing frame (IoU 0.995, two 'second roof bird' tracks momentarily coincident) — benchmark can't distinguish crossing from duplication. New sub-issue found, not fixed: interaction evidence cites raw SAM frame indices, but the store holds sparse observer-sampled frames, so 15/80 events on 235ff86d37 (all interaction-source) fail frame-store resolution while geometrically grounded (unsupported_event_rate 0.1875; ≤0.1 elsewhere, 0.0 on 7 clips). Single-frame close-pair pool still 0/15 — the ≥2-frame rule adds no measured recall cost. Proposed next (not implemented): split SAM tracklets at resume-implausible gaps in `entities.py`; map interaction `evidence_frames` to stored frame indices in `build_candidates`; no threshold changes.

### [2026-11-05] — v0.2 candidate assessment: stored-data root-cause proof + GT identity baseline

**What changed:**
- `events.py`: `build_interactions` dedupes entity ids + skips `a_id == b_id` pairs (duplicate registry rows can never form a pair); `state_timeline` orders via `temporal.order_events` (was raw float sort — downstream consumer now uses normalized timestamps).
- `benchmark/results_v01/`: frozen copy of the 15 baseline result JSONs (v0.1 baseline preserved; live `results/` untouched by rescoring).
- No benchmark expectation changes: same 11 tests, same thresholds; only additive `applicable`/`evaluated`/`coverage`/`metrics` fields.

**Why:**
- Stored baseline sessions proved the interaction root cause: pipeline-time duplicate `global_id` rows produced self-pair `(g,g)` IoU=1.0 "contacts" — 6 in 0ddb35f585 (the interaction fail) and 28 in 235ff86d37 (masked as skip: only 1 distinct observed id). Rebuilding interactions from stored observations with the fixed code: 0ddb35f585 → 3 contacts, 0 self-pairs; 235ff86d37 → 0. The P1 `assign` conflict-fork prevents recurrence; the `a_id == b_id` guard makes it structurally impossible.
- Rescoring old sessions with new code isolates the benchmark-code effect: all 7 frame_shuffle fails → pass (P0 tie-break on real data); occlusion 0/5/10, rapid_text 2/2/11, interaction 1/1/13 unchanged — these need a live pipeline rerun, not a re-score.
- GT identity scoring (Long-RVOS Annotations tar, normalized-box IoU≥0.5, ±0.5 s window): 4/15 clips track the target well single-identity (0ddb35f585 234/241, bda2f20019 218/235, 5d2c735adb 182/235, b4192dbf2b 58/66); 235ff86d37 partial (31/104); 3 clips track the WRONG objects (313/147/75 obs, 0 coverage — concept-proposal bottleneck, not SAM); 6 clips zero-obs (need `_coverage_log` from rerun to split concept vs SAM failure). Single-frame close-pair pool is 0/15 — the ≥2-frame rule adds no measured false negatives on baseline.

**Impact:**
- v0.1 baseline = saved `results/` JSONs + rescored table above (new code on old sessions). v0.2 verdict requires the live 15-clip rerun to `results_v02/` (API verified reachable: sam-3.1 + muse-spark-1.3-contributor listed); GT + single-frame scripts rerun unchanged on v0.2 sessions. Residual risks carried forward: IoU>0.02 may miss small/distant contacts; 2.0 s OCR window may merge fast-changing similar text; `state_timeline` merge gap (0.5 s) still raw-float — cosmetic only.

### [2026-11-05] — P0–P4 correctness fixes: temporal determinism, constrained identity, grounded interaction, OCR provenance, coverage honesty

**What changed:**
- `temporal.py`: added `_norm_ts` (ms rounding) + `_eid`; `relation` normalises inputs first; `order_events` keys on (norm start, norm end, event_id/candidate_id); `find_*` use normalised times.
- `identity.py`: added `LONG_GAP_S/LONG_GAP_APPEARANCE_THR/CUT_APPEARANCE_THR`, `_can_merge` (rejects temporal overlap, shared-frame conflict, impossible displacement, incompatible appearance, scene-cut without strong evidence, long gap without strong evidence); stage-1 same-gid merges only via `_can_merge` else forks unknown id; stage-2 uses `_can_merge`; `assign` resolves simultaneous same-gid conflicts and forwards scene-cut times.
- `events.py`: `build_interactions` contact requires IoU>0.02 (dist alone is near); runs with <2 co-present frames rejected (before-and-after evidence).
- `ocr.py`: added `OCR_GROUP_WINDOW_S=2.0`; voting groups only within window + similarity ≥0.8 in time order; output keeps best row's own t+frame_index (provenance).
- `entities.py`: added `_coverage_log` (concepts, streams launched/ok, SAM objects/boxes, candidates, merged-away, empty-dropped, entities/obs) to log + trace; no thresholds lowered.
- `benchmark.py`: frame_shuffle checks 5 permutations + full key equality; `run_all` returns `applicable`/`evaluated`/`coverage[]` + `metrics` (identity_precision_proxy, identity_recall=None+note, unsupported_event_rate via evidence gate).

**Why:**
- Frame-shuffle failures came from stable-sort ties (input-position signal); identity failures from stage-1 unconditional same-gid merges; interaction hallucinations from dist-only contact + single-frame runs; OCR mismatches from min-t/best-frame mixing across unbounded windows; entity gaps were silent.

**Impact:**
- Identical timestamps now order by id deterministically; lookalike same-gid fragments fork instead of merging; proximity without overlap is near; single-frame interactions dropped; OCR rows always cite their detection frame; 0-entity sessions carry a diagnosable coverage line; benchmark reports effective sample size + measurable proxies (16 offline tests pass).

### [2026-11-05] — Expert occlusion recovery + real interaction understanding

**What changed:**
- `identity.py`: added `BRIDGE_APPEARANCE_THR=0.75`/`BRIDGE_MAX_GAP_S=5.0`, `_box_iou/_center/_diag`, `resume_plausible`, `stitch_tracklets` (same-id merge + greedy same-kind gap bridge with appearance+spatial gate); `assign` now computes `_desc` per candidate, bridges, then updates gallery once per merged entity.
- `events.py`: added `CONTACT_IOU/CONTACT_DIST/NEAR_DIST/MIN_INTERACT_S/MAX_LINK_GAP_S`, `_pair_iou/_pair_metric`, `build_interactions` (contact/near runs with exact observation timestamps, confidence from pair confidence + duration + peak IoU); `build_candidates` emits interaction candidates with `entity_ids=[a,b]`, typed contact/proximity state, `source="interaction"`.
- `benchmark.py`: occlusion test triple-checks collision + recomputed appearance cosine (<0.50 = switch) + `resume_plausible`; NEW `interaction` test (hallucinated vs missed); total 10 → 11.
- `test_capabilities.py`: NEW tracklet-stitching + contact tests; totals 10 → 11. `test_engine2.py`: total 10 → 11.

**Why:**
- Occlusion was collision-absence only (tautological: gaps are defined per-entity, so resume was assumed). Interaction was presence/motion only with no contact predicate and no benchmark coverage.
- Now reappearance is independently verified (appearance + motion continuity) and fragmented SAM tracklets are explicitly reunited; contact/proximity are measured geometry with precision+recall enforcement.

**Impact:**
- `entities.track_session` output shrinks when fragments merge (fewer entities, longer observations); `save_entities` REPLACE semantics unchanged. Benchmark `total` is 11 — callers asserting 10 must update. All 29 tests pass (16 offline capability/engine + live API).

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
