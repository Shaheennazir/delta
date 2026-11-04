# delta — Video Intelligence Engine (real models only)

Upload any MP4. Backend runs a full evidence-grounded engine on real Meta models, no mocks:

1. `muse-voice-transcribe-1.0` — speech transcript + speaker turns; deterministic
   energy/onset analysis adds non-speech audio events.
2. Adaptive visual observation — motion/scene-cut driven sampling (0.5/3/15 FPS
   tiers) into a timestamped frame store, never silent gaps.
3. `sam-3.1` — persistent entity registry (detection + tracking + IoU identity).
4. Independent video-OCR pipeline — sharp-frame still-image passes + multi-frame voting.
5. Explicit event graph — Spark proposes, Allen interval algebra disposes.
6. `muse-spark-1.3-contributor` — answers the viewer prompt directly, then summary,
   themes, grounded moments, on-screen text.
7. Evidence verifier — every factual answer carries timestamps + frame refs or abstains.
8. Investigator agent — plan → observe → reflect QA over the event graph.
9. Adversarial benchmark — 10 offline tests (shuffle, deletion, desync, reversal…).

```bash
python -m pip install -r requirements.txt
export META_API_KEY=...   # never commit
python -m uvicorn app:app --port 8000
cd frontend && npm install && npm run dev  # http://localhost:3000
```

Flow: `POST /api/v1/uploads` → `POST /api/v1/sessions {title, focus, video_url}` →
poll `GET /api/v1/sessions/{id}` + `/progress`, or SSE `/stream`.
Stages: `queued → uploading → transcribing → observing → tracking → ocr → events → understanding → verifying → completed/failed`.
Then: `GET …/events`, `…/entities`, `…/frames`, `…/ocr`, `…/audio-events`,
`POST …/qa {question}`, `POST …/benchmark`.
Failures return `failed` with the real error in `/progress` — never fake data.
Perception sub-steps (tracking/OCR/event proposal) degrade to honest empty on
error and lower the ablation level; they never invent data.
