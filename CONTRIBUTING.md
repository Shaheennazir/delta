# CONTRIBUTING.md — delta

## Conventions

- Preserve behavior: no API/schema/DB, prompt, threshold, or abstention-logic
  changes without explicit approval. Structural fixes that need behavior change
  stop and get documented instead.
- One module, one responsibility (see `ARCHITECTURE.md`). Prefer editing the
  existing module over new files; prefer stdlib/existing deps over new deps.
- Names: domain terms (`entity_id`, `global_id`, `evidence_frames`,
  `verification`), no new ambiguous abbreviations.
- Types: explicit signatures; no new `Any`; domain models stay separate from
  transport schemas in `schemas.py` (Pydantic `extra=forbid` on inputs).
- Errors: consistent types, context preserved; expected (perception gaps) →
  honest empty + trace `failed`; unexpected/critical → `failed` with real error,
  never silent empty. No new bare `except:`; no `except Exception` that hides
  errors outside bounded per-row degradation (log to trace).
- Logging: `logging.getLogger("delta.<module>")`, structured messages; no
  `print` in backend live path (`cli.py`/`benchmark/long_rvos.py` exempt);
  never log secrets or keys.
- Config: env via `sam_client.api_key()` order
  (`META_API_KEY`→`MODEL_API_KEY`→`SAM_API_KEY`→`.env`); no secrets in code;
  keep `.env.example` accurate; no new scattered hardcoded tunables — put
  constants at module top with units (e.g. `OCR_GROUP_WINDOW_S`).

## Structure

Flat root modules (no packages). Add a module only for a genuinely new
responsibility with ≥2 callers or a clear boundary (perception / reasoning /
persistence). Do not create single-trivial-file directories. Update `MAP.md`
(change log) with every behavior-relevant change.

## Workflow

1. Read the execution path (`app.py` → `orchestrator.py` → `agent.py` →
   perception → `events.py` → `intel.py` → `evidence.py`) before editing.
2. Grep all callers of any function you touch; fix root cause in shared code;
   update imports + tests + `MAP.md`.
3. Keep diffs small and reviewable; no broad rewrites or "preferred"
   reimplementations of working subsystems.

## Testing (required)

```bash
python -m pytest -q
cd frontend && npm run build
```

- Every structural change: full suite green (or pre-existing failure noted with
  evidence it predates the change).
- New non-trivial logic (branch/loop/parser/money/security/pipeline): add or
  extend an offline test; protect temporal invariance, identity continuity,
  interaction evidence, OCR provenance, frame resolution, verification,
  abstention, failure reporting, degraded perception.
- Never weaken tests to make a refactor pass. Benchmark honesty holds:
  unevaluable → `skip`, never pass.
