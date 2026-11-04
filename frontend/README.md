# delta Coaching Dashboard (Phase 3)

Next.js 15 review UI: drill builder form, mask video player, SVG telemetry HUD, violation timeline, diagnostic cards.

## Run

```bash
npm install
npm run dev        # http://localhost:3000
```

Backend first (separate terminal):

```bash
cd .. && python -m uvicorn app:app --port 8000
```

Configure with `.env` (see `.env.example`): `NEXT_PUBLIC_API_BASE=http://localhost:8000`.

## Mask SDK

`@meta-sam/react` + `@meta-sam/parser` are real dependencies. `SamVideoPlayer`
renders `<Video result>` with parsed segmentation; `lib/sam.ts` parses the
backend's raw output-text lanes per role and merges them (object ids namespaced
per role, e.g. `ball:0`, since the same wire id in two streams names different
objects). Overlay anchor positions in `TacticalSvgOverlay` use the backend's
real foot-contact anchors (`hud_overlay.anchor_x/y`) when present and fall back
to deterministic placeholders on mock reports.

## Upload flow

The sidebar uploads MP4 clips (max 60s, max 1080p, validated server-side) via
`POST /api/v1/uploads` and receives an `upload://` handle. The clip plays
locally from an object URL while the backend resolves the handle, uploads to
the Model API Files endpoint, and streams sam-3.1. Live progress arrives over
SSE (`/sessions/{id}/stream`); masks load from `/segmentation` as lanes grow.
