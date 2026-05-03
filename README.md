# AI Car Tuning Editor

An application where users upload a car photo, describe what they want changed ("add a spoiler", "swap the wheels"), and an AI agent handles semantic search, segmentation, and image inpainting to produce photorealistic results.

## Services

| Service | Description |
|---------|-------------|
| `accessories_processing` | Manages the parts library — uploads, background removal, embedding generation, and semantic search |
| `core_service` | LLM agent that orchestrates the full tuning pipeline end-to-end |
| `frontend` | React UI — Studio page is the main UX |

## API Endpoints

### Parts Library (`/api/parts`)
- `POST /upload` — upload accessory image (triggers background removal + embedding)
- `GET /` — list parts, filterable by category/domain
- `GET /{id}` — get single part metadata
- `DELETE /{id}` — delete part
- `POST /search` — semantic/vector search over parts

### Agent Sessions (`/api/agent`) — main UX
- `POST /sessions` — start a session (car image + text prompt)
- `GET /sessions/{id}` — get session state and results
- `GET /sessions/{id}/stream` — SSE stream of live agent events

### Jobs (`/api/jobs`) — manual/advanced mode
- `POST /create` — create an inpainting job with a specific part selected
- `GET /` — list all jobs
- `GET /{id}` — get job details and result image URLs

### Infrastructure
- `GET /health` — healthcheck
- `GET /storage/{category}/{filename}` — serve stored images (originals/processed/masks/results)

## Tools & Services Required

| Tool | Purpose |
|------|---------|
| PostgreSQL 16 + pgvector | Main DB + vector similarity search |
| Redis 7 | Task queue backend (SAQ workers) |
| OpenAI API | LLM powering the ReAct agent (optional — falls back to stub) |
| Polza.ai | FLUX inpainting API (primary image generation provider) |
| SAM2 | Image segmentation (requires manual checkpoint download) |
| GroundingDINO | Object detection and localization (requires manual setup) |
| rembg | Background removal for uploaded part images |
| sentence-transformers | Generates embeddings for semantic search |

## Environment Variables

```env
DATABASE_URL=postgres://...
REDIS_URL=redis://...

INPAINT_PROVIDER=stub|sdxl|replicate|polza
POLZA_AI_API_KEY=         # for polza.ai FLUX
OPENAI_API_KEY=           # for real LLM agent (omit = stub mode)

SAM2_CHECKPOINT=          # path to SAM2 .pt checkpoint
SAM2_CONFIG=              # path to SAM2 config yaml
GROUNDING_DINO_CONFIG=
GROUNDING_DINO_CHECKPOINT=

STORAGE_PATH=             # local filesystem path for image storage
CORS_ORIGINS=             # comma-separated allowed origins
```

## What's Implemented

- FastAPI backend with async SQLAlchemy and PostgreSQL
- LangGraph pipelines for part preparation and inpainting jobs
- Pluggable inpainting providers: stub / SDXL (local) / Replicate / Polza.ai (FLUX)
- LangChain ReAct agent with tools: `search_components`, `segment_object`, `generate_image`
- SSE streaming of agent events to frontend in real time
- React frontend with Studio as main user-facing page
- Docker-compose setup for all infrastructure

## What's Missing / Needs Work

**Critical:**
- No authentication — all endpoints are fully public; anyone can exhaust inpainting API quota
- No rate limiting on expensive external calls (Polza, OpenAI)
- No image validation — no size limits or format checks before processing
- SAM2 and GroundingDINO setup is manual and undocumented (checkpoint download, path config)

**Frontend:**
- `Editor.jsx` and `Gallery.jsx` pages are incomplete/unused
- No UI for correcting bad segmentation masks
- No batch processing, favorites, or history

**Operational:**
- No monitoring or error alerting
- DB migration file ordering unclear
- `VITE_API_BASE` hardcoded to localhost as frontend fallback
- No OpenAPI/Swagger docs published

## Running Locally

```bash
docker-compose up
```

Backend available at `http://localhost:8000`, frontend at `http://localhost:5173`.

For the agent to use a real LLM, set `OPENAI_API_KEY`. Without it, the stub agent runs deterministic demo flows at no cost.
