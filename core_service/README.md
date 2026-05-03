# Core Service

LLM-powered agent that orchestrates the full car tuning pipeline. Accepts a user's natural language request and car photos, then autonomously finds parts, segments the target region, and generates the inpainted result.

## Responsibilities

1. Accept user prompt + car images
2. Use a ReAct agent (LangChain + LangGraph) to plan and execute tuning steps
3. Call `accessories_processing` to find relevant parts via semantic search
4. Run SAM2 + GroundingDINO to segment the target area on the car
5. Call an inpainting provider to generate the final result
6. Stream events back to the frontend in real time via SSE

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/agent/sessions` | Start a new agent session (car images + prompt) |
| `GET` | `/api/agent/sessions/{id}` | Get session state, events, and result image URLs |
| `GET` | `/api/agent/sessions/{id}/stream` | SSE stream — live agent thinking/tool call events |
| `POST` | `/api/jobs/create` | Create a manual inpainting job (part pre-selected, skips agent) |
| `GET` | `/api/jobs/` | List all jobs |
| `GET` | `/api/jobs/{id}` | Get job details and result variants |

## Agent Tools

The ReAct agent has three tools it can call:

| Tool | Description |
|------|-------------|
| `search_components(query)` | Semantic search over the parts library using embeddings |
| `segment_object(car_image, text_prompt)` | GroundingDINO + SAM2 to produce a segmentation mask |
| `generate_image(source, mask, prompt)` | Calls the configured inpainting provider to produce result variants |

If `OPENAI_API_KEY` is not set, a stub agent runs instead — deterministic demo flows with no LLM cost.

## Inpainting Providers

Pluggable via `INPAINT_PROVIDER` env var:

| Provider | Description |
|----------|-------------|
| `stub` | Returns source image with colored overlay — zero cost, for development |
| `sdxl` | Local SDXL pipeline via diffusers (requires CUDA/GPU) |
| `replicate` | Stable Diffusion via Replicate API (requires `REPLICATE_API_TOKEN`) |
| `polza` | FLUX.2-Pro via polza.ai API — **current primary provider** |

## Inpainting Job Pipeline (LangGraph)

When a job is created directly (not via agent), this pipeline runs:

```
load_assets_node
    → auto_mask_node        (GroundingDINO + SAM2)
    → build_prompt_node
    → run_inpainting_node   (configured provider)
    → save_results_node
```

## Data Models

### AgentSession

| Field | Type | Notes |
|-------|------|-------|
| `id` | UUID | Primary key |
| `status` | str | pending / processing / complete / failed |
| `user_prompt` | str | User's tuning request |
| `car_image_urls` | list[str] | Uploaded car photos |
| `events` | JSONB | Streamed agent events (thinking, tool calls, results) |
| `result_image_urls` | list[str] | Final generated images |
| `error_message` | str | Set if failed |

### Job

| Field | Type | Notes |
|-------|------|-------|
| `id` | UUID | Primary key |
| `status` | str | pending / processing / complete / failed |
| `source_image_url` | str | Uploaded car image |
| `auto_mask_text` | str | Text prompt for GroundingDINO (e.g. "bumper") |
| `user_prompt` | str | Additional tuning description |
| `selected_part_id` | UUID | FK to Part |
| `generated_mask_url` | str | Output mask from segmentation |
| `prompt` | str | Final prompt sent to inpainting provider |
| `result_variants` | list[str] | URLs of generated image variants |
| `error_message` | str | Set if failed |

## Key Dependencies

| Package | Purpose |
|---------|---------|
| FastAPI | REST API + SSE streaming |
| LangChain + LangGraph | ReAct agent and state-graph pipelines |
| OpenAI SDK | LLM backend for agent reasoning |
| SAM2 | Segmentation model |
| GroundingDINO | Object detection / localization |
| diffusers | Local SDXL inpainting pipeline |
| sentence-transformers | Embeddings for semantic search |
| SAQ + Redis | Async background task workers |
| PostgreSQL + pgvector | Persistence + vector search |

## Storage Layout

```
storage/
├── originals/    # uploaded car images
├── masks/        # segmentation masks from SAM2
└── results/      # final inpainted image variants
```

## Environment Variables

```env
OPENAI_API_KEY=           # if unset, stub agent is used
INPAINT_PROVIDER=polza    # stub | sdxl | replicate | polza
POLZA_AI_API_KEY=

SAM2_CHECKPOINT=          # path to SAM2 .pt file
SAM2_CONFIG=              # path to SAM2 config yaml
GROUNDING_DINO_CONFIG=
GROUNDING_DINO_CHECKPOINT=
```

## Missing / Needs Work

- No auth — public endpoints can exhaust inpainting API quota
- No rate limiting on agent sessions or job creation
- SAM2 + GroundingDINO require manual checkpoint download (not documented)
- Segmentation fallback is a crude center-box mask if models unavailable
- Stub agent always returns 2 hardcoded results regardless of user prompt
- Polza jobs timeout hard at 180s with no retry
