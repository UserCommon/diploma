# Accessories Processing Service

Manages the parts library — handles upload, image preprocessing, embedding generation, and semantic search for car accessories (spoilers, wheels, bumpers, hoods, mirrors, etc.).

## Responsibilities

1. Accept part image uploads via REST API
2. Remove background from uploaded images (rembg)
3. Generate vector embeddings from part name + description (sentence-transformers)
4. Store embeddings in PostgreSQL via pgvector
5. Expose semantic search so the agent can find relevant parts by natural language query

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/parts/upload` | Upload a new part image; enqueues background processing |
| `GET` | `/api/parts/` | List parts, filterable by `category` and `domain` |
| `GET` | `/api/parts/{id}` | Get a single part's metadata and image URLs |
| `DELETE` | `/api/parts/{id}` | Delete a part |
| `POST` | `/api/parts/search` | Semantic search — returns closest parts by embedding cosine similarity |

## Part Processing Pipeline (LangGraph)

When a part is uploaded, a background worker runs this pipeline:

```
remove_background_node
    → save_result_node
    → compute_embedding_node
```

1. `remove_background_node` — rembg strips the background, outputs a transparent PNG
2. `save_result_node` — saves processed image to `storage/processed/`
3. `compute_embedding_node` — encodes `name + description` with `all-MiniLM-L6-v2`, stores vector in pgvector

## Data Model (Part)

| Field | Type | Notes |
|-------|------|-------|
| `id` | UUID | Primary key |
| `name` | str | Display name |
| `category` | str | spoiler / wheels / bumper / hood / mirror / other |
| `domain` | str | car / moto / interior |
| `original_image_url` | str | Raw uploaded image |
| `processed_image_url` | str | Background-removed PNG |
| `description` | str | Optional text description |
| `status` | str | pending / ready / failed |
| `embedding` | vector | pgvector float array for semantic search |
| `created_at` | datetime | |

## Key Dependencies

| Package | Purpose |
|---------|---------|
| FastAPI | REST API |
| SQLAlchemy (async) | ORM |
| PostgreSQL 16 + pgvector | Storage + vector search |
| Redis + SAQ | Async task queue for background processing |
| rembg | Background removal |
| sentence-transformers | Embedding model (`all-MiniLM-L6-v2`) |

## Storage Layout

```
storage/
├── originals/    # raw uploaded part images
└── processed/    # background-removed PNGs
```

## Missing / Needs Work

- No image size or format validation on upload
- No deduplication (same image can be uploaded multiple times)
- Embedding model loaded lazily on first use — cold start delay on first search
- No pagination on the list endpoint
