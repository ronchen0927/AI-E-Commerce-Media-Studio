# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install dependencies
uv sync

# Run API server
uv run uvicorn app.main:app --reload

# Run Celery worker (Linux/macOS)
uv run celery -A app.core.celery_app worker --loglevel=info

# Run Celery worker (Windows)
uv run celery -A app.core.celery_app worker --pool=solo --loglevel=info

# Run tests
uv run pytest -v

# Run a single test file
uv run pytest tests/test_api.py -v

# Run a single test
uv run pytest tests/test_api.py::test_health_check -v

# Lint
uv run ruff check .

# Type check
uv run mypy .
```

## Architecture

**Request flow:** Client → FastAPI (`POST /api/v1/upload`) → Redis → Celery worker → AI service → Storage → Client polls `GET /api/v1/task-status/{id}`.

**Two processing modes** selected at upload time via the `mode` form field:
- `remove_bg` — RMBG-1.4 background removal → outputs transparent PNG
- `edit` — FireRed-Image-Edit-1.1 instruction-based editing → outputs PNG

**AI service strategy (both modes):** Replicate API → Custom cloud API → local GPU model. Configured via env vars; if no API token is set, falls back to local inference automatically.

### Key files

| File | Role |
|---|---|
| `app/core/config.py` | All settings via `pydantic-settings`; `get_settings()` is `@lru_cache` — call `get_settings.cache_clear()` in tests that mutate settings |
| `app/api/routes.py` | FastAPI endpoints; `task_metadata_store` is an **in-memory dict** (task metadata only — actual task STATUS comes from Celery/Redis via `AsyncResult`) |
| `app/tasks/image_processing.py` | Celery task `process_image`; bridges async AI services to sync Celery via `run_async()` helper; uses a temp dir per task that is always cleaned up |
| `app/services/ai_service.py` | `BackgroundRemovalService` and `FireRedEditService` (both extend `AIService`); AI models are **lazy-loaded** into module-level globals (`_rmbg_model`, `_firered_pipe`) on first use |
| `app/services/storage.py` | `StorageService` interface with `LocalStorage` and `GCSStorage` implementations |
| `app/core/auth.py` | API key + JWT auth and rate limiting; both are **feature-flagged off by default** (`AUTH_ENABLED=false`, `RATE_LIMIT_ENABLED=false`) |

### Notable implementation details

- **RMBG-1.4 transformer 5.x patch** (`ai_service.py:_load_rmbg_model`): `BriaRMBG` (loaded with `trust_remote_code`) is missing `all_tied_weights_keys` required by transformers 5.x. The loader temporarily monkey-patches `get_class_from_dynamic_module` to inject the missing property before restoring the original.

- **FireRed image dimensions**: Input images must be divisible by 64 for diffusers. The local inference path (`_process_local_model`) and Replicate path both resize and clamp dimensions accordingly.

- **FireRed GGUF vs standard**: If `FIRERED_MODEL_PATH` points to a `.gguf` file, the local loader uses `GGUFQuantizationConfig` + `QwenImageTransformer2DModel`; otherwise it loads the standard HuggingFace model.

- **Task metadata vs task status**: `task_metadata_store` in `routes.py` holds upload metadata (created_at, original URL, mode). Task status (PENDING/EDITING/COMPLETED/FAILED) is stored in Celery/Redis and retrieved via `AsyncResult`. The two are joined in `get_task_status` and `get_result`.

### Configuration

Copy `.env.example` to `.env`. Key variables:

```env
STORAGE_TYPE=local            # "local" or "gcs"
REDIS_URL=redis://localhost:6379/0
REPLICATE_API_TOKEN=          # Recommended for production
RMBG_API_URL=                 # Custom RMBG API (alternative to Replicate)
FIRERED_API_URL=              # Custom FireRed API (alternative to Replicate)
FIRERED_MODEL_PATH=           # Path to local .gguf file for local inference
AUTH_ENABLED=false
RATE_LIMIT_ENABLED=false
```

### CI/CD

GitHub Actions (`.github/workflows/ci.yml`): **Lint & type check → Unit tests → Docker build** on every push/PR to `main`.

Production deploy via Google Cloud Build → Cloud Run:
```bash
gcloud builds submit --config deploy/cloudbuild.yaml \
  --substitutions=_REGION=asia-east1,_SERVICE_NAME=ecommerce-visual-pro
```
