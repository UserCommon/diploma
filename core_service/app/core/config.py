from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    DATABASE_URL: str = "postgresql+asyncpg://user:password@localhost:5432/core_db"
    REDIS_URL: str = "redis://localhost:6379"
    QUEUE_NAME: str = "core"
    STORAGE_PATH: str = "./storage"
    CORS_ORIGINS: str = "http://localhost:5173"

    ACCESSORIES_SERVICE_URL: str = "http://localhost:8000"

    INPAINT_PROVIDER: str = "stub"
    POLZA_AI_API_KEY: str = ""
    POLZA_MODEL: str = "black-forest-labs/flux.1-schnell"
    GPT_IMAGE_MODEL: str = "openai/gpt-image-1.5"

    # gen-api.ru FLUX inpainting
    GENAPI_API_KEY: str = ""
    GENAPI_MODEL: str = "inpainting"
    GENAPI_STEPS: int = 40
    GENAPI_KONTEXT_MODEL: str = "max"
    GENAPI_KLEIN_MODEL: str = "4B Standard"

    # Local diffusers inpainting (FLUX.1 Fill Dev)
    DIFFUSERS_MODEL: str = "black-forest-labs/FLUX.1-Fill-dev"
    DIFFUSERS_STEPS: int = 30
    DIFFUSERS_GUIDANCE: float = 30.0
    HF_TOKEN: str = ""

    # LLM — set LLM_PROVIDER to enable the real agent, leave empty for stub
    # Supported: openai | anthropic | openai_compatible
    # openai_compatible works with any OpenAI-compatible API (polza, groq, etc.)
    LLM_PROVIDER: str = ""
    LLM_API_KEY: str = ""
    LLM_MODEL: str = "gpt-4o-mini"
    LLM_BASE_URL: str = ""  # only needed for openai_compatible

    SAM2_CHECKPOINT: str = "./models/sam2_hiera_base_plus.pt"
    SAM2_CONFIG: str = "sam2_hiera_b+"
    GROUNDING_DINO_CONFIG: str = "./models/GroundingDINO_SwinT_OGC.py"
    GROUNDING_DINO_CHECKPOINT: str = "./models/groundingdino_swint_ogc.pth"

    model_config = {"env_file": ".env", "extra": "ignore"}

    def cors_list(self) -> list[str]:
        return [o.strip() for o in self.CORS_ORIGINS.split(",")]

    def storage_subdir(self, sub: str) -> Path:
        path = Path(self.STORAGE_PATH) / sub
        path.mkdir(parents=True, exist_ok=True)
        return path

    def url_to_path(self, url: str) -> Path:
        prefix = "/storage/"
        relative = url[len(prefix):] if url.startswith(prefix) else url.lstrip("/")
        return Path(self.STORAGE_PATH) / relative


settings = Settings()
