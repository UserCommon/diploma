from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    DATABASE_URL: str = (
        "postgresql+asyncpg://user:password@localhost:5432/accessories_db"
    )
    REDIS_URL: str = "redis://localhost:6379"
    QUEUE_NAME: str = "accessories"
    STORAGE_PATH: str = "/app/storage"
    CORS_ORIGINS: str = "http://localhost:5173"

    model_config = {"env_file": ".env", "extra": "ignore"}

    def cors_list(self) -> list[str]:
        return [o.strip() for o in self.CORS_ORIGINS.split(",")]

    def storage_subdir(self, sub: str) -> Path:
        path = Path(self.STORAGE_PATH) / sub
        path.mkdir(parents=True, exist_ok=True)
        return path

    def url_to_path(self, url: str) -> Path:
        prefix = "/storage/"
        if url.startswith(prefix):
            url = url[len(prefix) :]
        return Path(self.STORAGE_PATH) / url


settings = Settings()
