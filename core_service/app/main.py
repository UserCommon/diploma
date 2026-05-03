from contextlib import asynccontextmanager
from collections.abc import AsyncGenerator
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.api import sessions
from app.core.config import settings


# Create storage dirs at import time so StaticFiles mount succeeds
Path(settings.STORAGE_PATH).mkdir(parents=True, exist_ok=True)
settings.storage_subdir("originals")
settings.storage_subdir("masks")
settings.storage_subdir("results")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    yield


app = FastAPI(title="Core Service", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_list(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(sessions.router, prefix="/api/agent", tags=["agent"])

app.mount(
    "/storage",
    StaticFiles(directory=settings.STORAGE_PATH),
    name="storage",
)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}
