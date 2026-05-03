from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.api import parts
from app.core.config import settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings.storage_subdir("originals")
    settings.storage_subdir("processed")
    yield


app = FastAPI(title="Accessories Processing Service", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_list(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(parts.router, prefix="/api/parts", tags=["parts"])

app.mount(
    "/storage",
    StaticFiles(directory=settings.STORAGE_PATH),
    name="storage",
)


@app.get("/health")
async def health():
    return {"status": "ok"}
