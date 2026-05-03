import uuid
from datetime import datetime

from pydantic import BaseModel


class PartRead(BaseModel):
    id: uuid.UUID
    name: str
    category: str
    domain: str
    original_image_url: str | None
    processed_image_url: str | None
    description: str | None
    status: str
    created_at: datetime
    score: float | None = None

    model_config = {"from_attributes": True}


class PartListResponse(BaseModel):
    items: list[PartRead]


class UploadPartResponse(BaseModel):
    task_id: str


class SearchRequest(BaseModel):
    query: str
    domain: str | None = None
    limit: int = 5


class SearchResponse(BaseModel):
    items: list[PartRead]
