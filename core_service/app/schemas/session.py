import uuid
from datetime import datetime
from typing import Any

from pydantic import BaseModel


class SessionRead(BaseModel):
    id: uuid.UUID
    status: str
    system_prompt: str | None
    user_prompt: str
    car_image_urls: list[str]
    events: list[dict[str, Any]]
    result_image_urls: list[str]
    error_message: str | None
    created_at: datetime

    model_config = {"from_attributes": True}


class CreateSessionResponse(BaseModel):
    session_id: str
