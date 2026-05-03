from __future__ import annotations

import json
import logging
import uuid
from collections.abc import AsyncGenerator
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.db.models import AgentSession
from app.db.session import get_db
from app.schemas.session import CreateSessionResponse, SessionRead
from app.workers.queue import get_queue

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/sessions", status_code=202, response_model=CreateSessionResponse)
async def create_session(
    user_prompt: str = Form(...),
    system_prompt: str = Form(""),
    n: int = Form(2),
    inpaint_provider: str = Form(""),
    images: list[UploadFile] = File(default=[]),
    db: AsyncSession = Depends(get_db),
) -> CreateSessionResponse:
    originals_dir = settings.storage_subdir("originals")
    car_image_urls: list[str] = []

    for img_file in images:
        data = await img_file.read()
        import io
        from PIL import Image as PILImage
        img = PILImage.open(io.BytesIO(data)).convert("RGB")
        filename = f"{uuid.uuid4()}_car.jpg"
        img.save(originals_dir / filename, "JPEG", quality=95)
        car_image_urls.append(f"/storage/originals/{filename}")

    session = AgentSession(
        status="pending",
        system_prompt=system_prompt or None,
        user_prompt=user_prompt,
        car_image_urls=car_image_urls or None,
        events=[],
    )
    db.add(session)
    await db.commit()
    await db.refresh(session)

    queue = await get_queue()
    await queue.enqueue("run_agent_session", session_id=str(session.id), n=max(1, min(n, 8)), inpaint_provider=inpaint_provider, timeout=3600)

    return CreateSessionResponse(session_id=str(session.id))


@router.get("/sessions/{session_id}", response_model=SessionRead)
async def get_session(
    session_id: str, db: AsyncSession = Depends(get_db)
) -> SessionRead:
    s = await db.get(AgentSession, uuid.UUID(session_id))
    if s is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return SessionRead(
        id=s.id,
        status=s.status,
        system_prompt=s.system_prompt,
        user_prompt=s.user_prompt,
        car_image_urls=s.car_image_urls or [],
        events=s.events or [],
        result_image_urls=s.result_image_urls or [],
        error_message=s.error_message,
        created_at=s.created_at,
    )


@router.get("/sessions/{session_id}/stream")
async def stream_session(session_id: str) -> StreamingResponse:
    """SSE stream — polls DB until session reaches terminal state."""

    async def generator() -> AsyncGenerator[str, None]:
        import asyncio

        from app.db.session import async_session

        seen = 0
        while True:
            async with async_session() as db:
                s = await db.get(AgentSession, uuid.UUID(session_id))
                if s is None:
                    yield f"data: {json.dumps({'type': 'error', 'message': 'session not found'})}\n\n"
                    return

                for event in (s.events or [])[seen:]:
                    yield f"data: {json.dumps(event)}\n\n"
                seen = len(s.events or [])

                if s.status in ("complete", "failed"):
                    yield f"data: {json.dumps({'type': 'done', 'status': s.status})}\n\n"
                    return

            await asyncio.sleep(0.5)

    return StreamingResponse(
        generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
