import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi import status as http_status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.db.models import Part
from app.db.session import get_db
from app.schemas.part import (
    PartListResponse,
    PartRead,
    SearchRequest,
    SearchResponse,
    UploadPartResponse,
)
from app.services.embeddings import embed_text
from app.workers.queue import get_queue

router = APIRouter()


@router.post(
    "/upload",
    status_code=http_status.HTTP_202_ACCEPTED,
    response_model=UploadPartResponse,
)
async def upload_part(
    image: UploadFile = File(...),
    name: str = Form(...),
    category: str = Form(...),
    domain: str = Form(...),
    description: str | None = Form(None),
    db: AsyncSession = Depends(get_db),
):
    image_bytes = await image.read()
    ext = Path(image.filename).suffix or ".png"
    filename = f"{uuid.uuid4()}{ext}"
    dest = settings.storage_subdir("originals") / filename
    dest.write_bytes(image_bytes)

    part = Part(
        name=name,
        category=category,
        domain=domain,
        description=description,
        original_image_url=f"/storage/originals/{filename}",
        status="pending",
    )
    db.add(part)
    await db.commit()
    await db.refresh(part)

    queue = await get_queue()
    job = await queue.enqueue("prepare_part", part_id=str(part.id), timeout=120)

    return UploadPartResponse(task_id=job.id)


@router.get("/", response_model=PartListResponse)
async def list_parts(
    domain: str | None = None,
    category: str | None = None,
    db: AsyncSession = Depends(get_db),
):
    stmt = select(Part)
    if domain:
        stmt = stmt.where(Part.domain == domain)
    if category:
        stmt = stmt.where(Part.category == category)
    stmt = stmt.order_by(Part.created_at.desc())
    result = await db.execute(stmt)
    parts = result.scalars().all()
    return PartListResponse(items=[PartRead.model_validate(p) for p in parts])


@router.get("/{part_id}", response_model=PartRead)
async def get_part(part_id: uuid.UUID, db: AsyncSession = Depends(get_db)):
    part = await db.get(Part, part_id)
    if not part:
        raise HTTPException(status_code=404, detail="Part not found")
    return PartRead.model_validate(part)


@router.delete("/{part_id}", status_code=http_status.HTTP_204_NO_CONTENT)
async def delete_part(part_id: uuid.UUID, db: AsyncSession = Depends(get_db)):
    part = await db.get(Part, part_id)
    if not part:
        raise HTTPException(status_code=404, detail="Part not found")
    await db.delete(part)
    await db.commit()


@router.post("/search", response_model=SearchResponse)
async def search_parts(body: SearchRequest, db: AsyncSession = Depends(get_db)):
    query_vec = await embed_text(body.query)

    stmt = select(Part).where(Part.status == "ready")
    if body.domain:
        stmt = stmt.where(Part.domain == body.domain)
    result = await db.execute(stmt)
    parts = result.scalars().all()

    # cosine similarity in Python (keeps DB portable; switch to pgvector <=> for scale)
    import numpy as np

    q = np.array(query_vec)

    def cosine_sim(part: Part) -> float:
        if part.embedding is None:
            return -1.0
        v = np.array(part.embedding)
        denom = np.linalg.norm(q) * np.linalg.norm(v)
        return float(np.dot(q, v) / denom) if denom else 0.0

    min_score = 0.4
    scored = sorted(((cosine_sim(p), p) for p in parts), reverse=True)
    items = []
    for score, p in scored[: body.limit]:
        if score < min_score:
            break
        part_data = PartRead.model_validate(p)
        part_data.score = round(score, 3)
        items.append(part_data)
    return SearchResponse(items=items)
