"""
SAQ worker — runs background tasks enqueued by the API.

Tasks:
  prepare_part(part_id)  — bg removal + embedding for a newly uploaded part
"""

import logging
import uuid

from app.core.config import settings
from app.db.models import Part
from app.db.session import async_session
from app.models.rembg_model import remove_background
from app.services.embeddings import embed_text
from app.workers.queue import get_queue

logger = logging.getLogger(__name__)


async def prepare_part(ctx: dict, *, part_id: str) -> None:
    logger.info("prepare_part started for part_id=%s", part_id)

    async with async_session() as db:
        part = await db.get(Part, uuid.UUID(part_id))
        if not part:
            logger.error("Part %s not found", part_id)
            return
        if not part.original_image_url:
            logger.error("Part %s has no original_image_url", part_id)
            return
        image_path = settings.url_to_path(part.original_image_url)

    try:
        # 1. Remove background
        image_bytes = image_path.read_bytes()
        processed_bytes = await remove_background(image_bytes)

        # 2. Save processed image
        filename = f"{uuid.uuid4()}.png"
        dest = settings.storage_subdir("processed") / filename
        dest.write_bytes(processed_bytes)
        processed_url = f"/storage/processed/{filename}"

        # 3. Compute embedding and mark ready
        async with async_session() as db:
            part = await db.get(Part, uuid.UUID(part_id))
            text = f"{part.name} {part.description or ''}".strip()
            part.embedding = await embed_text(text)
            part.processed_image_url = processed_url
            part.status = "ready"
            await db.commit()

        logger.info("prepare_part complete for part_id=%s", part_id)

    except Exception as exc:
        logger.error("prepare_part failed for %s: %s", part_id, exc)
        async with async_session() as db:
            part = await db.get(Part, uuid.UUID(part_id))
            if part:
                part.status = "failed"
                await db.commit()


async def main():
    import saq

    queue = await get_queue()
    worker = saq.Worker(
        queue,
        functions=[prepare_part],
        concurrency=2,
    )
    await worker.start()


if __name__ == "__main__":
    import asyncio

    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
