from saq import Queue  # type: ignore[import]

from app.core.config import settings

_queue: Queue | None = None


async def get_queue() -> Queue:
    global _queue
    if _queue is None:
        _queue = Queue.from_url(settings.REDIS_URL, name=settings.QUEUE_NAME)
    return _queue
