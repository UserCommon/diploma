import asyncio
from functools import lru_cache

import rembg


@lru_cache(maxsize=1)
def _get_session():
    return rembg.new_session("u2net")


async def remove_background(image_bytes: bytes) -> bytes:
    loop = asyncio.get_running_loop()
    session = _get_session()
    result = await loop.run_in_executor(
        None, lambda: rembg.remove(image_bytes, session=session)
    )
    return result
