import asyncio
from functools import lru_cache

from sentence_transformers import SentenceTransformer

_MODEL_NAME = "all-MiniLM-L6-v2"


@lru_cache(maxsize=1)
def _get_model() -> SentenceTransformer:
    return SentenceTransformer(_MODEL_NAME)


async def embed_text(text: str) -> list[float]:
    loop = asyncio.get_running_loop()
    model = _get_model()
    vector = await loop.run_in_executor(None, lambda: model.encode(text))
    return vector.tolist()
