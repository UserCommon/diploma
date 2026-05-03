"""GPT Image inpainting via Polza AI /media endpoint.

GPT Image mask format: RGBA PNG where alpha=0 (transparent) = region to edit.
Our internal mask format: L-mode grayscale, white=edit.
This provider converts between the two.
"""

import asyncio
import base64
import io
import logging

import httpx
from PIL import Image

from app.core.config import settings
from app.models.inpaint.polza import _extract_url, _nearest_aspect, _poll

logger = logging.getLogger(__name__)

_API_URL = "https://polza.ai/api/v1/media"


def _img_to_b64(img: Image.Image, fmt: str = "PNG") -> str:
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode()


def _mask_to_rgba(mask: Image.Image, size: tuple[int, int]) -> Image.Image:
    """Convert L-mode mask (white=edit) → RGBA (transparent=edit)."""
    m = mask.convert("L").resize(size, Image.LANCZOS)
    rgba = Image.new("RGBA", size, (0, 0, 0, 255))  # fully opaque = keep
    # Where mask is white (255) → make transparent (alpha=0) = edit here
    import numpy as np
    arr = np.array(m)
    rgba_arr = np.array(rgba)
    rgba_arr[:, :, 3] = 255 - arr  # invert: white→0 (transparent), black→255 (opaque)
    return Image.fromarray(rgba_arr, "RGBA")


async def _call_api(
    image: Image.Image,
    mask: Image.Image,
    prompt: str,
    part_image: Image.Image | None,
    api_key: str,
    model: str,
) -> Image.Image:
    img_resized = image.copy()
    img_resized.thumbnail((1024, 1024))
    rgba_mask = _mask_to_rgba(mask, img_resized.size)

    images_payload: list[dict[str, str]] = [
        {"type": "base64", "data": _img_to_b64(img_resized)},
        {"type": "base64", "data": _img_to_b64(rgba_mask)},
    ]
    if part_image is not None:
        ref = part_image.convert("RGB").resize(img_resized.size, Image.LANCZOS)
        images_payload.append({"type": "base64", "data": _img_to_b64(ref)})

    payload: dict[str, object] = {
        "model": model,
        "input": {
            "prompt": prompt,
            "aspect_ratio": _nearest_aspect(img_resized.size),
            "images": images_payload,
        },
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    async with httpx.AsyncClient(timeout=120) as client:
        resp = await client.post(_API_URL, json=payload, headers=headers)
        if resp.status_code >= 400:
            logger.error("GPT Image API error %s: %s", resp.status_code, resp.text)
            resp.raise_for_status()
        data: dict[str, object] = resp.json()

    output_url = _extract_url(data)
    if output_url is None:
        job_id = data.get("id")
        if not isinstance(job_id, str):
            raise RuntimeError(f"Unexpected gpt-image response: {data}")
        output_url = await _poll(job_id, api_key)

    async with httpx.AsyncClient(timeout=httpx.Timeout(10, read=120)) as client:
        img_resp = await client.get(output_url)
        img_resp.raise_for_status()

    return Image.open(io.BytesIO(img_resp.content)).convert("RGB")


class GptImageInpaintProvider:
    """OpenAI GPT Image inpainting via Polza AI."""

    def __init__(self) -> None:
        if not settings.POLZA_AI_API_KEY:
            raise RuntimeError("POLZA_AI_API_KEY is not set")
        self._api_key = settings.POLZA_AI_API_KEY
        self._model = settings.GPT_IMAGE_MODEL

    async def generate(
        self,
        image: Image.Image,
        mask: Image.Image,
        prompt: str,
        n: int = 1,
        part_image: Image.Image | None = None,
        extra_images: list[Image.Image] | None = None,
    ) -> list[Image.Image]:
        tasks = [
            _call_api(image, mask, prompt, part_image, self._api_key, self._model)
            for _ in range(n)
        ]
        return list(await asyncio.gather(*tasks))
