"""Flux 2 Klein via gen-api.ru — fast img2img with reference images.

No mask support — compositing applied post-generation using SAM2 mask.
Endpoint: https://api.gen-api.ru/api/v1/networks/flux-2-klein
"""

import io
import logging

import httpx
from PIL import Image, ImageFilter

from app.core.config import settings

logger = logging.getLogger(__name__)

_API_URL = "https://api.gen-api.ru/api/v1/networks/flux-2-klein"

_NEGATIVE_PROMPT = (
    "extra spokes, wrong number of spokes, deformed wheels, distorted rims, "
    "blurry, low quality, artifacts, unrealistic"
)


def _to_png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _composite_with_mask(
    result: Image.Image,
    original: Image.Image,
    mask: Image.Image,
) -> Image.Image:
    result_matched = result.resize(original.size, Image.LANCZOS)
    mask_full = mask.convert("L").resize(original.size, Image.LANCZOS)
    mask_feathered = mask_full.filter(ImageFilter.GaussianBlur(6))
    return Image.composite(result_matched, original, mask_feathered)


async def _call_api(
    image: Image.Image,
    mask: Image.Image,
    prompt: str,
    api_key: str,
    model: str,
    part_image: Image.Image | None,
    seed: int,
) -> Image.Image:
    img = image.copy()
    img.thumbnail((1024, 1024), Image.LANCZOS)
    # Klein requires dimensions divisible by 16
    w, h = img.size
    w = max(16, (w // 16) * 16)
    h = max(16, (h // 16) * 16)
    img = img.resize((w, h), Image.LANCZOS)

    headers = {
        "Accept": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    files: list[tuple] = [
        ("image_urls[]", ("car.png", _to_png_bytes(img), "image/png")),
    ]
    if part_image is not None:
        ref = part_image.convert("RGB").copy()
        ref.thumbnail((512, 512), Image.LANCZOS)
        files.append(("image_urls[]", ("reference.png", _to_png_bytes(ref), "image/png")))

    data = {
        "prompt": prompt,
        "model": model,
        "width": str(w),
        "height": str(h),
        "negative_prompt": _NEGATIVE_PROMPT,
        "guidance_scale": "7",
        "num_inference_steps": "5",
        "num_images": "1",
        "enable_safety_checker": "false",
        "output_format": "png",
        "seed": str(seed),
        "is_sync": "true",
    }

    async with httpx.AsyncClient(timeout=300) as client:
        resp = await client.post(_API_URL, data=data, files=files, headers=headers)
        if resp.status_code >= 400:
            logger.error("flux-2-klein error %s: %s", resp.status_code, resp.text[:500])
            resp.raise_for_status()
        result = resp.json()

    logger.debug("klein response: %s", str(result)[:300])

    output_url = _extract_url(result)
    if not output_url:
        raise RuntimeError(f"No image URL in klein response: {result}")

    async with httpx.AsyncClient(timeout=60) as client:
        img_resp = await client.get(output_url)
        img_resp.raise_for_status()

    return Image.open(io.BytesIO(img_resp.content)).convert("RGB")


def _extract_url(data: dict) -> str | None:
    for key in ("output", "image_url", "url", "result"):
        val = data.get(key)
        if isinstance(val, str) and val.startswith("http"):
            return val
        if isinstance(val, list) and val and isinstance(val[0], str):
            return val[0]
    for key in ("data", "images", "output_images"):
        items = data.get(key)
        if isinstance(items, list) and items:
            first = items[0]
            if isinstance(first, str):
                return first
            if isinstance(first, dict):
                return first.get("url") or first.get("image_url")
    return None


class GenApiKleinProvider:
    """Flux 2 Klein — fast img2img + mask compositing."""

    def __init__(self) -> None:
        if not settings.GENAPI_API_KEY:
            raise RuntimeError("GENAPI_API_KEY is not set")
        self._api_key = settings.GENAPI_API_KEY
        self._model = settings.GENAPI_KLEIN_MODEL

    async def generate(
        self,
        image: Image.Image,
        mask: Image.Image,
        prompt: str,
        n: int = 1,
        part_image: Image.Image | None = None,
        extra_images: list[Image.Image] | None = None,
    ) -> list[Image.Image]:
        import asyncio
        tasks = [
            _call_api(image, mask, prompt, self._api_key, self._model, part_image, 1000 + i * 42)
            for i in range(n)
        ]
        return list(await asyncio.gather(*tasks))
