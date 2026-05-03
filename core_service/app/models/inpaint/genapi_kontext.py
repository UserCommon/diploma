"""Flux.Kontext inpainting via gen-api.ru

No mask needed — Kontext edits based on text instruction.
Supports multiple input images: car photo + reference part image.
"""

import io
import logging

import httpx
from PIL import Image

from app.core.config import settings

logger = logging.getLogger(__name__)

_API_URL = "https://api.gen-api.ru/api/v1/networks/flux-kontext"


def _describe_part(img: Image.Image) -> str:
    """Extract dominant color and basic visual description from a part image."""
    import numpy as np
    try:
        rgb = img.convert("RGB").resize((64, 64), Image.LANCZOS)
        arr = np.array(rgb).reshape(-1, 3)
        # Filter out near-white/transparent pixels (background)
        mask = ~((arr[:, 0] > 230) & (arr[:, 1] > 230) & (arr[:, 2] > 230))
        pixels = arr[mask]
        if len(pixels) == 0:
            pixels = arr
        avg = pixels.mean(axis=0).astype(int)
        r, g, b = avg
        # Map to color name
        if r > 180 and g > 180 and b > 180:
            color = "silver/chrome"
        elif r < 60 and g < 60 and b < 60:
            color = "black matte"
        elif r > 150 and g < 80 and b < 80:
            color = "red"
        elif r > 150 and g > 100 and b < 80:
            color = "gold/bronze"
        elif r < 80 and g < 80 and b > 150:
            color = "blue"
        else:
            color = f"color rgb({r},{g},{b})"
        return f"{color} finish"
    except Exception:
        return ""


def _to_png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


async def _call_api(
    image: Image.Image,
    prompt: str,
    api_key: str,
    model: str,
    part_image: Image.Image | None = None,
) -> Image.Image:
    img = image.copy().convert("RGB")
    img.thumbnail((768, 768), Image.LANCZOS)

    if part_image is not None:
        # Resize reference to same height as car image
        ref = part_image.convert("RGB")
        ref_h = img.height
        ref_w = int(ref.width * ref_h / ref.height)
        ref = ref.resize((ref_w, ref_h), Image.LANCZOS)

        # Concatenate side by side: car on left, reference on right
        combined = Image.new("RGB", (img.width + ref_w, img.height))
        combined.paste(img, (0, 0))
        combined.paste(ref, (img.width, 0))

        instruction = (
            f"The image shows two photos side by side. "
            f"Left: a car. Right: a reference part. "
            f"{prompt} on the car on the left, making it look like the part on the right. "
            f"Keep the car body, color, background, and everything else identical."
        )
        send_img = combined
    else:
        instruction = (
            f"This is a car photo. {prompt}. "
            f"Keep everything else exactly unchanged. Photorealistic result."
        )
        send_img = img

    headers = {
        "Accept": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    files: list[tuple] = [
        ("images[]", ("combined.png", _to_png_bytes(send_img), "image/png")),
    ]

    data = {
        "prompt": instruction,
        "model": model,
        "guidance_scale": "7",
        "num_images": "1",
        "output_format": "jpeg",
        "safety_tolerance": "6",
        "is_sync": "true",
    }

    async with httpx.AsyncClient(timeout=300) as client:
        resp = await client.post(_API_URL, data=data, files=files, headers=headers)
        if resp.status_code >= 400:
            logger.error("gen-api kontext error %s: %s", resp.status_code, resp.text[:500])
            resp.raise_for_status()
        result = resp.json()

    logger.debug("kontext response: %s", str(result)[:300])

    output_url = _extract_url(result)
    if not output_url:
        raise RuntimeError(f"No image URL in kontext response: {result}")

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


class GenApiKontextProvider:
    """gen-api.ru Flux.Kontext — multi-image instruction-based editing."""

    def __init__(self) -> None:
        if not settings.GENAPI_API_KEY:
            raise RuntimeError("GENAPI_API_KEY is not set")
        self._api_key = settings.GENAPI_API_KEY
        self._model = settings.GENAPI_KONTEXT_MODEL

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
            _call_api(image, prompt, self._api_key, self._model, part_image)
            for _ in range(n)
        ]
        return list(await asyncio.gather(*tasks))
