"""FLUX Inpainting via gen-api.ru

Endpoint: https://api.gen-api.ru/api/v1/networks/flux
Supports mask-based inpainting via image + mask parameters.
"""

import asyncio
import base64
import io
import logging

import httpx
from PIL import Image

from app.core.config import settings

logger = logging.getLogger(__name__)

_API_URL = "https://api.gen-api.ru/api/v1/networks/flux"


def _img_to_b64_url(img: Image.Image, fmt: str = "PNG") -> str:
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    b64 = base64.b64encode(buf.getvalue()).decode()
    mime = "image/png" if fmt == "PNG" else "image/jpeg"
    return f"data:{mime};base64,{b64}"


def _to_png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _fit_size(w: int, h: int) -> tuple[int, int]:
    """Resize so both dims are divisible by 8 and >= 512, max 1024."""
    scale = min(1024 / max(w, h), 1.0)
    w = max(512, (round(w * scale) // 8) * 8)
    h = max(512, (round(h * scale) // 8) * 8)
    return w, h


async def _call_api(
    image: Image.Image,
    mask: Image.Image,
    prompt: str,
    api_key: str,
    model: str,
    n_steps: int,
    part_image: Image.Image | None = None,
) -> Image.Image:
    w, h = _fit_size(*image.size)
    img_resized = image.resize((w, h), Image.LANCZOS).convert("RGB")
    mask_resized = mask.convert("L").resize((w, h), Image.LANCZOS)

    headers = {
        "Accept": "application/json",
        "Authorization": f"Bearer {api_key}",
    }


    # If a reference part image is provided, paste it into the masked region
    # so the inpainting model uses it as visual context.
    if part_image is not None:
        bbox = mask_resized.getbbox()
        if bbox:
            bw = bbox[2] - bbox[0]
            bh = bbox[3] - bbox[1]
            if bw > 0 and bh > 0:
                ref = part_image.convert("RGBA")
                ref.thumbnail((bw, bh), Image.LANCZOS)
                # Center inside bbox
                px = bbox[0] + (bw - ref.width) // 2
                py = bbox[1] + (bh - ref.height) // 2
                canvas = img_resized.copy()
                # Use alpha channel as mask so transparent background isn't pasted
                alpha = ref.split()[3] if ref.mode == "RGBA" else None
                canvas.paste(ref.convert("RGB"), (px, py), mask=alpha)
                img_resized = canvas

    files = {
        "image": ("image.png", _to_png_bytes(img_resized), "image/png"),
        "mask": ("mask.png", _to_png_bytes(mask_resized), "image/png"),
    }
    data = {
        "prompt": prompt,
        "model": model,
        "width": str(w),
        "height": str(h),
        "num_inference_steps": str(n_steps),
        "guidance_scale": "7",
        "num_images": "1",
        "enable_safety_checker": "false",
        "strength": "0.99",
        "is_sync": "true",
    }

    async with httpx.AsyncClient(timeout=300) as client:
        resp = await client.post(_API_URL, data=data, files=files, headers=headers)
        if resp.status_code >= 400:
            logger.error("gen-api error %s: %s", resp.status_code, resp.text[:500])
            resp.raise_for_status()
        data_resp = resp.json()

    logger.debug("gen-api response: %s", str(data_resp)[:300])

    # Extract result image URL
    output_url = _extract_url(data_resp)
    if not output_url:
        raise RuntimeError(f"No image URL in gen-api response: {data_resp}")

    async with httpx.AsyncClient(timeout=60) as client:
        img_resp = await client.get(output_url)
        img_resp.raise_for_status()

    return Image.open(io.BytesIO(img_resp.content)).convert("RGB")


def _extract_url(data: dict) -> str | None:
    # Try common response shapes
    for key in ("output", "image_url", "url", "result"):
        val = data.get(key)
        if isinstance(val, str) and val.startswith("http"):
            return val
        if isinstance(val, list) and val and isinstance(val[0], str):
            return val[0]
    items = data.get("data") or data.get("images") or data.get("output_images")
    if isinstance(items, list) and items:
        first = items[0]
        if isinstance(first, str):
            return first
        if isinstance(first, dict):
            return first.get("url") or first.get("image_url")
    return None


class GenApiFluxInpaintProvider:
    """gen-api.ru FLUX inpainting — true mask-based inpainting."""

    def __init__(self) -> None:
        if not settings.GENAPI_API_KEY:
            raise RuntimeError("GENAPI_API_KEY is not set")
        self._api_key = settings.GENAPI_API_KEY
        self._model = settings.GENAPI_MODEL
        self._steps = settings.GENAPI_STEPS

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
            _call_api(image, mask, prompt, self._api_key, self._model, self._steps, part_image)
            for _ in range(n)
        ]
        return list(await asyncio.gather(*tasks))
