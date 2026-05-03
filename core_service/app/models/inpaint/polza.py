import asyncio
import base64
import io
import logging

import httpx
from PIL import Image

from app.core.config import settings

logger = logging.getLogger(__name__)

_ALLOWED_RATIOS: dict[str, float] = {
    "1:1": 1 / 1,
    "4:3": 4 / 3,
    "3:4": 3 / 4,
    "16:9": 16 / 9,
    "9:16": 9 / 16,
    "3:2": 3 / 2,
    "2:3": 2 / 3,
}

_API_URL = "https://polza.ai/api/v1/media"


def _nearest_aspect(size: tuple[int, int]) -> str:
    w, h = size
    ratio = w / h
    return min(_ALLOWED_RATIOS, key=lambda k: abs(_ALLOWED_RATIOS[k] - ratio))


def _img_to_b64(img: Image.Image, fmt: str = "PNG") -> str:
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode()


def _extract_url(data: dict[str, object]) -> str | None:
    items = data.get("data")
    if isinstance(items, list) and items:
        first = items[0]
        if isinstance(first, dict):
            url = first.get("url")
            if isinstance(url, str):
                return url
    for key in ("output", "image_url", "url", "result"):
        val = data.get(key)
        if isinstance(val, str) and val.startswith("http"):
            return val
        if isinstance(val, list) and val and isinstance(val[0], str):
            return val[0]
    return None


async def _poll(
    job_id: str,
    api_key: str,
    interval: float = 2.0,
    max_wait: float = 180.0,
) -> str:
    headers = {"Authorization": f"Bearer {api_key}"}
    elapsed = 0.0
    async with httpx.AsyncClient(timeout=30) as client:
        while elapsed < max_wait:
            await asyncio.sleep(interval)
            elapsed += interval
            resp = await client.get(f"{_API_URL}/{job_id}", headers=headers)
            resp.raise_for_status()
            data: dict[str, object] = resp.json()
            status = data.get("status", "")
            if status in ("succeeded", "completed", "done", "success"):
                url = _extract_url(data)
                if url:
                    return url
            elif status in ("failed", "error", "cancelled"):
                raise RuntimeError(f"Polza job {job_id} failed: {data}")
            logger.debug("Polza job %s status: %s", job_id, status)
    raise RuntimeError(f"Polza job {job_id} timed out after {max_wait}s")


async def _call_api(
    image: Image.Image,
    mask: Image.Image,
    prompt: str,
    seed: int,
    part_image: Image.Image | None,
    api_key: str,
    model: str,
    extra_images: list[Image.Image] | None = None,
) -> Image.Image:
    img_resized = image.copy()
    img_resized.thumbnail((1024, 1024))
    mask_resized = mask.convert("L").copy()
    mask_resized.thumbnail((1024, 1024))

    # Order: [0] mask, [1] reference (if any), [2] car, [3+] extra car photos
    images_payload: list[dict[str, str]] = [
        {"type": "base64", "data": _img_to_b64(mask_resized)},
    ]

    if part_image is not None:
        ref = part_image.convert("RGB").copy()
        ref.thumbnail((1024, 1024), Image.LANCZOS)
        images_payload.append({"type": "base64", "data": _img_to_b64(ref)})

    images_payload.append({"type": "base64", "data": _img_to_b64(img_resized)})

    for extra in (extra_images or []):
        ex = extra.convert("RGB").copy()
        ex.thumbnail((1024, 1024), Image.LANCZOS)
        images_payload.append({"type": "base64", "data": _img_to_b64(ex)})

    if part_image is not None:
        n_extra = len(extra_images or [])
        n_car = 1 + n_extra
        car_indices = ", ".join(str(i + 3) for i in range(n_car))
        final_prompt = (
            f"{prompt}. "
            "Use image 1 as the inpainting mask. "
            "Use image 2 as the reference part — replicate its exact visual appearance, color, and finish. "
            f"Edit car photo{'s' if n_car > 1 else ''} in image{'s' if n_car > 1 else ''} {car_indices}."
        )
    else:
        final_prompt = (
            f"{prompt}. "
            "Use image 1 as the inpainting mask. "
            "Edit car photo in image 2."
        )

    payload: dict[str, object] = {
        "model": model,
        "input": {
            "prompt": final_prompt,
            "aspect_ratio": _nearest_aspect(img_resized.size),
            "image_resolution": "2K",
            "strength": 0.8,
            "seed": seed,
            "images": images_payload,
        },
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    async with httpx.AsyncClient(timeout=360) as client:
        resp = await client.post(_API_URL, json=payload, headers=headers)
        if resp.status_code >= 400:
            logger.error("Polza API error %s: %s", resp.status_code, resp.text)
            resp.raise_for_status()
        data: dict[str, object] = resp.json()

    output_url = _extract_url(data)
    if output_url is None:
        job_id = data.get("id")
        if not isinstance(job_id, str):
            raise RuntimeError(f"Unexpected polza.ai response: {data}")
        output_url = await _poll(job_id, api_key)

    async with httpx.AsyncClient(timeout=httpx.Timeout(10, read=120)) as client:
        img_resp = await client.get(output_url)
        img_resp.raise_for_status()

    return Image.open(io.BytesIO(img_resp.content)).convert("RGB")


class PolzaInpaintProvider:
    """polza.ai inpainting — model is configurable via POLZA_MODEL env var."""

    def __init__(self) -> None:
        if not settings.POLZA_AI_API_KEY:
            raise RuntimeError("POLZA_AI_API_KEY is not set")
        self._api_key = settings.POLZA_AI_API_KEY
        self._model = settings.POLZA_MODEL

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
            _call_api(
                image, mask, prompt,
                seed=1000 + i * 42,
                part_image=part_image,
                api_key=self._api_key,
                model=self._model,
                extra_images=extra_images,
            )
            for i in range(n)
        ]
        return list(await asyncio.gather(*tasks))
