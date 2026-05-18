"""LangChain tools for the ReAct agent."""

from __future__ import annotations

import json
import logging
import uuid

import io

import httpx
from langchain_core.tools import tool  # type: ignore[import]
from PIL import Image

from app.agent.context import n_variants
from app.core.config import settings
from app.models.inpaint import get_inpaint_provider
from app.models.segmentation import segment_with_text

logger = logging.getLogger(__name__)


@tool  # type: ignore[misc]
async def search_parts(query: str, domain: str = "car") -> str:
    """Search the accessories library for parts matching a natural language query.

    Args:
        query: Description of the part to find (e.g. "carbon fibre rear spoiler")
        domain: Filter by domain — car, moto, or interior. Default: car

    Returns:
        JSON list of matching parts with id, name, category, processed_image_url
    """
    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.post(
            f"{settings.ACCESSORIES_SERVICE_URL}/api/parts/search",
            json={"query": query, "domain": domain, "limit": 5},
        )
        resp.raise_for_status()
        data = resp.json()

    return json.dumps(data.get("items", []))


@tool  # type: ignore[misc]
async def segment_object(image_url: str, text_prompt: str) -> str:
    """Segment an object in a car image using SAM2 + GroundingDINO.

    Args:
        image_url: URL of the car image (e.g. /storage/originals/xxx.jpg)
        text_prompt: Object to segment (e.g. "rear bumper")

    Returns:
        URL of the generated greyscale mask image
    """
    file_path = settings.url_to_path(image_url)
    image = Image.open(file_path).convert("RGB")

    mask = await segment_with_text(image, text_prompt)

    filename = f"{uuid.uuid4()}_mask.png"
    mask_path = settings.storage_subdir("masks") / filename
    mask.save(mask_path)

    return f"/storage/masks/{filename}"


@tool  # type: ignore[misc]
async def generate_image(
    source_image_url: str,
    mask_url: str,
    prompt: str,
    part_image_url: str = "",
    extra_source_image_urls: str = "",
) -> str:
    """Inpaint a car image using the configured provider.

    Args:
        source_image_url: URL of the primary car source image
        mask_url: URL of the segmentation mask (white = region to edit)
        prompt: Text prompt for the desired result
        part_image_url: Optional URL of a reference part image
        extra_source_image_urls: Optional comma-separated URLs of additional car photos

    Returns:
        JSON list of result image URLs
    """
    source = Image.open(settings.url_to_path(source_image_url)).convert("RGB")
    mask = Image.open(settings.url_to_path(mask_url)).convert("L")

    part_img: Image.Image | None = None
    if part_image_url:
        try:
            full_url = (
                f"{settings.ACCESSORIES_SERVICE_URL}{part_image_url}"
                if part_image_url.startswith("/")
                else part_image_url
            )
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.get(full_url)
                resp.raise_for_status()
            part_img = Image.open(io.BytesIO(resp.content)).convert("RGBA")
        except Exception:
            logger.warning("Could not load part_image_url: %s", part_image_url)

    extra_imgs: list[Image.Image] = []
    if extra_source_image_urls:
        for url in extra_source_image_urls.split(","):
            url = url.strip()
            if url:
                try:
                    extra_imgs.append(Image.open(settings.url_to_path(url)).convert("RGB"))
                except Exception:
                    logger.warning("Could not load extra image: %s", url)

    provider = get_inpaint_provider()
    results = await provider.generate(
        source, mask, prompt,
        n=n_variants.get(),
        part_image=part_img,
        extra_images=extra_imgs or None,
    )

    result_urls: list[str] = []
    results_dir = settings.storage_subdir("results")
    for i, img in enumerate(results):
        filename = f"{uuid.uuid4()}_{i}.jpg"
        img.save(results_dir / filename, "JPEG", quality=90)
        result_urls.append(f"/storage/results/{filename}")

    return json.dumps(result_urls)
