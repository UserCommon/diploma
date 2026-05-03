"""SDXL Inpainting via diffusers — runs locally on CPU/MPS/CUDA.

Model: diffusers/stable-diffusion-xl-1.0-inpainting-0.1 (~7GB, free, no token needed)
"""

import asyncio
import logging
from typing import Any

from PIL import Image

logger = logging.getLogger(__name__)

_pipe: Any = None
_SDXL_MODEL = "diffusers/stable-diffusion-xl-1.0-inpainting-0.1"


def _load_pipeline() -> Any:
    global _pipe
    if _pipe is not None:
        return _pipe

    import torch

    # transformers==4.36.2 (needed for GroundingDINO) lacks several Siglip symbols
    # that diffusers 0.32 imports for IP-Adapter (which we don't use).
    # Inject stubs so the import succeeds.
    import transformers as _transformers
    for _name in ("SiglipImageProcessor", "SiglipVisionModel", "SiglipVisionConfig"):
        if not hasattr(_transformers, _name):
            setattr(_transformers, _name, type(_name, (), {}))

    from diffusers import StableDiffusionXLInpaintPipeline  # type: ignore[import]

    if torch.cuda.is_available():
        dtype = torch.float16
        device = "cuda"
    elif torch.backends.mps.is_available():
        dtype = torch.float16
        device = "mps"
    else:
        dtype = torch.float32
        device = "cpu"

    logger.info("Loading SDXL Inpainting pipeline (%s, device=%s)…", _SDXL_MODEL, device)

    _pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
        _SDXL_MODEL,
        torch_dtype=dtype,
    ).to(device)

    logger.info("SDXL Inpainting pipeline ready.")
    return _pipe


def _generate_sync(
    image: Image.Image,
    mask: Image.Image,
    prompt: str,
    steps: int,
    guidance: float,
    seed: int,
) -> Image.Image:
    import torch
    pipe = _load_pipeline()

    # SDXL works best at 1024x1024, resize while keeping aspect ratio
    w, h = image.size
    # Scale so that the larger side = 1024, then round to nearest 8
    scale = 1024 / max(w, h)
    w = max(8, (round(w * scale) // 8) * 8)
    h = max(8, (round(h * scale) // 8) * 8)

    image = image.resize((w, h), Image.LANCZOS).convert("RGB")
    mask = mask.resize((w, h), Image.LANCZOS).convert("L")

    # Dilate mask slightly so SDXL has context around the edges
    import numpy as np
    from PIL import ImageFilter
    mask = mask.filter(ImageFilter.MaxFilter(15))

    generator = torch.Generator().manual_seed(seed)
    result = pipe(
        prompt=prompt,
        image=image,
        mask_image=mask,
        height=h,
        width=w,
        num_inference_steps=steps,
        guidance_scale=guidance,
        strength=0.99,
        padding_mask_crop=32,  # crop around mask for better small-area inpainting
        generator=generator,
    )
    return result.images[0]


class DiffusersSDXLInpaintProvider:
    """Local SDXL Inpainting via diffusers (~7GB, no token required)."""

    async def generate(
        self,
        image: Image.Image,
        mask: Image.Image,
        prompt: str,
        n: int = 1,
        part_image: Image.Image | None = None,
        extra_images: list[Image.Image] | None = None,
    ) -> list[Image.Image]:
        loop = asyncio.get_running_loop()
        tasks = [
            loop.run_in_executor(
                None,
                _generate_sync,
                image,
                mask,
                prompt,
                30,    # steps
                7.5,   # guidance (SDXL sweet spot)
                1000 + i * 42,
            )
            for i in range(n)
        ]
        return list(await asyncio.gather(*tasks))
