"""FLUX.1 Fill Dev inpainting via diffusers — runs locally on CPU/MPS/CUDA.

First run downloads ~20GB of weights from HuggingFace.
Requires HF_TOKEN env var if the model is gated.
"""

import asyncio
import logging
from typing import Any

from PIL import Image

from app.core.config import settings

logger = logging.getLogger(__name__)

_pipe: Any = None


def _load_pipeline() -> Any:
    global _pipe
    if _pipe is not None:
        return _pipe

    import torch

    # transformers==4.36.2 requires tokenizers<0.19 but FLUX needs tokenizers>=0.19.
    # Patch the version check so both can coexist (GroundingDINO only uses BertTokenizer
    # which works fine with newer tokenizers despite the strict version gate).
    try:
        from transformers.utils import versions as _tv
        _orig_check = _tv.require_version
        def _permissive_check(requirement: str, hint: str = "") -> None:
            if "tokenizers" in requirement:
                return
            _orig_check(requirement, hint)
        _tv.require_version = _permissive_check
        # Also patch where it's already imported
        import transformers.tokenization_utils_fast as _tuf
        if hasattr(_tuf, "require_version"):
            _tuf.require_version = _permissive_check  # type: ignore[attr-defined]
    except Exception:
        pass

    from diffusers import FluxFillPipeline  # type: ignore[import]

    model_id = settings.DIFFUSERS_MODEL

    if torch.cuda.is_available():
        dtype = torch.bfloat16
        device = "cuda"
    elif torch.backends.mps.is_available():
        dtype = torch.bfloat16
        device = "mps"
    else:
        dtype = torch.float32
        device = "cpu"

    logger.info("Loading FluxFillPipeline (%s, device=%s)…", model_id, device)

    _pipe = FluxFillPipeline.from_pretrained(
        model_id,
        torch_dtype=dtype,
        token=settings.HF_TOKEN or None,
    ).to(device)

    logger.info("FluxFillPipeline ready.")
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

    w, h = image.size
    # FLUX Fill requires dimensions divisible by 8
    w = (w // 8) * 8
    h = (h // 8) * 8
    image = image.resize((w, h), Image.LANCZOS)
    mask = mask.resize((w, h), Image.LANCZOS)

    generator = torch.Generator().manual_seed(seed)
    result = pipe(
        prompt=prompt,
        image=image,
        mask_image=mask,
        height=h,
        width=w,
        num_inference_steps=steps,
        guidance_scale=guidance,
        generator=generator,
    )
    return result.images[0]


class DiffusersFluxFillProvider:
    """Local FLUX.1 Fill Dev inpainting via diffusers."""

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
                settings.DIFFUSERS_STEPS,
                settings.DIFFUSERS_GUIDANCE,
                1000 + i * 42,
            )
            for i in range(n)
        ]
        return list(await asyncio.gather(*tasks))
