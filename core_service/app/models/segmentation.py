"""SAM2 + GroundingDINO segmentation — lazy singletons, runs in thread pool.

Falls back to a centre-box stub mask if the models are not installed.
"""

import asyncio
import logging
from typing import Any

import numpy as np
from PIL import Image

from app.core.config import settings

logger = logging.getLogger(__name__)

_sam2_predictor: Any = None
_gdino_model: Any = None


def _get_device() -> str:
    import torch
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"  # MPS lacks ops needed by SAM2/GroundingDINO


def _load_sam2() -> Any:
    global _sam2_predictor
    if _sam2_predictor is not None:
        return _sam2_predictor

    from sam2.build_sam import build_sam2  # type: ignore[import]
    from sam2.sam2_image_predictor import SAM2ImagePredictor  # type: ignore[import]

    device = _get_device()
    logger.info("Loading SAM2 (device=%s)…", device)
    model = build_sam2(settings.SAM2_CONFIG, settings.SAM2_CHECKPOINT, device=device)
    _sam2_predictor = SAM2ImagePredictor(model)
    logger.info("SAM2 ready.")
    return _sam2_predictor


def _load_gdino() -> Any:
    global _gdino_model
    if _gdino_model is not None:
        return _gdino_model

    from groundingdino.util.inference import load_model  # type: ignore[import]

    device = _get_device()
    logger.info("Loading GroundingDINO (device=%s)…", device)
    _gdino_model = load_model(
        settings.GROUNDING_DINO_CONFIG, settings.GROUNDING_DINO_CHECKPOINT,
        device=device,
    )
    logger.info("GroundingDINO ready.")
    return _gdino_model


def _get_gdino_boxes(image: Image.Image, text: str) -> list[list[float]]:
    import torchvision.transforms as T
    from groundingdino.util.inference import predict  # type: ignore[import]

    model = _load_gdino()
    device = _get_device()
    w, h = image.size

    transform = T.Compose([
        T.Resize((800, 800)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    img_tensor: Any = transform(image.convert("RGB"))

    boxes_cx, _logits, _phrases = predict(
        model=model,
        image=img_tensor,
        caption=text,
        box_threshold=0.35,
        text_threshold=0.25,
        device=device,
    )

    result: list[list[float]] = []
    for box in boxes_cx.tolist():
        cx, cy, bw, bh = box
        result.append([
            (cx - bw / 2) * w,
            (cy - bh / 2) * h,
            (cx + bw / 2) * w,
            (cy + bh / 2) * h,
        ])
    return result


def _stub_mask(image: Image.Image) -> Image.Image:
    from PIL import ImageDraw
    w, h = image.size
    mask = Image.new("L", (w, h), 0)
    ImageDraw.Draw(mask).rectangle(
        [int(w * 0.25), int(h * 0.25), int(w * 0.75), int(h * 0.75)],
        fill=255,
    )
    logger.warning("Using stub centre-box mask (SAM2/GroundingDINO not installed).")
    return mask


def _segment_sync(image: Image.Image, text: str) -> Image.Image:
    try:
        import groundingdino  # type: ignore[import]  # noqa: F401
        import sam2  # type: ignore[import]  # noqa: F401
    except ImportError:
        return _stub_mask(image)

    boxes = _get_gdino_boxes(image, text)
    if not boxes:
        logger.warning("GroundingDINO found no boxes for '%s'. Empty mask.", text)
        return Image.new("L", image.size, 0)

    predictor = _load_sam2()
    predictor.set_image(np.array(image.convert("RGB")))

    combined = np.zeros(image.size[::-1], dtype=np.uint8)
    for box in boxes:
        box_np = np.array(box, dtype=np.float32)
        masks, _scores, _ = predictor.predict(
            point_coords=None,
            point_labels=None,
            box=box_np[None, :],
            multimask_output=False,
        )
        combined = np.maximum(combined, masks[0].astype(np.uint8) * 255)

    return Image.fromarray(combined, mode="L")


async def segment_with_text(image: Image.Image, text: str) -> Image.Image:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, _segment_sync, image, text)
