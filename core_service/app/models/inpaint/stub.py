import numpy as np
from PIL import Image


class StubInpaintProvider:
    """Returns source image with a coloured overlay on the mask region.

    Zero dependencies beyond Pillow/NumPy — safe for CI and local dev.
    """

    _COLORS: list[tuple[int, int, int, int]] = [
        (220, 50, 50, 140),
        (50, 180, 50, 140),
        (50, 50, 220, 140),
        (200, 140, 0, 140),
    ]

    async def generate(
        self,
        image: Image.Image,
        mask: Image.Image,
        prompt: str,
        n: int = 1,
        part_image: Image.Image | None = None,
        extra_images: list[Image.Image] | None = None,
    ) -> list[Image.Image]:
        mask_arr = np.array(mask.convert("L"))
        base = image.convert("RGBA")
        results: list[Image.Image] = []

        for i in range(n):
            color = self._COLORS[i % len(self._COLORS)]
            overlay = np.zeros((*mask_arr.shape, 4), dtype=np.uint8)
            overlay[mask_arr > 128] = color
            overlay_img = Image.fromarray(overlay, "RGBA")
            composite = Image.alpha_composite(base, overlay_img).convert("RGB")
            results.append(composite)

        return results
