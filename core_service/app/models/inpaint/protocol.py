from typing import Protocol, runtime_checkable

from PIL import Image


@runtime_checkable
class InpaintProvider(Protocol):
    """Strategy protocol for inpainting backends.

    Implement this to add a new provider — no base class needed.
    """

    async def generate(
        self,
        image: Image.Image,
        mask: Image.Image,
        prompt: str,
        n: int = 1,
        part_image: Image.Image | None = None,
        extra_images: list[Image.Image] | None = None,
    ) -> list[Image.Image]:
        """Return *n* inpainted variants of *image* using *mask* as the edit region.

        extra_images: additional car photos (extra angles) passed alongside image.
        """
        ...
