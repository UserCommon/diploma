import logging

from app.models.inpaint.protocol import InpaintProvider

logger = logging.getLogger(__name__)

_provider: InpaintProvider | None = None


def _make_provider(name: str) -> InpaintProvider:
    n = name.lower()
    if n == "polza":
        from app.models.inpaint.polza import PolzaInpaintProvider
        return PolzaInpaintProvider()
    if n == "gpt_image":
        from app.models.inpaint.gpt_image import GptImageInpaintProvider
        return GptImageInpaintProvider()
    if n == "diffusers":
        from app.models.inpaint.diffusers_flux import DiffusersFluxFillProvider
        return DiffusersFluxFillProvider()
    if n == "sdxl":
        from app.models.inpaint.diffusers_sdxl import DiffusersSDXLInpaintProvider
        return DiffusersSDXLInpaintProvider()
    if n == "genapi":
        from app.models.inpaint.genapi import GenApiFluxInpaintProvider
        return GenApiFluxInpaintProvider()
    if n == "kontext":
        from app.models.inpaint.genapi_kontext import GenApiKontextProvider
        return GenApiKontextProvider()
    if n == "klein":
        from app.models.inpaint.genapi_klein import GenApiKleinProvider
        return GenApiKleinProvider()
    from app.models.inpaint.stub import StubInpaintProvider
    return StubInpaintProvider()


def get_inpaint_provider() -> InpaintProvider:
    from app.agent.context import inpaint_provider as _ctx

    override = _ctx.get()
    if override:
        return _make_provider(override)

    global _provider
    if _provider is not None:
        return _provider

    from app.core.config import settings

    name = settings.INPAINT_PROVIDER.lower()

    _provider = _make_provider(name)
    logger.info("InpaintProvider: %s", type(_provider).__name__)
    return _provider
