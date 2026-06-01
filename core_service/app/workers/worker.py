"""SAQ worker — runs agent sessions in the background."""

from __future__ import annotations

# Patch transformers tokenizers version check BEFORE transformers is imported.
# transformers==4.36.2 (needed for GroundingDINO) gates on tokenizers<0.19,
# but FLUX Fill needs tokenizers>=0.19.
def _patch_tokenizers_check() -> None:
    import importlib.util as _ilu
    import sys as _sys

    # Load the real transformers.utils.versions without triggering the broader
    # transformers import chain (it's a standalone module with no side effects).
    mod_name = "transformers.utils.versions"
    if mod_name not in _sys.modules:
        try:
            spec = _ilu.find_spec(mod_name)
        except (ModuleNotFoundError, ValueError):
            return
        if spec and spec.loader:
            import types as _types
            mod = _types.ModuleType(mod_name)
            spec.loader.exec_module(mod)  # type: ignore[union-attr]
            _sys.modules[mod_name] = mod

    mod = _sys.modules.get(mod_name)
    if mod is None:
        return

    _orig = mod.require_version  # type: ignore[attr-defined]

    def _lenient(requirement: str, hint: str = "") -> None:
        if requirement.startswith("tokenizers"):
            return
        _orig(requirement, hint)

    mod.require_version = _lenient  # type: ignore[attr-defined]

_patch_tokenizers_check()

import asyncio
import logging
import uuid
import warnings
from typing import Any

import saq  # type: ignore[import]

from app.core.config import settings
from app.workers.queue import get_queue

logger = logging.getLogger(__name__)


async def run_agent_session(ctx: dict[str, Any], *, session_id: str, n: int = 2, inpaint_provider: str = "") -> None:
    from app.agent.runner import run_agent_stream
    from app.db.models import AgentSession
    from app.db.session import async_session

    sid = uuid.UUID(session_id)

    async with async_session() as db:
        s = await db.get(AgentSession, sid)
        if s is None:
            logger.error("Session %s not found", session_id)
            return
        s.status = "processing"
        s.events = []
        await db.commit()

    async with async_session() as db:
        s = await db.get(AgentSession, sid)
        if s is None:
            return
        user_prompt = s.user_prompt
        car_image_urls = list(s.car_image_urls or [])
        system_prompt = s.system_prompt or ""

    try:
        async for event in run_agent_stream(
            session_id=session_id,
            user_prompt=user_prompt,
            car_image_urls=car_image_urls,
            system_prompt=system_prompt,
            n=n,
            inpaint_provider=inpaint_provider,
        ):
            async with async_session() as db:
                s = await db.get(AgentSession, sid)
                if s is None:
                    break
                s.events = list(s.events or []) + [event]
                if event.get("type") == "result":
                    s.result_image_urls = event.get("result_image_urls", [])
                await db.commit()

        async with async_session() as db:
            s = await db.get(AgentSession, sid)
            if s:
                s.status = "complete"
                await db.commit()

    except Exception as exc:
        logger.exception("run_agent_session failed: %s", exc)
        async with async_session() as db:
            s = await db.get(AgentSession, sid)
            if s:
                s.status = "failed"
                s.error_message = str(exc)
                await db.commit()


async def _preload_models() -> None:
    """Pre-load SAM2 + GroundingDINO in the right order to avoid torch.jit conflicts."""
    import asyncio
    from app.models.segmentation import _load_gdino, _load_sam2
    loop = asyncio.get_running_loop()
    try:
        await loop.run_in_executor(None, _load_sam2)
        await loop.run_in_executor(None, _load_gdino)
        logger.info("Models pre-loaded successfully.")
    except Exception as exc:
        logger.warning("Model pre-load skipped: %s", exc)


async def main() -> None:
    await _preload_models()
    queue = await get_queue()
    worker = saq.Worker(
        queue,
        functions=[run_agent_session],
        concurrency=2,
    )
    try:
        await worker.start()
    finally:
        await queue.disconnect()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    warnings.filterwarnings("ignore", category=ResourceWarning)
    asyncio.run(main())
