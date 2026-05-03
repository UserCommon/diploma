"""LangChain ReAct agent orchestrating the car tuning pipeline."""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncGenerator
from typing import Any
from uuid import UUID

from langchain_core.callbacks import AsyncCallbackHandler  # type: ignore[import]
from langchain_core.outputs import LLMResult  # type: ignore[import]

from app.core.config import settings

logger = logging.getLogger(__name__)


class _LoggingCallback(AsyncCallbackHandler):
    """Logs every LangChain event at INFO level."""

    async def on_llm_start(self, serialized: dict[str, Any], prompts: list[str], **kw: Any) -> None:
        for i, p in enumerate(prompts):
            logger.info("LLM PROMPT [%d]:\n%s", i, p)

    async def on_llm_end(self, response: LLMResult, **kw: Any) -> None:
        for gen_list in response.generations:
            for gen in gen_list:
                logger.info("LLM RESPONSE:\n%s", gen.text)

    async def on_tool_start(self, serialized: dict[str, Any], input_str: str, **kw: Any) -> None:
        logger.info("TOOL CALL → %s | input: %s", serialized.get("name"), input_str)

    async def on_tool_end(self, output: str, **kw: Any) -> None:
        logger.info("TOOL RESULT: %s", output[:300])

    async def on_agent_action(self, action: Any, **kw: Any) -> None:
        logger.info("AGENT ACTION: tool=%s | input=%s | log:\n%s", action.tool, action.tool_input, action.log)

    async def on_agent_finish(self, finish: Any, **kw: Any) -> None:
        logger.info("AGENT FINISH: %s", finish.return_values)

_DEFAULT_SYSTEM = (
    "You are an AI car tuning assistant. Help users visually tune cars by:\n"
    "1. Searching the parts library for relevant components\n"
    "2. Segmenting the target area on the car to create a mask\n"
    "3. Generating an inpainted result with the new part applied\n\n"
    "CRITICAL RULE for generate_image prompt:\n"
    "Use ONLY the exact part name and description returned by search_parts.\n"
    "Do NOT invent, add, or infer any visual details (color, finish, material, spoke count, etc.) "
    "that are not explicitly stated in the search results.\n"
    "Do NOT use your own knowledge about what that brand/model looks like.\n"
    "Example: if search returns name='Volk Racing TE37' with no color info, "
    "the prompt must be 'Replace wheels with Volk Racing TE37, photorealistic, matching car lighting' — "
    "nothing more.\n\n"
    "If multiple car images are provided, pass the first as source_image_url "
    "and the rest as extra_source_image_urls (comma-separated).\n\n"
    "Think step by step. Use tools in order."
)

AgentEvent = dict[str, Any]


# ── Stub agent ────────────────────────────────────────────────────────────────

async def _run_stub_agent(
    user_prompt: str,
    car_image_urls: list[str],
    system_prompt: str,
    n: int = 2,
    inpaint_provider: str = "",
) -> AsyncGenerator[AgentEvent, None]:
    from app.agent.context import inpaint_provider as _ip_ctx
    from app.agent.context import n_variants
    n_variants.set(n)
    _ip_ctx.set(inpaint_provider)
    from app.agent.tools import generate_image, search_parts, segment_object

    yield {"type": "thinking", "content": "Stub agent active (no OPENAI_API_KEY set)."}

    # 1 — search parts
    yield {"type": "thinking", "content": f"Searching parts for: {user_prompt}"}
    try:
        parts_json: str = await search_parts.ainvoke({"query": user_prompt, "domain": "car"})
        parts: list[dict[str, Any]] = json.loads(parts_json)
    except Exception as exc:
        parts = []
        logger.warning("search_parts failed: %s", exc)
    yield {"type": "tool_call", "tool": "search_parts", "input": user_prompt}
    yield {"type": "tool_result", "tool": "search_parts", "output": parts_json if parts else "[]"}

    if not car_image_urls:
        yield {"type": "result", "output": "No car images provided.", "result_image_urls": []}
        return

    # 2 — segment
    car_url = car_image_urls[0]
    seg_prompt = user_prompt[:60]
    yield {"type": "thinking", "content": f"Segmenting '{seg_prompt}'…"}
    try:
        mask_url: str = await segment_object.ainvoke(
            {"image_url": car_url, "text_prompt": seg_prompt}
        )
    except Exception as exc:
        logger.exception("segment_object failed: %s", exc)
        yield {"type": "error", "message": f"Segmentation failed: {exc}"}
        return
    yield {"type": "tool_call", "tool": "segment_object", "input": f"{car_url} | {seg_prompt}"}
    yield {"type": "tool_result", "tool": "segment_object", "output": mask_url}

    # 3 — generate
    part_img_url = parts[0].get("processed_image_url", "") if parts else ""
    yield {"type": "thinking", "content": "Generating inpainted variants…"}
    try:
        result_json: str = await generate_image.ainvoke({
            "source_image_url": car_url,
            "mask_url": mask_url,
            "prompt": user_prompt,
            "part_image_url": part_img_url,
        })
        result_urls: list[str] = json.loads(result_json)
    except Exception as exc:
        logger.exception("generate_image failed: %s", exc)
        yield {"type": "error", "message": f"Generation failed: {exc}"}
        return
    yield {"type": "tool_call", "tool": "generate_image", "input": user_prompt[:80]}
    yield {"type": "tool_result", "tool": "generate_image", "output": result_json}

    yield {
        "type": "result",
        "output": (
            f"Done. Found {len(parts)} part(s). "
            f"Generated {len(result_urls)} variant(s)."
        ),
        "result_image_urls": result_urls,
    }


# ── LLM factory ──────────────────────────────────────────────────────────────

def _build_llm() -> Any:
    """Instantiate the right LangChain chat model from env vars."""
    provider = settings.LLM_PROVIDER.lower()

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic  # type: ignore[import]
        return ChatAnthropic(
            model=settings.LLM_MODEL,
            api_key=settings.LLM_API_KEY,  # type: ignore[arg-type]
            temperature=0.2,
        )

    if provider == "openai_compatible":
        from langchain_openai import ChatOpenAI  # type: ignore[import]
        return ChatOpenAI(
            model=settings.LLM_MODEL,
            api_key=settings.LLM_API_KEY,  # type: ignore[arg-type]
            base_url=settings.LLM_BASE_URL or None,
            temperature=0.2,
            request_timeout=300,
        )

    # default: openai
    from langchain_openai import ChatOpenAI  # type: ignore[import]
    return ChatOpenAI(
        model=settings.LLM_MODEL,
        api_key=settings.LLM_API_KEY,  # type: ignore[arg-type]
        temperature=0.2,
        request_timeout=300,
    )


# ── Real LangChain ReAct agent ────────────────────────────────────────────────

async def _run_real_agent(
    user_prompt: str,
    car_image_urls: list[str],
    system_prompt: str,
    n: int = 2,
    inpaint_provider: str = "",
) -> AsyncGenerator[AgentEvent, None]:
    from app.agent.context import inpaint_provider as _ip_ctx
    from app.agent.context import n_variants
    n_variants.set(n)
    _ip_ctx.set(inpaint_provider)
    from langchain.agents import AgentExecutor, create_tool_calling_agent  # type: ignore[import]
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder  # type: ignore[import]

    from app.agent.tools import generate_image, search_parts, segment_object

    tools = [search_parts, segment_object, generate_image]
    llm = _build_llm()

    sys = system_prompt.strip() or _DEFAULT_SYSTEM
    images_ctx = "\n\nCar images:\n" + "\n".join(f"- {u}" for u in car_image_urls)
    full_input = f"{user_prompt}{images_ctx if car_image_urls else ''}"

    prompt = ChatPromptTemplate.from_messages([
        ("system", sys),
        ("human", "{input}"),
        MessagesPlaceholder("agent_scratchpad"),
    ])

    agent = create_tool_calling_agent(llm, tools, prompt)
    executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        return_intermediate_steps=True,
        max_iterations=10,
        callbacks=[_LoggingCallback()],
    )

    yield {"type": "thinking", "content": f"Agent started ({settings.LLM_PROVIDER}/{settings.LLM_MODEL})…"}

    result: dict[str, Any] = await executor.ainvoke({"input": full_input})

    for action, observation in result.get("intermediate_steps", []):
        yield {"type": "tool_call", "tool": action.tool, "input": str(action.tool_input)}
        yield {"type": "tool_result", "tool": action.tool, "output": str(observation)[:500]}

    result_urls: list[str] = []
    for action, observation in result.get("intermediate_steps", []):
        if action.tool == "generate_image":
            try:
                result_urls.extend(json.loads(observation))
            except Exception:
                pass

    yield {
        "type": "result",
        "output": result.get("output", ""),
        "result_image_urls": result_urls,
    }


# ── Public entry point ────────────────────────────────────────────────────────

async def run_agent_stream(
    session_id: str,
    user_prompt: str,
    car_image_urls: list[str],
    system_prompt: str = "",
    n: int = 2,
    inpaint_provider: str = "",
) -> AsyncGenerator[AgentEvent, None]:
    use_stub = not settings.LLM_PROVIDER.strip()
    runner = _run_stub_agent if use_stub else _run_real_agent

    try:
        async for event in runner(user_prompt, car_image_urls, system_prompt, n=n, inpaint_provider=inpaint_provider):
            yield event
    except Exception as exc:
        logger.exception("Agent error: %s", exc)
        yield {"type": "error", "message": str(exc)}
