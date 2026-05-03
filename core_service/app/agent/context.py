"""Shared context variables for the agent pipeline."""

from __future__ import annotations

from contextvars import ContextVar

# Number of inpaint variants to generate — set before running the agent
n_variants: ContextVar[int] = ContextVar("n_variants", default=2)

# Inpaint provider override — empty string means use settings.INPAINT_PROVIDER
inpaint_provider: ContextVar[str] = ContextVar("inpaint_provider", default="")
