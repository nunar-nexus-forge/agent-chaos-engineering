# SPDX-License-Identifier: Apache-2.0
"""LangChain integration: wrap tools so tool-error / delay / perturbation faults apply."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from ..injector import ChaosInjector


def chaos_tool(tool: Any, injector: ChaosInjector, *, agent: str | None = None) -> Any:
    """Return a chaos-wrapped copy of a LangChain ``BaseTool`` (or a plain callable)."""
    try:
        from langchain_core.tools import BaseTool, StructuredTool
    except ImportError as e:  # pragma: no cover - only without the extra
        raise ImportError(
            "chaos_tool needs langchain-core: pip install 'agent-chaos-engineering[langgraph]'"
        ) from e

    if isinstance(tool, BaseTool):
        original = tool

        def call(**kwargs: Any) -> Any:
            return original.invoke(kwargs)

        wrapped: Callable[..., Any] = injector.tool(original.name, agent)(call)
        return StructuredTool.from_function(
            func=wrapped,
            name=original.name,
            description=original.description,
            args_schema=original.args_schema,
        )
    return injector.tool(getattr(tool, "__name__", "tool"), agent)(tool)
