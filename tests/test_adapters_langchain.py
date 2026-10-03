from __future__ import annotations

import pytest

pytest.importorskip("langchain_core")

from langchain_core.tools import tool

from agent_chaos.adapters.langchain import chaos_tool
from agent_chaos.faults import ToolError
from agent_chaos.injector import ChaosInjector
from agent_chaos.plan import ChaosPlan, ScheduleMode


@tool
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


def test_chaos_tool_wraps_base_tool():
    chaos = ChaosInjector(
        ChaosPlan([ToolError(1.0, errors=["TimeoutError"])], mode=ScheduleMode.ALL, force_primary=False),
        seed=1,
    )
    wrapped = chaos_tool(add, chaos, agent="calc")
    assert wrapped.name == "add" and wrapped.description == add.description
    with pytest.raises(TimeoutError):
        wrapped.invoke({"a": 1, "b": 2})
    with chaos.disabled():
        assert wrapped.invoke({"a": 1, "b": 2}) == 3

    def plain(x):
        return x

    with pytest.raises(TimeoutError):
        chaos_tool(plain, chaos)(1)
