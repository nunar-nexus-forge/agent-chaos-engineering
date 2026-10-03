from __future__ import annotations

import contextlib

import pytest

pytest.importorskip("ma_trace")

import ma_trace as mt

from agent_chaos.adapters.ma_trace import bind_ma_trace
from agent_chaos.faults import ToolError
from agent_chaos.healer import HealingConfig, SelfHealer
from agent_chaos.injector import ChaosInjector
from agent_chaos.plan import ChaosPlan, ScheduleMode


def test_bridge_logs_into_episode():
    tracer = mt.MATrace(store="memory", otel_metrics=False)
    mt.set_tracer(tracer)
    chaos = ChaosInjector(ChaosPlan([ToolError(1.0)], mode=ScheduleMode.ALL, force_primary=False), seed=1)
    healer = SelfHealer(["a"], HealingConfig(min_consecutive=1))
    bind_ma_trace(chaos, healer)
    with mt.episode("chaos") as ep:
        with contextlib.suppress(Exception):
            chaos.tool("t")(lambda: 1)()
        healer.checkpoint("a", {"s": 1}, 1.0)
        healer.heal("a", coherence=0.1)
    logs = ep.events_of(mt.LogEvent)
    assert logs[0].level == "warning" and "chaos" in logs[0].message
    assert logs[1].level == "info" and logs[1].data["pattern"] == "semantic_checkpointing"
    mt.set_tracer(None)
