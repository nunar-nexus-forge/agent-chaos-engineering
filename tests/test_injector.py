from __future__ import annotations

import asyncio

import pytest

from agent_chaos.faults import (
    ChaosToolError,
    CyclicDependency,
    FaultKind,
    HallucinationLoop,
    InputPerturbation,
    Miscoordination,
    NetworkDelay,
    StaleMemory,
    ToolError,
)
from agent_chaos.injector import ChaosInjector
from agent_chaos.plan import ChaosPlan, ScheduleMode


def make(plan_faults, **kw):
    kw.setdefault("force_primary", False)
    plan = ChaosPlan(
        plan_faults,
        mode=ScheduleMode.ALL,
        force_primary=kw.pop("force_primary"),
        expected_steps=kw.pop("expected_steps", 10),
    )
    return ChaosInjector(plan, seed=kw.pop("seed", 1), **kw)


def test_no_plan_passes_everything_through():
    chaos = ChaosInjector()

    @chaos.tool("t")
    def t(x):
        return x

    assert (
        t(1) == 1
        and chaos.message("a", "b", "m").delivered
        and chaos.reason("a", 5) == 5
        and chaos.records == []
    )
    assert chaos.input("a", 3) == 3


def test_tool_error_and_delay(clock):
    slept = []
    chaos = make([ToolError(1.0, errors=["TimeoutError"])], sleep=slept.append, clock=clock.now)

    @chaos.tool("search", agent="worker")
    def search(q):
        return "ok"

    with pytest.raises(TimeoutError):
        search("x")
    rec = chaos.records[0]
    assert (
        rec.fault is FaultKind.TOOL_ERROR
        and rec.agent == "worker"
        and rec.target == "search"
        and rec.action == "raise"
    )
    delayed = make([NetworkDelay(1.0, min_ms=100, max_ms=100)], sleep=slept.append)

    @delayed.tool("search")
    def search2(q):
        return "ok"

    assert search2("x") == "ok" and slept == [0.1]


def test_tool_argument_perturbation_and_disabled():
    chaos = make([InputPerturbation(1.0, epsilon=0.5)])
    seen = []

    @chaos.tool("calc")
    def calc(x, y=0):
        seen.append((x, y))
        return x + y

    calc(10.0, y=20.0)
    assert seen[0] != (10.0, 20.0) and chaos.records[0].action == "replace"
    with chaos.disabled():
        calc(1.0, y=1.0)
    assert seen[1] == (1.0, 1.0) and len(chaos.records) == 1


def test_async_tool():
    chaos = make([ToolError(1.0, errors=["ConnectionError"])])

    @chaos.tool("fetch")
    async def fetch():
        return 1

    with pytest.raises(ConnectionError):
        asyncio.run(fetch())


def test_reasoning_step_and_previous(clock):
    chaos = make([HallucinationLoop(1.0, mode="loop")], clock=clock.now)

    @chaos.reasoning_step("planner")
    def think(x):
        return x

    assert think(1) == 1  # first call: no previous output -> loop mode falls back to the value itself
    assert think(2) == 1  # looped on previous output
    assert [r.action for r in chaos.records] == ["replace", "replace"]

    @chaos.reasoning_step("async_planner")
    async def athink(x):
        return x

    assert asyncio.run(athink(7)) == 7


def test_messages_drop_reroute_delay():
    slept = []
    chaos = make([Miscoordination(1.0, mode="drop")], sleep=slept.append)
    d = chaos.message("a", "b", "hi")
    assert not d.delivered and d.fault is FaultKind.MISCOORDINATION and d.record is chaos.records[0]
    sent = []
    send = chaos.send(lambda s, t, c: sent.append((s, t, c)))
    assert send("a", "b", "x") is None and sent == []

    chaos2 = make([CyclicDependency(1.0)], agents=("a", "b", "c"), dependencies={"b": ["a"], "c": ["b"]})
    d2 = chaos2.message("a", "c", "task")
    assert d2.delivered and d2.target == "b" and d2.original_target == "c"

    chaos3 = make([NetworkDelay(1.0, min_ms=200, max_ms=200)], sleep=slept.append)
    d3 = chaos3.message("a", "b", "x")
    assert d3.delivered and d3.delay_s == 0.2 and slept == [0.2]

    chaos4 = make([Miscoordination(1.0, mode="corrupt")])
    d4 = chaos4.message("a", "b", "hello there")
    assert d4.delivered and "corrupted" in d4.content


def test_memory_stale_reads_and_history():
    chaos = make([StaleMemory(1.0, staleness=1, max_staleness=1)])
    mem = chaos.memory({}, agent="analyst")
    mem["k"] = 1
    mem["k"] = 2
    mem["k"] = 3
    assert mem.history("k") == [1, 2, 3] and mem.raw()["k"] == 3
    assert mem.get("k") == 2 and mem["k"] == 2 and mem.stale_reads == 2
    assert "k" in mem and len(mem) == 1 and list(mem) == ["k"]
    fresh = chaos.memory({}, agent="other")
    fresh["only"] = 1
    with pytest.raises(KeyError):
        fresh["only"]  # history has one version -> the fault serves a miss
    assert fresh.get("only", "default") == "default"
    del fresh["only"]
    assert "only" not in fresh


def test_input_surface():
    chaos = make([InputPerturbation(1.0, epsilon=0.1)])
    assert chaos.input("sensor", 100.0) != 100.0
    assert chaos.records[0].surface.value == "input"


def test_round_robin_forced_primary_and_limits():
    plan = ChaosPlan(
        [ToolError(0.0), HallucinationLoop(0.0)],
        expected_steps=3,
        force_primary=True,
        max_injections_per_episode=1,
    )
    chaos = ChaosInjector(plan, seed=5)
    chaos.begin_episode(1)  # primary = HallucinationLoop, forced at some step < 3
    assert chaos.primary_fault.kind is FaultKind.HALLUCINATION_LOOP and chaos.active_faults == [
        plan.faults[1]
    ]
    outputs = []
    for step in range(6):
        chaos.tick(step)
        outputs.append(chaos.reason("a", 1.0))
    assert len(chaos.records) == 1 and chaos.records[0].episode == 1 and chaos.records[0].step < 3
    assert any(o != 1.0 for o in outputs)
    chaos.begin_episode()  # auto-increment
    assert chaos.episode == 2 and chaos.primary_fault.kind is FaultKind.TOOL_ERROR
    assert chaos.records_for_episode(1)[0].fault is FaultKind.HALLUCINATION_LOOP


def test_summary_note_recovery_and_callback():
    seen = []
    chaos = make([ToolError(1.0)], on_injection=seen.append)

    @chaos.tool("t")
    def t():
        return 1

    with pytest.raises((TimeoutError, ConnectionError, ChaosToolError)):
        t()
    chaos.note_recovery(0, pattern="semantic_checkpointing", latency_s=1.5, success=True)
    rec = chaos.records[0]
    assert (
        rec.recovered and rec.recovery_pattern == "semantic_checkpointing" and rec.recovery_latency_s == 1.5
    )
    assert seen == [rec] and rec.to_dict()["fault"] == "tool_error"
    s = chaos.summary()
    assert (
        s["injections"] == 1
        and s["by_fault"] == {"tool_error": 1}
        and s["by_surface"] == {"tool": 1}
        and s["episodes"] == 1
    )
