from __future__ import annotations

from typing import TypedDict

import pytest

pytest.importorskip("langgraph")

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from agent_chaos.adapters.langgraph import SelfHealingCheckpointSaver, chaos_node, heal_node
from agent_chaos.faults import HallucinationLoop
from agent_chaos.healer import HealingConfig, SelfHealer
from agent_chaos.injector import ChaosInjector
from agent_chaos.plan import ChaosPlan, ScheduleMode
from agent_chaos.runner import VirtualClock


class State(TypedDict):
    n: int


def test_chaos_node_corrupts_output():
    chaos = ChaosInjector(
        ChaosPlan([HallucinationLoop(1.0, mode="drift")], mode=ScheduleMode.ALL, force_primary=False), seed=1
    )

    def step(state: State):
        return {"n": state["n"] + 1}

    g = StateGraph(State)
    g.add_node("step", chaos_node(step, chaos, agent="step"))
    g.add_edge(START, "step")
    g.add_edge("step", END)
    out = g.compile().invoke({"n": 1})
    assert out["n"] != 2 and chaos.records[0].agent == "step"


def test_heal_node_restores_checkpoint():
    clock = VirtualClock()
    healer = SelfHealer(["writer"], HealingConfig(min_consecutive=1), clock=clock.now)
    calls = {"n": 0}

    def writer(state: State):
        calls["n"] += 1
        return {"n": 1000 if calls["n"] == 3 else state["n"] + 1}

    node = heal_node(writer, healer, agent="writer", coherence=lambda upd: [upd["n"] < 100])
    assert node({"n": 1}) == {"n": 2}
    assert node({"n": 2}) == {"n": 3}
    assert node({"n": 3}) == {"n": 3}  # anomalous output replaced by the last valid checkpoint
    assert healer.incidents and healer.incidents[0].action.kind == "restore"


def test_self_healing_checkpoint_saver():
    clock = VirtualClock()
    saver = SelfHealingCheckpointSaver(
        InMemorySaver(),
        coherence=lambda values: 1.0 if values.get("n", 0) < 3 else 0.4,
        threshold=0.85,
        clock=clock.now,
    )

    def inc(state: State):
        clock.advance(1)
        return {"n": state["n"] + 1}

    g = StateGraph(State)
    g.add_node("inc", inc)
    g.add_edge(START, "inc")
    g.add_edge("inc", END)
    graph = g.compile(checkpointer=saver)
    cfg = {"configurable": {"thread_id": "t1"}}
    graph.invoke({"n": 0}, config=cfg)
    graph.invoke({"n": 2}, config=cfg)
    graph.invoke({"n": 3}, config=cfg)
    hist = saver.coherence_history("t1")
    assert hist and hist[0][1] == 0.4 and saver.latest_coherence("t1") == 0.4 and saver.needs_rollback("t1")
    best = saver.best_valid_config("t1")
    assert best is not None
    assert graph.get_state(best).values["n"] < 3
    assert saver.get_tuple(cfg) is not None and list(saver.list(cfg))
    saver.delete_thread("t1")
    assert saver.best_valid_config("t1") is None and saver.latest_coherence("t1") is None
