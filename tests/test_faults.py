from __future__ import annotations

import random

import pytest

from agent_chaos.faults import (
    FAULT_TYPES,
    ChaosToolError,
    CyclicDependency,
    DataPoisoning,
    Fault,
    FaultKind,
    HallucinationLoop,
    InputPerturbation,
    Miscoordination,
    NetworkDelay,
    PerceptionSpoofing,
    StaleMemory,
    Surface,
    ToolError,
    describe_faults,
    map_numbers,
)


def test_matching_filters(ctx_factory):
    f = ToolError(1.0, agents=["a"], targets=["search"], steps=(2, 5))
    assert f.matches(ctx_factory(Surface.TOOL, agent="a", target="search", step=3))
    assert not f.matches(ctx_factory(Surface.TOOL, agent="b", target="search", step=3))
    assert not f.matches(ctx_factory(Surface.TOOL, agent="a", target="other", step=3))
    assert not f.matches(ctx_factory(Surface.TOOL, agent="a", target="search", step=7))
    assert not f.matches(ctx_factory(Surface.MESSAGE, agent="a", target="search", step=3))


def test_probability_and_validation(ctx_factory):
    with pytest.raises(ValueError):
        ToolError(1.5)
    never = ToolError(0.0)
    always = ToolError(1.0)
    ctx = ctx_factory(Surface.TOOL, agent="a", target="t")
    assert not never.should_fire(ctx) and always.should_fire(ctx)
    p = ToolError(0.3)
    fires = sum(p.should_fire(ctx_factory(Surface.TOOL, seed=i, target="t")) for i in range(500))
    assert 100 < fires < 200


def test_network_delay(ctx_factory):
    out = NetworkDelay(1.0, min_ms=50, max_ms=500).apply(ctx_factory(Surface.MESSAGE))
    assert out.action == "delay" and 0.05 <= out.delay_s <= 0.5 and out.detail["delay_ms"] >= 50
    with pytest.raises(ValueError):
        NetworkDelay(min_ms=10, max_ms=5)


def test_tool_error_types(ctx_factory):
    out = ToolError(1.0, errors=["TimeoutError"]).apply(ctx_factory(Surface.TOOL, target="search"))
    assert out.action == "raise" and isinstance(out.error, TimeoutError) and "search" in str(out.error)
    out2 = ToolError(1.0, errors=["ChaosToolError"]).apply(ctx_factory(Surface.TOOL))
    assert isinstance(out2.error, ChaosToolError)
    out3 = ToolError(1.0, errors=["NotAnErrorName"]).apply(ctx_factory(Surface.TOOL))
    assert isinstance(out3.error, ChaosToolError)


def test_hallucination_modes(ctx_factory):
    drift = HallucinationLoop(1.0, mode="drift", magnitude=5.0).apply(ctx_factory(payload=10.0))
    assert drift.action == "replace" and abs(drift.value - 10.0) > 20
    loop = HallucinationLoop(1.0, mode="loop").apply(ctx_factory(payload=10.0, previous=3.0))
    assert loop.value == 3.0
    fab = HallucinationLoop(1.0, mode="fabricate").apply(ctx_factory(payload="the answer is 4"))
    assert fab.value.startswith("the answer is 4 [") and fab.detail["mode"] == "fabricate"
    vec = HallucinationLoop(1.0, mode="drift").apply(ctx_factory(payload={"a": [1.0, 2.0], "b": "x"}))
    assert vec.value["b"] == "x" and vec.value["a"] != [1.0, 2.0]
    auto = HallucinationLoop(1.0).apply(ctx_factory(payload="text", previous="older"))
    assert auto.detail["mode"] in ("loop", "drift")
    with pytest.raises(ValueError):
        HallucinationLoop(mode="nope")


def test_cyclic_dependency_reroutes_to_a_dependent(ctx_factory):
    deps = {"b": ("a",), "c": ("b",)}
    out = CyclicDependency(1.0).apply(
        ctx_factory(Surface.MESSAGE, agent="a", target="b", dependencies=deps, agents=("a", "b", "c"))
    )
    assert out.action == "reroute"
    assert out.value == "a"  # only dependent other than the target is none -> back to source
    out2 = CyclicDependency(1.0).apply(ctx_factory(Surface.MESSAGE, agent="a", target="c", dependencies=deps))
    assert out2.value == "b" and out2.detail["cycle"] == ["a", "b", "a"]


def test_stale_memory(ctx_factory):
    out = StaleMemory(1.0, staleness=1, max_staleness=2).apply(
        ctx_factory(Surface.MEMORY, payload=3, history=[1, 2, 3])
    )
    assert out.action == "replace" and out.value in (1, 2) and out.detail["versions_back"] in (1, 2)
    miss = StaleMemory(1.0).apply(ctx_factory(Surface.MEMORY, payload=1, history=[1]))
    assert miss.value is None and miss.detail["miss"]


def test_miscoordination_modes(ctx_factory):
    assert Miscoordination(1.0, mode="drop").apply(ctx_factory(Surface.MESSAGE)).action == "drop"
    mis = Miscoordination(1.0, mode="misroute").apply(
        ctx_factory(Surface.MESSAGE, agent="a", target="b", agents=("a", "b", "c"))
    )
    assert mis.action == "reroute" and mis.value == "c"
    fallback = Miscoordination(1.0, mode="misroute").apply(
        ctx_factory(Surface.MESSAGE, agent="a", target="b", agents=("a", "b"))
    )
    assert fallback.action == "drop"
    corrupt = Miscoordination(1.0, mode="corrupt").apply(ctx_factory(Surface.MESSAGE, payload="hello world"))
    assert corrupt.action == "replace" and "corrupted" in corrupt.value
    cd = Miscoordination(1.0, mode="corrupt").apply(ctx_factory(Surface.MESSAGE, payload={"k": 1, "j": 2}))
    assert cd.value == {"j": 2}
    cn = Miscoordination(1.0, mode="corrupt").apply(ctx_factory(Surface.MESSAGE, payload=4))
    assert cn.value == -4
    with pytest.raises(ValueError):
        Miscoordination(mode="x")


def test_adversarial_faults(ctx_factory):
    pert = InputPerturbation(1.0, epsilon=0.1).apply(ctx_factory(Surface.INPUT, payload=[10.0, 20.0]))
    assert pert.action == "replace" and all(
        abs(a - b) <= 0.1 * (abs(b) + 1) + 1e-9 for a, b in zip(pert.value, [10.0, 20.0])
    )
    sign = InputPerturbation(1.0, epsilon=0.1, mode="sign").apply(ctx_factory(Surface.INPUT, payload=10.0))
    assert abs(abs(sign.value - 10.0) - 1.1) < 1e-9
    with pytest.raises(ValueError):
        InputPerturbation(mode="bad")
    poison = DataPoisoning(1.0, fraction=0.5).apply(
        ctx_factory(Surface.INPUT, payload=[{"label": True, "x": 1.0}] * 4)
    )
    assert poison.detail["records"] == 2 and sum(1 for r in poison.value if r["label"] is False) == 2
    single = DataPoisoning(1.0).apply(ctx_factory(Surface.INPUT, payload=True))
    assert single.value is False
    spoof = PerceptionSpoofing(1.0, spoof=99).apply(ctx_factory(Surface.INPUT, payload=1))
    assert spoof.value == 99
    spoof_fn = PerceptionSpoofing(1.0, spoof=lambda v, rng: v * 2).apply(
        ctx_factory(Surface.INPUT, payload=4)
    )
    assert spoof_fn.value == 8 and spoof_fn.detail["spoof"] == "callable"
    default = PerceptionSpoofing(1.0).apply(ctx_factory(Surface.INPUT, payload="camera frame"))
    assert default.value == "[spoofed observation]"
    default_num = PerceptionSpoofing(1.0).apply(ctx_factory(Surface.INPUT, payload=[0.0, 2.0]))
    assert default_num.value == [1.0, -2.0]


def test_map_numbers_skips_bools():
    assert map_numbers({"a": True, "b": 2, "c": (1, "x")}, lambda x: x * 10) == {
        "a": True,
        "b": 20,
        "c": (10, "x"),
    }


def test_serialisation_roundtrip():
    for kind, cls in FAULT_TYPES.items():
        f = cls(0.25, agents=["a"], targets=["t"], steps=(1, 3), name="custom")
        back = Fault.from_dict(f.to_dict())
        assert type(back) is cls and back.kind is kind and back.probability == 0.25
        assert (
            back.agents == {"a"} and back.targets == {"t"} and back.steps == (1, 3) and back.name == "custom"
        )
        assert back.params() == f.params()
    assert "p=0.25" in repr(NetworkDelay(0.25))


def test_describe_faults_covers_taxonomy():
    rows = describe_faults()
    assert {r["kind"] for r in rows} == {k.value for k in FaultKind}
    assert all(r["description"] for r in rows)


def test_default_rng_in_context():
    from agent_chaos.faults import FaultContext

    ctx = FaultContext(surface=Surface.TOOL)
    assert isinstance(ctx.rng, random.Random)
