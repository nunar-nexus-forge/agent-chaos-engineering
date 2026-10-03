from __future__ import annotations

import json

from agent_chaos.faults import FaultKind, HallucinationLoop, ToolError
from agent_chaos.healer import HealingConfig
from agent_chaos.plan import ChaosPlan
from agent_chaos.runner import (
    RunResult,
    ScenarioContext,
    ScenarioOutcome,
    ScenarioRunner,
    VirtualClock,
    injection_recovery_latencies,
)
from agent_chaos.scorecard import EpisodeResult, build_scorecard, summarise_arm
from agent_chaos.sim import AGENTS, DEPENDENCIES, describe, pipeline_scenario


def test_virtual_clock():
    c = VirtualClock(5.0)
    c.sleep(2)
    c.advance(-1)
    assert c.now() == 7.0


def test_pipeline_baseline_is_correct():
    runner = ScenarioRunner(pipeline_scenario, plan=ChaosPlan.standard(), episodes=5, seed=1)
    results = [runner.run_episode("baseline", i) for i in range(5)]
    assert all(r.correct for r in results) and all(not r.injections for r in results)
    assert results[0].agents == len(AGENTS) and runner.dependencies == DEPENDENCIES


def test_hallucination_hurts_and_healing_helps():
    plan = ChaosPlan([HallucinationLoop(0.0, mode="drift", magnitude=8.0)], expected_steps=3)
    runner = ScenarioRunner(
        pipeline_scenario,
        plan=plan,
        episodes=16,
        seed=7,
        params={"steps": 8, "success_threshold": 0.9},
        config=HealingConfig(min_consecutive=1),
    )
    result = runner.run()
    sc = result.scorecard
    assert sc.arms["baseline"].accuracy == 1.0
    assert sc.arms["chaos"].accuracy < 1.0
    assert sc.arms["healed"].accuracy >= sc.arms["chaos"].accuracy
    assert sc.arms["healed"].incidents > 0 and "semantic_checkpointing" in sc.arms["healed"].pattern_usage
    assert sc.arms["chaos"].success_by_fault == {"hallucination_loop": sc.arms["chaos"].accuracy}
    md = sc.to_markdown()
    assert "task success" in md and "hallucination_loop" in md and "recovery patterns used (healed)" in md
    assert sc.to_dict()["arms"]["healed"]["episodes"] == 16
    assert result.scenario.endswith(":pipeline_scenario")


def test_tool_errors_and_standard_plan_run(tmp_path):
    runner = ScenarioRunner(pipeline_scenario, plan=ChaosPlan.standard(), episodes=8, seed=3)
    result = runner.run(("chaos", "healed"))
    assert {r.arm for r in result.results} == {"chaos", "healed"}
    kinds = {r.primary_fault for r in result.results}
    assert kinds == {
        k.value
        for k in (
            FaultKind.NETWORK_DELAY,
            FaultKind.TOOL_ERROR,
            FaultKind.HALLUCINATION_LOOP,
            FaultKind.CYCLIC_DEPENDENCY,
        )
    }
    assert all(r.injections for r in result.results)  # forced primary injection in every episode
    assert result.scorecard.accuracy_retention["chaos"] is None  # no baseline arm
    path = tmp_path / "run.json"
    result.save(path)
    loaded = RunResult.load(path)
    assert (
        len(loaded.results) == 16
        and loaded.results[0].injections[0].fault is result.results[0].injections[0].fault
    )
    assert json.loads(path.read_text())["plan"]["name"] == "standard"


def test_injection_recovery_latencies():
    outcome = ScenarioOutcome(True, 3, step_correct=[True, False, True], step_times=[0.0, 1.0, 2.0])

    class Rec:
        def __init__(self, t):
            self.t = t

    lat, unrec = injection_recovery_latencies(outcome, [Rec(0.5), Rec(2.5)])
    assert lat == [1.5] and unrec == 1


def test_scorecard_edge_cases():
    r = EpisodeResult(
        "chaos",
        0,
        False,
        4,
        4.0,
        0.001,
        decisions=4,
        correct_decisions=1,
        available_steps=3,
        agents=2,
        propagated_failures=2,
        primary_fault="tool_error",
        unrecovered_injections=1,
    )
    s = summarise_arm([r])
    assert (
        s.recovery_latency_s is None
        and s.recovery_rate == 0.0
        and s.decision_integrity == 0.25
        and s.system_availability == 0.75
    )
    assert s.cross_agent_failure_rate == 2 / 8
    sc = build_scorecard([r])
    assert sc.rli is None and sc.composite_reliability is None and "n/a" in sc.to_markdown()
    d = r.to_dict()
    assert EpisodeResult.from_dict(d).primary_fault == "tool_error"


def test_custom_scenario_function():
    def scenario(ctx: ScenarioContext) -> ScenarioOutcome:
        @ctx.chaos.tool("t", agent="w")
        def t():
            return 1

        ok = True
        try:
            t()
        except Exception:
            ok = False
        ctx.clock.advance(1)
        return ScenarioOutcome(
            ok, 1, step_correct=[ok], step_times=[0.0], decisions=1, correct_decisions=int(ok)
        )

    plan = ChaosPlan([ToolError(1.0)], expected_steps=1)
    result = ScenarioRunner(scenario, plan=plan, episodes=3, seed=0, agents=["w"]).run(("baseline", "chaos"))
    assert (
        result.scorecard.arms["baseline"].accuracy == 1.0 and result.scorecard.arms["chaos"].accuracy == 0.0
    )
    assert result.scorecard.arms["chaos"].injections == 3


def test_describe():
    assert describe()["agents"] == list(AGENTS)
