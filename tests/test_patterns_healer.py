from __future__ import annotations

from agent_chaos.healer import HealingConfig, SelfHealer
from agent_chaos.patterns import (
    AdaptiveConsensusReset,
    CognitiveRollback,
    FailureContext,
    PatternSelector,
    QuarantineIsolation,
    SemanticCheckpointing,
)


def ctx(**kw):
    base = {"agent": "a", "degraded_agents": ["a"]}
    base.update(kw)
    return FailureContext(**base)


def test_matching_criteria():
    assert SemanticCheckpointing().matches(
        ctx(failure_duration_s=1, affected_agents=1, coherence_degradation=0.1)
    )
    assert not SemanticCheckpointing().matches(ctx(failure_duration_s=6))
    assert CognitiveRollback().matches(
        ctx(failure_duration_s=6, error_propagation_depth=3, checkpoint_validity=0.95)
    )
    assert not CognitiveRollback().matches(
        ctx(failure_duration_s=6, error_propagation_depth=3, checkpoint_validity=0.5)
    )
    assert QuarantineIsolation().matches(
        ctx(
            degraded_agents=["a", "b"],
            affected_agents=2,
            cascading_risk=0.8,
            system_can_operate_degraded=True,
        )
    )
    assert not QuarantineIsolation().matches(
        ctx(affected_agents=2, cascading_risk=0.8, system_can_operate_degraded=False)
    )
    assert AdaptiveConsensusReset().matches(
        ctx(degraded_agents=["a", "b"], reasoning_variance_sigma=2.5, alternative_agents_available=True)
    )
    assert not AdaptiveConsensusReset().matches(ctx(degraded_agents=["a"], reasoning_variance_sigma=2.5))


def test_selector_prefers_matching_then_cost():
    sel = PatternSelector()
    p, ranked = sel.select(ctx(failure_duration_s=1, affected_agents=1, coherence_degradation=0.05))
    assert p.name == "semantic_checkpointing" and ranked[0][0] == "semantic_checkpointing"
    p2, ranked2 = sel.select(
        ctx(failure_duration_s=9, affected_agents=1, coherence_degradation=0.5)
    )  # nothing matches
    assert len(ranked2) == 4 and p2.name == "semantic_checkpointing"  # cheapest by cost
    rows = sel.rank(ctx(degraded_agents=["a", "b"], affected_agents=2, cascading_risk=0.9))
    assert rows[0][0].name == "quarantine_isolation" and rows[0][2]
    custom = PatternSelector([CognitiveRollback()], w_impact=0.0, w_accuracy=0.0)
    assert custom.select(ctx())[0].name == "cognitive_rollback"


def test_healer_end_to_end(clock):
    healer = SelfHealer(
        ["a", "b", "c"],
        HealingConfig(min_consecutive=2, warmup=1),
        dependencies={"b": ["a"], "c": ["b"]},
        clock=clock.now,
    )
    assert healer.downstream("a") == ["b", "c"] and healer.downstream("c") == []
    healer.checkpoint("a", {"plan": "good"}, coherence=1.0)
    clock.advance(1)
    healer.checkpoint("a", {"plan": "better"}, coherence=0.95)
    sig = healer.observe("a", task="t1", output=1.0, coherence=0.99, accuracy=0.98)
    assert not sig.anomalous and healer.open_incident("a") is None
    clock.advance(1)
    s1 = healer.observe("a", task="t2", output=1.0, coherence=0.3)
    assert s1.anomalous and not s1.confirmed
    clock.advance(1)
    s2, action = healer.heal("a", task="t3", output=1.0, coherence=0.2)
    assert (
        s2.confirmed
        and action is not None
        and action.kind == "restore"
        and action.state == {"plan": "better"}
    )
    assert action.pattern == "semantic_checkpointing" and action.alternatives and action.cost > 0
    inc = healer.open_incident("a")
    assert (
        inc is not None
        and inc.action is action
        and inc.context is not None
        and inc.context.failure_duration_s == 1.0
    )
    clock.advance(2)
    healer.observe("a", task="t4", output=1.0, coherence=0.95)  # healthy -> resolves
    assert inc.success and inc.latency_s == 2.0 and healer.open_incident("a") is None
    report = healer.report()
    assert (
        report["incidents"] == 1
        and report["resolved"] == 1
        and report["patterns"] == {"semantic_checkpointing": 1}
    )
    assert report["mean_recovery_latency_s"] == 2.0 and report["checkpoints"] == 2
    assert "restore" in action.describe()


def test_healer_quarantine_and_probation(clock):
    healer = SelfHealer(
        ["a", "b", "c", "d"],
        HealingConfig(min_consecutive=1, probation_observations=2),
        dependencies={"b": ["a"], "c": ["a"], "d": ["a"]},
        clock=clock.now,
    )
    # a and b both degraded, a has many dependents -> quarantine isolation applies
    healer.observe("b", coherence=0.1)
    sig = healer.observe("a", coherence=0.1)
    context = healer.failure_context("a")
    assert context.multiple_agents_degraded and context.cascading_risk > 0.7 and context.affected_agents == 2
    action = healer.recover(sig)
    assert action.kind == "quarantine" and set(action.agents) == {"a", "b"} and healer.is_quarantined("a")
    assert healer.influence()["a"] == 0.0 and abs(sum(healer.influence().values()) - 1.0) < 1e-9
    healer.observe("a", coherence=0.99)
    assert healer.is_quarantined("a")
    healer.observe("a", coherence=0.99)
    assert not healer.is_quarantined("a") and healer.quarantine_log[-1][2] == "release"
    healer.observe("a", coherence=0.1)
    healer.quarantine("a", "manual")
    healer.observe("a", coherence=0.1)
    assert healer.quarantined["a"].healthy_observations == 0


def test_healer_consensus_reset_and_rollback(clock):
    cfg = HealingConfig(min_consecutive=1, warmup=1, variance_sigma=1.0)
    healer = SelfHealer(["a", "b", "c"], cfg, clock=clock.now)
    for step in range(4):
        for agent, val in (("a", 10.0), ("b", 10.0), ("c", 10.0)):
            healer.observe(agent, task=f"t{step}", output=val + 0.01 * step, coherence=0.99, accuracy=0.99)
    healer.observe("c", task="t9", output=10.0, coherence=0.99)
    healer.observe("b", task="t9", output=60.0, coherence=0.99, accuracy=0.2)
    sig = healer.observe("a", task="t9", output=90.0, coherence=0.99, accuracy=0.1)
    fc = healer.failure_context("a")
    assert fc.multiple_agents_degraded and fc.reasoning_variance_sigma > 2 and fc.alternative_agents_available
    action = healer.recover(sig)
    assert (
        action.kind == "reweight" and action.weights is not None and action.weights["c"] > action.weights["a"]
    )

    # cognitive rollback: long-running failure with deep propagation and valid checkpoints
    h2 = SelfHealer(
        ["a", "b", "c", "d"],
        HealingConfig(min_consecutive=1),
        dependencies={"b": ["a"], "c": ["b"], "d": ["c"]},
        clock=clock.now,
    )
    for agent in "abcd":
        h2.checkpoint(agent, {"agent": agent}, coherence=0.97)
    for agent in "dcb":
        h2.observe(agent, coherence=0.2)
    h2.observe("a", coherence=0.2)
    clock.advance(6)
    sig2 = h2.observe("a", coherence=0.2)
    fc2 = h2.failure_context("a")
    assert fc2.failure_duration_s == 6 and fc2.error_propagation_depth == 3 and fc2.checkpoint_validity >= 0.9
    action2 = h2.recover(sig2)
    assert (
        action2.pattern == "cognitive_rollback"
        and action2.kind == "restore"
        and set(action2.states) == {"a", "b", "c", "d"}
    )


def test_no_checkpoint_gives_none_action(clock):
    healer = SelfHealer(["a"], HealingConfig(min_consecutive=1), clock=clock.now)
    sig = healer.observe("a", coherence=0.1)
    action = healer.recover(sig)
    assert action.kind == "none" and action.success is False and not healer.open_incident("a")
    try:
        healer.recover("ghost")
    except ValueError:
        pass
    else:  # pragma: no cover
        raise AssertionError("expected ValueError")
