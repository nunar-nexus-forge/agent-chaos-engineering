# SPDX-License-Identifier: Apache-2.0
"""The four recovery patterns and the selection logic that picks one for a failure.

* **Semantic checkpointing** - return the agent to its last *valid* checkpoint.
* **Cognitive rollback** - return to the checkpoint maximising ``coherence − λ·age``,
  possibly several steps back, when degradation spread undetected.
* **Quarantine isolation** - detach severely degraded agents so they cannot influence
  healthy ones; re-admit after probation.
* **Adaptive consensus reset** - rebalance influence weights by recent accuracy so the
  system degrades gracefully while some agents underperform.

Selection first applies the matching criteria of each pattern (failure duration,
affected agents, propagation depth, checkpoint validity, cascade risk, reasoning variance,
availability of alternative agents) and then minimises

``cost(P) = recovery_time(P) + w_impact · affected_agents(P) + w_accuracy · (1 − accuracy_retention(P))``

over the matching patterns (or all patterns when none matches).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from .checkpoint import SemanticCheckpoint


@dataclass
class FailureContext:
    """What is known about a failure when a recovery pattern must be chosen."""

    agent: str
    degraded_agents: list[str]
    failure_duration_s: float = 0.0
    affected_agents: int = 1
    coherence_degradation: float = 0.0
    error_propagation_depth: int = 0
    checkpoint_validity: float = 0.0
    cascading_risk: float = 0.0
    system_can_operate_degraded: bool = True
    reasoning_variance_sigma: float = 0.0
    alternative_agents_available: bool = True
    downstream: list[str] = field(default_factory=list)

    @property
    def multiple_agents_degraded(self) -> bool:
        return len(self.degraded_agents) > 1


@dataclass
class PatternProfile:
    """Calibration of a pattern's expected cost/benefit: expected recovery time in seconds and
    expected accuracy retention (0-1). The defaults are starting values; recalibrate per system."""

    expected_recovery_s: float
    expected_accuracy_retention: float


@dataclass
class RecoveryAction:
    pattern: str
    kind: str  # restore | quarantine | reweight | none
    agents: list[str]
    state: Any = None
    states: dict[str, Any] = field(default_factory=dict)
    checkpoint: SemanticCheckpoint | None = None
    weights: dict[str, float] | None = None
    detail: dict[str, Any] = field(default_factory=dict)
    latency_s: float = 0.0
    success: bool | None = None
    cost: float = 0.0
    alternatives: list[tuple[str, float]] = field(default_factory=list)

    def describe(self) -> str:
        extra = ""
        if self.kind == "restore" and self.checkpoint is not None:
            extra = f" -> checkpoint #{self.checkpoint.id} (coherence {self.checkpoint.coherence:.2f})"
        elif self.kind == "reweight" and self.weights:
            extra = " -> " + ", ".join(f"{a}={w:.2f}" for a, w in self.weights.items())
        return f"{self.pattern}: {self.kind} {self.agents}{extra}"


class RecoveryPattern(ABC):
    name: str = "pattern"
    profile: PatternProfile = PatternProfile(5.0, 0.85)

    @abstractmethod
    def matches(self, ctx: FailureContext) -> bool: ...

    def affected(self, ctx: FailureContext) -> int:
        return 1

    @abstractmethod
    def apply(self, ctx: FailureContext, healer: Any) -> RecoveryAction: ...

    def cost(self, ctx: FailureContext, *, w_impact: float = 1.0, w_accuracy: float = 10.0) -> float:
        p = self.profile
        return (
            p.expected_recovery_s
            + w_impact * self.affected(ctx)
            + w_accuracy * (1.0 - p.expected_accuracy_retention)
        )

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"


class SemanticCheckpointing(RecoveryPattern):
    name = "semantic_checkpointing"
    profile = PatternProfile(3.8, 0.92)

    def __init__(self, max_duration_s: float = 5.0, max_degradation: float = 0.15) -> None:
        self.max_duration_s = max_duration_s
        self.max_degradation = max_degradation

    def matches(self, ctx: FailureContext) -> bool:
        return (
            ctx.failure_duration_s < self.max_duration_s
            and ctx.affected_agents == 1
            and ctx.coherence_degradation < self.max_degradation
        )

    def apply(self, ctx: FailureContext, healer: Any) -> RecoveryAction:
        cp = healer.checkpoints.latest_valid(ctx.agent)
        if cp is None:
            return RecoveryAction(
                self.name, "none", [ctx.agent], detail={"reason": "no valid checkpoint"}, success=False
            )
        state = healer.checkpoints.restore(cp)
        return RecoveryAction(
            self.name, "restore", [ctx.agent], state=state, states={ctx.agent: state}, checkpoint=cp
        )


class CognitiveRollback(RecoveryPattern):
    name = "cognitive_rollback"
    profile = PatternProfile(4.9, 0.87)

    def __init__(self, min_duration_s: float = 5.0, min_depth: int = 2, min_validity: float = 0.90) -> None:
        self.min_duration_s = min_duration_s
        self.min_depth = min_depth
        self.min_validity = min_validity

    def matches(self, ctx: FailureContext) -> bool:
        return (
            ctx.failure_duration_s > self.min_duration_s
            and ctx.error_propagation_depth > self.min_depth
            and ctx.checkpoint_validity >= self.min_validity
        )

    def affected(self, ctx: FailureContext) -> int:
        return 1 + len(ctx.downstream)

    def apply(self, ctx: FailureContext, healer: Any) -> RecoveryAction:
        states: dict[str, Any] = {}
        chosen: SemanticCheckpoint | None = None
        for agent in [ctx.agent, *[a for a in ctx.degraded_agents if a != ctx.agent]]:
            cp = healer.checkpoints.best(agent)
            if cp is None:
                continue
            states[agent] = healer.checkpoints.restore(cp)
            if agent == ctx.agent:
                chosen = cp
        if not states:
            return RecoveryAction(
                self.name, "none", [ctx.agent], detail={"reason": "no checkpoint history"}, success=False
            )
        return RecoveryAction(
            self.name, "restore", list(states), state=states.get(ctx.agent), states=states, checkpoint=chosen
        )


class QuarantineIsolation(RecoveryPattern):
    name = "quarantine_isolation"
    profile = PatternProfile(5.2, 0.84)

    def __init__(self, min_affected: int = 2, min_risk: float = 0.70) -> None:
        self.min_affected = min_affected
        self.min_risk = min_risk

    def matches(self, ctx: FailureContext) -> bool:
        return (
            ctx.affected_agents >= self.min_affected
            and ctx.cascading_risk > self.min_risk
            and ctx.system_can_operate_degraded
        )

    def affected(self, ctx: FailureContext) -> int:
        return len(ctx.degraded_agents)

    def apply(self, ctx: FailureContext, healer: Any) -> RecoveryAction:
        targets = list(dict.fromkeys([ctx.agent, *ctx.degraded_agents]))
        for agent in targets:
            healer.quarantine(agent, reason=f"{self.name}: cascading risk {ctx.cascading_risk:.2f}")
        return RecoveryAction(
            self.name, "quarantine", targets, detail={"fallback_agents": healer.healthy_agents()}
        )


class AdaptiveConsensusReset(RecoveryPattern):
    name = "adaptive_consensus_reset"
    profile = PatternProfile(5.1, 0.89)

    def __init__(self, min_sigma: float = 2.0) -> None:
        self.min_sigma = min_sigma

    def matches(self, ctx: FailureContext) -> bool:
        return (
            ctx.multiple_agents_degraded
            and ctx.reasoning_variance_sigma > self.min_sigma
            and ctx.alternative_agents_available
        )

    def affected(self, ctx: FailureContext) -> int:
        return len(ctx.degraded_agents)

    def apply(self, ctx: FailureContext, healer: Any) -> RecoveryAction:
        accuracy = dict(healer.accuracy)
        for agent in ctx.degraded_agents:
            accuracy.setdefault(agent, 0.0)
        weights = healer.weights.update(accuracy)
        return RecoveryAction(self.name, "reweight", list(ctx.degraded_agents), weights=weights)


DEFAULT_PATTERNS: tuple[type[RecoveryPattern], ...] = (
    SemanticCheckpointing,
    CognitiveRollback,
    QuarantineIsolation,
    AdaptiveConsensusReset,
)


class PatternSelector:
    def __init__(
        self,
        patterns: Iterable[RecoveryPattern] | None = None,
        *,
        w_impact: float = 1.0,
        w_accuracy: float = 10.0,
    ) -> None:
        self.patterns: list[RecoveryPattern] = (
            list(patterns) if patterns is not None else [cls() for cls in DEFAULT_PATTERNS]
        )
        self.w_impact = w_impact
        self.w_accuracy = w_accuracy

    def rank(self, ctx: FailureContext) -> list[tuple[RecoveryPattern, float, bool]]:
        """All patterns with their cost and whether their criteria matched, cheapest first."""
        rows = [
            (p, p.cost(ctx, w_impact=self.w_impact, w_accuracy=self.w_accuracy), p.matches(ctx))
            for p in self.patterns
        ]
        return sorted(rows, key=lambda r: (not r[2], r[1]))

    def select(self, ctx: FailureContext) -> tuple[RecoveryPattern, list[tuple[str, float]]]:
        ranked = self.rank(ctx)
        matching = [r for r in ranked if r[2]]
        pool = matching or ranked
        best = min(pool, key=lambda r: r[1])
        return best[0], [(p.name, round(c, 3)) for p, c, _m in pool]
