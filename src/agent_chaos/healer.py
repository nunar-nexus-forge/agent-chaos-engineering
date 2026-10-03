# SPDX-License-Identifier: Apache-2.0
"""The :class:`SelfHealer`: the resilience layer that turns observations into recovery.

Loop: ``checkpoint`` good states → ``observe`` outputs → (anomaly confirmed) → ``recover``
selects and applies a pattern → the next healthy observation resolves the incident and
measures recovery latency.
"""

from __future__ import annotations

import time
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from .checkpoint import CheckpointStore, SemanticCheckpoint
from .consensus import ConsensusWeights
from .detection import AnomalyDetector, AnomalySignal
from .patterns import FailureContext, PatternSelector, RecoveryAction, RecoveryPattern


@dataclass
class HealingConfig:
    coherence_threshold: float = 0.85
    variance_sigma: float = 1.5
    min_consecutive: int = 2
    warmup: int = 3
    history: int = 50
    checkpoint_capacity: int = 20
    decay_lambda: float = 0.001
    alpha: float = 0.1
    accuracy_threshold: float = 0.85
    w_impact: float = 1.0
    w_accuracy: float = 10.0
    probation_observations: int = 2
    min_healthy_agents: int = 1
    cascade_severity_scale: float = 0.3


@dataclass
class QuarantineEntry:
    since: float
    reason: str
    healthy_observations: int = 0


@dataclass
class Incident:
    id: int
    agent: str
    detected_at: float
    detected_step: int
    signal: AnomalySignal
    context: FailureContext | None = None
    action: RecoveryAction | None = None
    resolved_at: float | None = None
    success: bool | None = None
    propagated_to: list[str] = field(default_factory=list)

    @property
    def latency_s(self) -> float | None:
        return None if self.resolved_at is None else self.resolved_at - self.detected_at

    @property
    def open(self) -> bool:
        return self.resolved_at is None


class SelfHealer:
    def __init__(
        self,
        agents: Iterable[str] = (),
        config: HealingConfig | None = None,
        *,
        dependencies: Mapping[str, Iterable[str]] | None = None,
        patterns: Iterable[RecoveryPattern] | None = None,
        clock: Callable[[], float] = time.time,
        on_recovery: Callable[[RecoveryAction], None] | None = None,
        distance: Callable[[Any, Any], float] | None = None,
    ) -> None:
        self.config = config or HealingConfig()
        c = self.config
        self.agents: list[str] = list(agents)
        self._clock = clock
        self.on_recovery = on_recovery
        self.detector = AnomalyDetector(
            coherence_threshold=c.coherence_threshold,
            variance_sigma=c.variance_sigma,
            min_consecutive=c.min_consecutive,
            history=c.history,
            warmup=c.warmup,
            distance=distance,
            clock=clock,
        )
        self.checkpoints = CheckpointStore(
            capacity_per_agent=c.checkpoint_capacity,
            validity_threshold=c.coherence_threshold,
            decay_lambda=c.decay_lambda,
            clock=clock,
        )
        self.weights = ConsensusWeights(self.agents, alpha=c.alpha, threshold=c.accuracy_threshold)
        self.selector = PatternSelector(patterns, w_impact=c.w_impact, w_accuracy=c.w_accuracy)
        self.dependencies: dict[str, set[str]] = {}
        for consumer, producers in (dependencies or {}).items():
            self.depends(consumer, producers)
        self.quarantined: dict[str, QuarantineEntry] = {}
        self.accuracy: dict[str, float] = {}
        self.incidents: list[Incident] = []
        self.actions: list[RecoveryAction] = []
        self.quarantine_log: list[tuple[float, str, str, str]] = []  # (t, agent, event, reason)

    # -- topology ----------------------------------------------------------------

    def depends(self, consumer: str, producers: Iterable[str]) -> None:
        """Declare that ``consumer`` uses the outputs of ``producers``."""
        for name in [consumer, *producers]:
            if name not in self.agents:
                self.agents.append(name)
                self.weights.ensure(name)
        self.dependencies.setdefault(consumer, set()).update(producers)

    def downstream(self, agent: str) -> list[str]:
        """Agents that (transitively) consume ``agent``'s outputs, in BFS order."""
        seen: list[str] = []
        frontier = [agent]
        while frontier:
            current = frontier.pop(0)
            for consumer, producers in self.dependencies.items():
                if current in producers and consumer not in seen and consumer != agent:
                    seen.append(consumer)
                    frontier.append(consumer)
        return seen

    def propagation_depth(self, agent: str, degraded: Sequence[str]) -> int:
        """Longest chain of degraded consumers starting at ``agent``."""
        degraded_set = set(degraded)

        def depth(node: str, visited: frozenset[str]) -> int:
            best = 0
            for consumer, producers in self.dependencies.items():
                if node in producers and consumer in degraded_set and consumer not in visited:
                    best = max(best, 1 + depth(consumer, visited | {consumer}))
            return best

        return depth(agent, frozenset({agent}))

    # -- checkpoints & observations ------------------------------------------------

    def checkpoint(self, agent: str, state: Any, coherence: float, **meta: Any) -> SemanticCheckpoint:
        if agent not in self.agents:
            self.agents.append(agent)
            self.weights.ensure(agent)
        return self.checkpoints.save(agent, state, coherence, **meta)

    def observe(
        self,
        agent: str,
        task: str | None = None,
        output: Any = None,
        *,
        coherence: float | None = None,
        constraints: Sequence[bool] | Mapping[str, bool] | None = None,
        accuracy: float | None = None,
        step: int | None = None,
    ) -> AnomalySignal:
        """Feed one output to the monitors; resolves open incidents and probation on healthy outputs."""
        if agent not in self.agents:
            self.agents.append(agent)
            self.weights.ensure(agent)
        signal = self.detector.observe(
            agent, task, output, coherence=coherence, constraints=constraints, step=step
        )
        if accuracy is not None:
            self.accuracy[agent] = float(accuracy)
        entry = self.quarantined.get(agent)
        if entry is not None:
            if signal.anomalous:
                entry.healthy_observations = 0
            else:
                entry.healthy_observations += 1
                if entry.healthy_observations >= self.config.probation_observations:
                    self.release(agent, reason="probation passed")
        if not signal.anomalous:
            incident = self.open_incident(agent)
            if incident is not None and incident.action is not None:
                self.resolve(agent, success=True)
        return signal

    def heal(
        self, agent: str, task: str | None = None, output: Any = None, **kw: Any
    ) -> tuple[AnomalySignal, RecoveryAction | None]:
        """``observe`` and, if the anomaly is confirmed and unhandled, ``recover`` in one call."""
        signal = self.observe(agent, task, output, **kw)
        action = None
        incident = self.open_incident(agent)
        if signal.confirmed and (incident is None or incident.action is None):
            action = self.recover(signal)
        return signal, action

    # -- incidents ------------------------------------------------------------------

    def open_incident(self, agent: str) -> Incident | None:
        for inc in reversed(self.incidents):
            if inc.agent == agent and inc.open:
                return inc
        return None

    def resolve(self, agent: str, success: bool = True) -> Incident | None:
        inc = self.open_incident(agent)
        if inc is None:
            return None
        inc.resolved_at = self._clock()
        inc.success = success
        if inc.action is not None:
            inc.action.success = success
        self.detector.reset(agent)
        return inc

    # -- quarantine -----------------------------------------------------------------

    def quarantine(self, agent: str, reason: str = "") -> None:
        if agent not in self.quarantined:
            self.quarantined[agent] = QuarantineEntry(self._clock(), reason)
            self.quarantine_log.append((self._clock(), agent, "quarantine", reason))

    def release(self, agent: str, reason: str = "") -> None:
        if agent in self.quarantined:
            del self.quarantined[agent]
            self.quarantine_log.append((self._clock(), agent, "release", reason))
            self.detector.reset(agent)

    def is_quarantined(self, agent: str) -> bool:
        return agent in self.quarantined

    def healthy_agents(self) -> list[str]:
        anomalous = set(self.detector.anomalous_agents())
        return [a for a in self.agents if a not in self.quarantined and a not in anomalous]

    def influence(self) -> dict[str, float]:
        """Current influence weights (quarantined agents get 0)."""
        weights = self.weights.influence()
        for q in self.quarantined:
            weights[q] = 0.0
        total = sum(weights.values())
        return {a: (w / total if total else 0.0) for a, w in weights.items()}

    # -- recovery -------------------------------------------------------------------

    def failure_context(self, agent: str) -> FailureContext:
        signal = self.detector.last(agent)
        degraded = list(dict.fromkeys([agent, *self.detector.anomalous_agents()]))
        downstream = self.downstream(agent)
        degraded_set = set(degraded)
        affected = len(degraded_set)
        depth = self.propagation_depth(agent, degraded)
        z = signal.z_score if signal is not None else 0.0
        degradation = signal.coherence_degradation if signal is not None else 0.0
        if (
            signal is not None
            and signal.coherence is not None
            and degradation == 0.0
            and signal.coherence < self.config.coherence_threshold
        ):
            degradation = self.config.coherence_threshold - signal.coherence
        severity = min(
            1.0,
            max(
                degradation / self.config.cascade_severity_scale, (z / 3.0) if z != float("inf") else 1.0, 0.0
            ),
        )
        share = (len(downstream) / max(1, len(self.agents) - 1)) if self.agents else 0.0
        risk = min(1.0, 0.5 * share + 0.5 * severity) if downstream else min(1.0, 0.2 * severity)
        remaining_healthy = [a for a in self.healthy_agents() if a not in degraded_set]
        return FailureContext(
            agent=agent,
            degraded_agents=degraded,
            failure_duration_s=self.detector.streak_duration(agent),
            affected_agents=affected,
            coherence_degradation=degradation,
            error_propagation_depth=depth,
            checkpoint_validity=self.checkpoints.validity(agent),
            cascading_risk=risk,
            system_can_operate_degraded=len(remaining_healthy) >= self.config.min_healthy_agents,
            reasoning_variance_sigma=z if z != float("inf") else 10.0,
            alternative_agents_available=bool(remaining_healthy),
            downstream=downstream,
        )

    def recover(self, signal_or_agent: AnomalySignal | str) -> RecoveryAction:
        """Select and apply a recovery pattern for the (confirmed) anomaly."""
        agent = signal_or_agent if isinstance(signal_or_agent, str) else signal_or_agent.agent
        signal = self.detector.last(agent) if isinstance(signal_or_agent, str) else signal_or_agent
        if signal is None:
            raise ValueError(f"no observations for agent {agent!r}")
        incident = self.open_incident(agent)
        if incident is None:
            incident = Incident(
                id=len(self.incidents) + 1,
                agent=agent,
                detected_at=self._clock(),
                detected_step=signal.step,
                signal=signal,
            )
            self.incidents.append(incident)
        ctx = self.failure_context(agent)
        pattern, alternatives = self.selector.select(ctx)
        start = self._clock()
        action = pattern.apply(ctx, self)
        action.latency_s = self._clock() - start
        action.cost = pattern.cost(ctx, w_impact=self.config.w_impact, w_accuracy=self.config.w_accuracy)
        action.alternatives = alternatives
        incident.context = ctx
        incident.action = action
        incident.propagated_to = [d for d in ctx.downstream if d in set(ctx.degraded_agents)]
        self.actions.append(action)
        if action.kind == "none":
            incident.resolved_at = self._clock()
            incident.success = False
        if self.on_recovery is not None:
            self.on_recovery(action)
        return action

    # -- reporting -----------------------------------------------------------------

    def report(self) -> dict[str, Any]:
        latencies = [i.latency_s for i in self.incidents if i.latency_s is not None and i.success]
        return {
            "agents": list(self.agents),
            "incidents": len(self.incidents),
            "resolved": sum(1 for i in self.incidents if i.success),
            "unresolved": sum(1 for i in self.incidents if i.open),
            "mean_recovery_latency_s": (sum(latencies) / len(latencies)) if latencies else None,
            "patterns": dict(Counter(a.pattern for a in self.actions)),
            "quarantined": list(self.quarantined),
            "influence": self.influence(),
            "checkpoints": len(self.checkpoints),
        }
