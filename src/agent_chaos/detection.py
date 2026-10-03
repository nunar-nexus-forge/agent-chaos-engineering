# SPDX-License-Identifier: Apache-2.0
"""Anomaly detection for the cognitive layer.

Two signals are tracked per agent:

* **reasoning variance** ``V_i(t)`` - the mean absolute deviation of agent *i*'s outputs
  from the mean output of its peers on the same tasks; anomalous when it exceeds a
  threshold derived from the agent's own history (``mean + k·σ``, ``k = 1.5`` by default);
* **semantic coherence** ``C_i(t)`` - the fraction of domain constraints the output
  satisfies; anomalous when below a fixed threshold (``0.85`` by default).

A **temporal filter** requires the anomaly condition to hold for ``min_consecutive``
observations before it is *confirmed*, which suppresses transient noise.

:class:`PathDiscrepancyDetector` compares the outputs of redundant inference paths
(the adversarial-resilience architecture) and flags disagreement above a threshold.
"""

from __future__ import annotations

import math
import statistics
import time
from collections import Counter, defaultdict, deque
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any


def as_vector(x: Any) -> list[float] | None:
    """Interpret ``x`` as a numeric vector, or return ``None`` if it is not numeric."""
    if isinstance(x, bool):
        return None
    if isinstance(x, (int, float)):
        return [float(x)]
    if (
        isinstance(x, (list, tuple))
        and x
        and all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in x)
    ):
        return [float(v) for v in x]
    return None


def default_distance(a: Any, b: Any) -> float:
    """Mean absolute difference for numeric outputs; 0/1 (dis)agreement otherwise."""
    va, vb = as_vector(a), as_vector(b)
    if va is not None and vb is not None and len(va) == len(vb):
        return sum(abs(x - y) for x, y in zip(va, vb)) / len(va)
    return 0.0 if a == b else 1.0


def _mean_vector(vectors: Sequence[list[float]]) -> list[float]:
    n = len(vectors)
    return [sum(v[i] for v in vectors) / n for i in range(len(vectors[0]))]


class ReasoningVarianceMonitor:
    """Tracks per-task outputs of every agent and computes ``V_i(t)``."""

    def __init__(
        self, sigma: float = 1.5, history: int = 50, distance: Callable[[Any, Any], float] | None = None
    ) -> None:
        self.sigma = sigma
        self.history_size = history
        self.distance = distance or default_distance
        self._outputs: dict[str, dict[str, Any]] = defaultdict(dict)  # task -> agent -> output
        self._recent_tasks: dict[str, deque[str]] = defaultdict(lambda: deque(maxlen=history))
        self._history: dict[str, deque[float]] = defaultdict(lambda: deque(maxlen=history))

    def record(self, agent: str, task: str, output: Any) -> None:
        self._outputs[task][agent] = output
        recent = self._recent_tasks[agent]
        if task in recent:
            recent.remove(task)
        recent.append(task)

    def variance(self, agent: str) -> float | None:
        """Mean deviation of ``agent`` from its peers over the tasks it recently worked on."""
        deviations: list[float] = []
        for task in self._recent_tasks[agent]:
            outputs = self._outputs.get(task, {})
            if agent not in outputs:
                continue
            peers = [o for a, o in outputs.items() if a != agent]
            if not peers:
                continue
            mine = outputs[agent]
            vectors = [v for v in (as_vector(p) for p in peers) if v is not None]
            my_vec = as_vector(mine)
            if my_vec is not None and vectors and all(len(v) == len(my_vec) for v in vectors):
                mu = _mean_vector(vectors)
                deviations.append(sum(abs(x - y) for x, y in zip(my_vec, mu)) / len(mu))
            else:
                deviations.append(sum(self.distance(mine, p) for p in peers) / len(peers))
        if not deviations:
            return None
        return sum(deviations) / len(deviations)

    def update_history(self, agent: str, value: float) -> None:
        self._history[agent].append(value)

    def history(self, agent: str) -> list[float]:
        return list(self._history[agent])

    def threshold(self, agent: str) -> float | None:
        """``mean + sigma·stdev`` of the agent's own variance history (``None`` until 2 samples)."""
        hist = self._history[agent]
        if len(hist) < 2:
            return None
        return statistics.fmean(hist) + self.sigma * statistics.pstdev(hist)

    def z_score(self, agent: str, value: float | None) -> float:
        hist = self._history[agent]
        if value is None or len(hist) < 2:
            return 0.0
        sd = statistics.pstdev(hist)
        if sd == 0:
            return 0.0 if value <= statistics.fmean(hist) else math.inf
        return (value - statistics.fmean(hist)) / sd

    def peers(self, agent: str, task: str) -> dict[str, Any]:
        return {a: o for a, o in self._outputs.get(task, {}).items() if a != agent}


class CoherenceMonitor:
    """Tracks ``C_i(t)`` and how far it dropped from the agent's baseline."""

    def __init__(self, threshold: float = 0.85, history: int = 50) -> None:
        self.threshold = threshold
        self._history: dict[str, deque[float]] = defaultdict(lambda: deque(maxlen=history))

    def record(self, agent: str, coherence: float) -> None:
        self._history[agent].append(coherence)

    def baseline(self, agent: str) -> float | None:
        hist = self._history[agent]
        return statistics.fmean(hist) if hist else None

    def degradation(self, agent: str, current: float) -> float:
        base = self.baseline(agent)
        if base is None:
            return max(0.0, 1.0 - current)
        return max(0.0, base - current)

    def latest(self, agent: str) -> float | None:
        hist = self._history[agent]
        return hist[-1] if hist else None


@dataclass
class AnomalySignal:
    agent: str
    step: int
    t: float
    anomalous: bool
    confirmed: bool
    consecutive: int
    variance: float | None = None
    variance_threshold: float | None = None
    z_score: float = 0.0
    coherence: float | None = None
    coherence_threshold: float = 0.85
    coherence_degradation: float = 0.0
    reasons: list[str] = field(default_factory=list)
    task: str | None = None

    def __str__(self) -> str:
        state = "CONFIRMED" if self.confirmed else ("anomalous" if self.anomalous else "ok")
        return f"{self.agent}@{self.step}: {state} {'; '.join(self.reasons)}".rstrip()


class AnomalyDetector:
    """Combines the variance and coherence monitors with a temporal filter."""

    def __init__(
        self,
        *,
        coherence_threshold: float = 0.85,
        variance_sigma: float = 1.5,
        min_consecutive: int = 2,
        history: int = 50,
        warmup: int = 3,
        distance: Callable[[Any, Any], float] | None = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.variance = ReasoningVarianceMonitor(variance_sigma, history, distance)
        self.coherence = CoherenceMonitor(coherence_threshold, history)
        self.coherence_threshold = coherence_threshold
        self.min_consecutive = max(1, min_consecutive)
        self.warmup = warmup
        self._clock = clock
        self._consecutive: Counter[str] = Counter()
        self._steps: Counter[str] = Counter()
        self._streak_start: dict[str, float] = {}
        self.signals: list[AnomalySignal] = []
        self._last: dict[str, AnomalySignal] = {}

    def observe(
        self,
        agent: str,
        task: str | None = None,
        output: Any = None,
        *,
        coherence: float | None = None,
        constraints: Sequence[bool] | Mapping[str, bool] | None = None,
        step: int | None = None,
    ) -> AnomalySignal:
        """Record one output of ``agent`` and evaluate the anomaly condition.

        ``coherence`` is the constraint-satisfaction ratio; alternatively pass
        ``constraints`` (booleans or ``{name: satisfied}``) and it is computed.
        ``task``/``output`` feed the reasoning-variance monitor (outputs of other agents
        on the same task are the peer reference).
        """
        if step is None:
            step = self._steps[agent]
        self._steps[agent] = step + 1
        reasons: list[str] = []

        variance = threshold = None
        z = 0.0
        if task is not None and output is not None:
            self.variance.record(agent, task, output)
            variance = self.variance.variance(agent)
            threshold = self.variance.threshold(agent)
            z = self.variance.z_score(agent, variance)
            if variance is not None:
                if (
                    threshold is not None
                    and len(self.variance.history(agent)) >= self.warmup
                    and variance > threshold
                ):
                    reasons.append(f"reasoning variance {variance:.3f} > {threshold:.3f}")
                self.variance.update_history(agent, variance)

        c: float | None = None
        if constraints is not None:
            values = list(constraints.values()) if isinstance(constraints, Mapping) else list(constraints)
            c = (sum(1 for v in values if v) / len(values)) if values else 1.0
        elif coherence is not None:
            c = float(coherence)
        degradation = 0.0
        if c is not None:
            degradation = self.coherence.degradation(agent, c)
            self.coherence.record(agent, c)
            if c < self.coherence_threshold:
                reasons.append(f"coherence {c:.2f} < {self.coherence_threshold:.2f}")

        now = self._clock()
        anomalous = bool(reasons)
        if anomalous:
            self._consecutive[agent] += 1
            self._streak_start.setdefault(agent, now)
        else:
            self._consecutive[agent] = 0
            self._streak_start.pop(agent, None)
        consecutive = self._consecutive[agent]
        signal = AnomalySignal(
            agent=agent,
            step=step,
            t=now,
            anomalous=anomalous,
            confirmed=consecutive >= self.min_consecutive,
            consecutive=consecutive,
            variance=variance,
            variance_threshold=threshold,
            z_score=z,
            coherence=c,
            coherence_threshold=self.coherence_threshold,
            coherence_degradation=degradation,
            reasons=reasons,
            task=task,
        )
        self.signals.append(signal)
        self._last[agent] = signal
        return signal

    def consecutive(self, agent: str) -> int:
        return self._consecutive[agent]

    def streak_duration(self, agent: str) -> float:
        start = self._streak_start.get(agent)
        return (self._clock() - start) if start is not None else 0.0

    def last(self, agent: str) -> AnomalySignal | None:
        return self._last.get(agent)

    def anomalous_agents(self) -> list[str]:
        return [a for a, n in self._consecutive.items() if n > 0]

    def reset(self, agent: str) -> None:
        self._consecutive[agent] = 0
        self._streak_start.pop(agent, None)


class PathDiscrepancyDetector:
    """Disagreement between redundant inference paths, normalised by their magnitude."""

    def __init__(self, threshold: float = 0.2) -> None:
        self.threshold = threshold

    def discrepancy(self, outputs: Sequence[Any]) -> float:
        vectors = [as_vector(o) for o in outputs]
        if len(outputs) < 2:
            return 0.0
        if all(v is not None for v in vectors) and len({len(v) for v in vectors if v}) == 1:
            vs = [v for v in vectors if v is not None]
            scale = max(1e-9, sum(abs(x) for v in vs for x in v) / (len(vs) * len(vs[0])))
            worst = 0.0
            for i in range(len(vs)):
                for j in range(i + 1, len(vs)):
                    d = sum(abs(x - y) for x, y in zip(vs[i], vs[j])) / len(vs[i])
                    worst = max(worst, d / scale)
            return worst
        distinct = len({repr(o) for o in outputs})
        return 0.0 if distinct == 1 else 1.0

    def check(self, outputs: Sequence[Any]) -> tuple[float, bool]:
        d = self.discrepancy(outputs)
        return d, d > self.threshold
