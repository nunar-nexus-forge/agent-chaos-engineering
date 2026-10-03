# SPDX-License-Identifier: Apache-2.0
"""Influence weights for adaptive consensus reset and trust-weighted decision fusion."""

from __future__ import annotations

import statistics
from collections.abc import Hashable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from .detection import PathDiscrepancyDetector, as_vector


class ConsensusWeights:
    """Per-agent influence weights updated from recent accuracy:

    ``w_i(t+1) = w_i(t) · (1 + α · (accuracy_i(t) − threshold))``

    with ``α = 0.1`` and ``threshold = 0.85`` by default. Weights are clamped to
    ``min_weight`` and (optionally) normalised to sum to 1.
    """

    def __init__(
        self,
        agents: Iterable[str] = (),
        *,
        alpha: float = 0.1,
        threshold: float = 0.85,
        min_weight: float = 0.01,
        normalise: bool = True,
    ) -> None:
        self.alpha = alpha
        self.threshold = threshold
        self.min_weight = min_weight
        self.normalise = normalise
        self.weights: dict[str, float] = dict.fromkeys(agents, 1.0)
        self.updates = 0
        if self.normalise:
            self._normalise()

    def _normalise(self) -> None:
        total = sum(self.weights.values())
        if total > 0:
            self.weights = {a: w / total for a, w in self.weights.items()}

    def ensure(self, agent: str) -> None:
        if agent not in self.weights:
            self.weights[agent] = 1.0 / max(1, len(self.weights)) if self.normalise and self.weights else 1.0
            if self.normalise:
                self._normalise()

    def update(self, accuracy: Mapping[str, float]) -> dict[str, float]:
        for agent, acc in accuracy.items():
            self.ensure(agent)
            new = self.weights[agent] * (1.0 + self.alpha * (float(acc) - self.threshold))
            self.weights[agent] = max(self.min_weight, new)
        if self.normalise:
            self._normalise()
        self.updates += 1
        return dict(self.weights)

    def reset(self, agents: Iterable[str] | None = None) -> None:
        names = list(agents) if agents is not None else list(self.weights)
        self.weights = dict.fromkeys(names, 1.0)
        if self.normalise:
            self._normalise()

    def influence(self) -> dict[str, float]:
        total = sum(self.weights.values())
        return {a: (w / total if total else 0.0) for a, w in self.weights.items()}

    def __getitem__(self, agent: str) -> float:
        return self.weights[agent]


def weighted_consensus(values: Mapping[str, Any], weights: Mapping[str, float] | ConsensusWeights) -> Any:
    """Weighted mean of numeric outputs (scalars or equal-length vectors)."""
    w = weights.weights if isinstance(weights, ConsensusWeights) else weights
    pairs: list[tuple[list[float], float]] = []
    for a, v in values.items():
        vec = as_vector(v)
        wt = w.get(a, 0.0)
        if vec is not None and wt > 0:
            pairs.append((vec, wt))
    if not pairs:
        raise ValueError("no numeric values with positive weight")
    total = sum(wt for _, wt in pairs)
    dim = len(pairs[0][0])
    out = [sum(v[i] * wt for v, wt in pairs) / total for i in range(dim)]
    return out[0] if dim == 1 else out


def weighted_vote(
    values: Mapping[str, Hashable], weights: Mapping[str, float] | ConsensusWeights
) -> tuple[Any, float]:
    """Weighted plurality vote over discrete outputs; returns ``(winner, share)``."""
    w = weights.weights if isinstance(weights, ConsensusWeights) else weights
    tally: dict[Hashable, float] = {}
    for agent, value in values.items():
        tally[value] = tally.get(value, 0.0) + w.get(agent, 0.0)
    if not tally:
        raise ValueError("no votes")
    winner = max(tally.items(), key=lambda kv: kv[1])
    total = sum(tally.values())
    return winner[0], (winner[1] / total if total else 0.0)


@dataclass
class FusedDecision:
    value: Any
    weights: dict[str, float]
    discrepancy: float
    consistent: bool
    suppressed: list[str] = field(default_factory=list)
    flagged_adversarial: bool = False
    detail: dict[str, Any] = field(default_factory=dict)


class TrustWeightedDecision:
    """Decision-layer fusion of redundant inference paths (adversarial-resilience architecture).

    Each path has a trust weight updated from its historical accuracy. A decision:

    1. measures the discrepancy between the paths' outputs;
    2. if it exceeds ``discrepancy_threshold``, flags possible adversarial manipulation and
       suppresses the least-trusted outlier(s) (cross-path consistency check), refusing to
       adopt either extreme;
    3. returns the trust-weighted consensus of the remaining paths.
    """

    def __init__(
        self,
        paths: Iterable[str],
        *,
        alpha: float = 0.1,
        threshold: float = 0.85,
        discrepancy_threshold: float = 0.5,
        min_paths: int = 1,
    ) -> None:
        self.trust = ConsensusWeights(paths, alpha=alpha, threshold=threshold)
        self.discrepancy = PathDiscrepancyDetector(discrepancy_threshold)
        self.min_paths = max(1, min_paths)
        self.history: list[FusedDecision] = []

    def decide(self, outputs: Mapping[str, Any]) -> FusedDecision:
        if not outputs:
            raise ValueError("no path outputs")
        d, disagree = self.discrepancy.check(list(outputs.values()))
        suppressed: list[str] = []
        kept = dict(outputs)
        if disagree and len(kept) > self.min_paths:
            # trust-weighted reference: the weighted median of scalar outputs, else the
            # most trusted path's output
            ref = self._reference(kept)
            ranked = sorted(kept, key=lambda p: (self.trust.weights.get(p, 0.0), -_dist(kept[p], ref)))
            while len(kept) > self.min_paths:
                worst = max(kept, key=lambda p: (_dist(kept[p], ref), -self.trust.weights.get(p, 0.0)))
                if _dist(kept[worst], ref) <= 1e-12:
                    break
                suppressed.append(worst)
                kept.pop(worst)
                if not self.discrepancy.check(list(kept.values()))[1]:
                    break
            _ = ranked
        try:
            value = weighted_consensus(kept, self.trust)
        except ValueError:
            value, _share = weighted_vote(kept, self.trust)
        decision = FusedDecision(
            value=value,
            weights=self.trust.influence(),
            discrepancy=d,
            consistent=not disagree,
            suppressed=suppressed,
            flagged_adversarial=disagree,
            detail={"kept": list(kept)},
        )
        self.history.append(decision)
        return decision

    def _reference(self, outputs: Mapping[str, Any]) -> Any:
        scalars: list[tuple[float, float]] = []
        for p, v in outputs.items():
            vec = as_vector(v)
            if vec is None or len(vec) != 1:
                scalars = []
                break
            scalars.append((vec[0], self.trust.weights.get(p, 0.0)))
        if scalars:
            ordered = sorted(scalars, key=lambda x: x[0])
            total = sum(w for _, w in ordered)
            acc = 0.0
            for x, w in ordered:
                acc += w
                if acc >= total / 2:
                    return x
            return ordered[-1][0]
        best = max(outputs, key=lambda p: self.trust.weights.get(p, 0.0))
        return outputs[best]

    def feedback(self, truth: Any, outputs: Mapping[str, Any]) -> dict[str, float]:
        """Update trust from the correct value: accuracy = 1 − normalised error (or exact match)."""
        acc: dict[str, float] = {}
        tv = as_vector(truth)
        for path, out in outputs.items():
            ov = as_vector(out)
            if tv is not None and ov is not None and len(tv) == len(ov):
                scale = max(1e-9, statistics.fmean(abs(x) for x in tv) or 1.0)
                err = statistics.fmean(abs(x - y) for x, y in zip(tv, ov)) / scale
                acc[path] = max(0.0, 1.0 - err)
            else:
                acc[path] = 1.0 if out == truth else 0.0
        return self.trust.update(acc)


def _dist(a: Any, b: Any) -> float:
    va, vb = as_vector(a), as_vector(b)
    if va is not None and vb is not None and len(va) == len(vb):
        return sum(abs(x - y) for x, y in zip(va, vb)) / len(va)
    return 0.0 if a == b else 1.0


def cross_path_consistency(outputs: Sequence[float], *, extreme: float = 0.9) -> bool:
    """Reject decisions where paths sit at opposite extremes of a probability scale."""
    if len(outputs) < 2:
        return True
    lo, hi = min(outputs), max(outputs)
    return not (lo <= 1.0 - extreme and hi >= extreme)
