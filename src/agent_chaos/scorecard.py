# SPDX-License-Identifier: Apache-2.0
"""Resilience scorecard: metrics over baseline / chaos / healed runs.

Per arm: task success (accuracy), recovery latency, cross-agent failure rate, recovery
success per fault kind, decision integrity, availability, compute overhead. Across arms:
accuracy retention, Recovery Latency Improvement (RLI), Cross-Agent Failure Rate
Improvement (CAFRI), Accuracy Retention Improvement (ARI) and a weighted composite
reliability score.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

from .faults import InjectionRecord

ARMS: tuple[str, str, str] = ("baseline", "chaos", "healed")


@dataclass
class EpisodeResult:
    arm: str
    index: int
    correct: bool
    steps: int
    virtual_duration_s: float
    cpu_time_s: float
    injections: list[InjectionRecord] = field(default_factory=list)
    incidents: list[dict[str, Any]] = field(default_factory=list)
    failed_agents: list[str] = field(default_factory=list)
    propagated_failures: int = 0
    agents: int = 1
    available_steps: int = 0
    decisions: int = 0
    correct_decisions: int = 0
    primary_fault: str | None = None
    recovery_latencies_s: list[float] = field(default_factory=list)
    unrecovered_injections: int = 0
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["injections"] = [r.to_dict() for r in self.injections]
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> EpisodeResult:
        from .faults import FaultKind, Surface

        injections = []
        for r in d.get("injections", []):
            r = dict(r)
            r["fault"] = FaultKind(r["fault"])
            r["surface"] = Surface(r["surface"])
            injections.append(InjectionRecord(**r))
        d = dict(d)
        d["injections"] = injections
        return cls(**d)


@dataclass
class ArmSummary:
    arm: str
    episodes: int
    accuracy: float
    mean_steps: float
    mean_virtual_duration_s: float
    mean_cpu_time_s: float
    injections: int
    incidents: int
    recovery_latency_s: float | None
    recovery_rate: float | None
    cross_agent_failure_rate: float
    decision_integrity: float
    system_availability: float
    success_by_fault: dict[str, float] = field(default_factory=dict)
    episodes_by_fault: dict[str, int] = field(default_factory=dict)
    pattern_usage: dict[str, int] = field(default_factory=dict)


@dataclass
class Scorecard:
    arms: dict[str, ArmSummary]
    accuracy_retention: dict[str, float | None]
    rli: float | None
    cafri: float | None
    ari: float | None
    composite_reliability: float | None
    overhead: float | None
    weights: tuple[float, float, float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "arms": {k: asdict(v) for k, v in self.arms.items()},
            "accuracy_retention": self.accuracy_retention,
            "rli": self.rli,
            "cafri": self.cafri,
            "ari": self.ari,
            "composite_reliability": self.composite_reliability,
            "overhead": self.overhead,
            "weights": list(self.weights),
        }

    def to_json(self, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    def to_markdown(self) -> str:
        arms = [a for a in ARMS if a in self.arms] + [a for a in self.arms if a not in ARMS]
        rows: list[tuple[str, list[str]]] = [
            ("task success (accuracy)", [_pct(self.arms[a].accuracy) for a in arms]),
            ("accuracy retention vs baseline", [_pct(self.accuracy_retention.get(a)) for a in arms]),
            ("recovery latency (s)", [_num(self.arms[a].recovery_latency_s) for a in arms]),
            ("recovery rate", [_pct(self.arms[a].recovery_rate) for a in arms]),
            ("cross-agent failure rate", [_pct(self.arms[a].cross_agent_failure_rate) for a in arms]),
            ("decision integrity", [_pct(self.arms[a].decision_integrity) for a in arms]),
            ("system availability", [_pct(self.arms[a].system_availability) for a in arms]),
            ("injections", [str(self.arms[a].injections) for a in arms]),
            ("incidents handled", [str(self.arms[a].incidents) for a in arms]),
            ("mean cpu time / episode (ms)", [f"{self.arms[a].mean_cpu_time_s * 1000:.1f}" for a in arms]),
        ]
        out = ["| metric | " + " | ".join(arms) + " |", "|---|" + "---|" * len(arms)]
        out += [f"| {name} | " + " | ".join(vals) + " |" for name, vals in rows]
        out.append("")
        out.append("| improvement (healed vs chaos) | value |")
        out.append("|---|---|")
        out.append(f"| recovery latency improvement (RLI) | {_pct(self.rli)} |")
        out.append(f"| cross-agent failure rate improvement (CAFRI) | {_pct(self.cafri)} |")
        out.append(f"| accuracy retention improvement (ARI) | {_pct(self.ari)} |")
        weights = ", ".join(f"{w:.2f}" for w in self.weights)
        out.append(f"| composite reliability (weights {weights}) | {_pct(self.composite_reliability)} |")
        out.append(f"| compute overhead of healing | {_pct(self.overhead)} |")
        faults = sorted({f for a in arms for f in self.arms[a].success_by_fault})
        if faults:
            out.append("")
            out.append("| recovery success by fault | " + " | ".join(arms) + " |")
            out.append("|---|" + "---|" * len(arms))
            for f in faults:
                out.append(
                    f"| {f} | " + " | ".join(_pct(self.arms[a].success_by_fault.get(f)) for a in arms) + " |"
                )
        usage = {a: self.arms[a].pattern_usage for a in arms if self.arms[a].pattern_usage}
        if usage:
            out.append("")
            for a, u in usage.items():
                out.append(
                    f"recovery patterns used ({a}): " + ", ".join(f"{k} x{v}" for k, v in sorted(u.items()))
                )
        return "\n".join(out)


def _pct(x: float | None) -> str:
    return "n/a" if x is None else f"{x * 100:.1f}%"


def _num(x: float | None) -> str:
    return "n/a" if x is None else f"{x:.2f}"


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def summarise_arm(results: Sequence[EpisodeResult]) -> ArmSummary:
    if not results:
        raise ValueError("no results")
    arm = results[0].arm
    latencies = [lat for r in results for lat in r.recovery_latencies_s]
    total_injections = sum(len(r.injections) for r in results)
    recovered = sum(len(r.recovery_latencies_s) for r in results)
    unrecovered = sum(r.unrecovered_injections for r in results)
    by_fault: dict[str, list[bool]] = defaultdict(list)
    for r in results:
        if r.primary_fault:
            by_fault[r.primary_fault].append(r.correct)
    patterns: Counter[str] = Counter()
    for r in results:
        for inc in r.incidents:
            if inc.get("pattern"):
                patterns[inc["pattern"]] += 1
    cafr = _mean([r.propagated_failures / max(1, r.agents * max(1, r.steps)) for r in results])
    decisions = sum(r.decisions for r in results)
    correct_decisions = sum(r.correct_decisions for r in results)
    steps = sum(r.steps for r in results)
    available = sum(r.available_steps for r in results)
    return ArmSummary(
        arm=arm,
        episodes=len(results),
        accuracy=_mean([1.0 if r.correct else 0.0 for r in results]),
        mean_steps=_mean([float(r.steps) for r in results]),
        mean_virtual_duration_s=_mean([r.virtual_duration_s for r in results]),
        mean_cpu_time_s=_mean([r.cpu_time_s for r in results]),
        injections=total_injections,
        incidents=sum(len(r.incidents) for r in results),
        recovery_latency_s=_mean(latencies) if latencies else None,
        recovery_rate=(recovered / (recovered + unrecovered)) if (recovered + unrecovered) else None,
        cross_agent_failure_rate=cafr,
        decision_integrity=(correct_decisions / decisions) if decisions else 0.0,
        system_availability=(available / steps) if steps else 0.0,
        success_by_fault={f: _mean([1.0 if c else 0.0 for c in v]) for f, v in sorted(by_fault.items())},
        episodes_by_fault={f: len(v) for f, v in sorted(by_fault.items())},
        pattern_usage=dict(sorted(patterns.items())),
    )


def _improvement(
    before: float | None, after: float | None, *, higher_is_better: bool = False
) -> float | None:
    if before is None or after is None or before == 0:
        return None
    return (after - before) / before if higher_is_better else (before - after) / before


def build_scorecard(
    results: Sequence[EpisodeResult], *, weights: tuple[float, float, float] = (1 / 3, 1 / 3, 1 / 3)
) -> Scorecard:
    grouped: dict[str, list[EpisodeResult]] = defaultdict(list)
    for r in results:
        grouped[r.arm].append(r)
    arms = {arm: summarise_arm(rs) for arm, rs in grouped.items()}
    base_acc = arms["baseline"].accuracy if "baseline" in arms else None
    retention: dict[str, float | None] = {}
    for arm, s in arms.items():
        retention[arm] = (s.accuracy / base_acc) if base_acc else None
    chaos, healed = arms.get("chaos"), arms.get("healed")
    rli = cafri = ari = composite = overhead = None
    if chaos and healed:
        rli = _improvement(chaos.recovery_latency_s, healed.recovery_latency_s)
        cafri = _improvement(chaos.cross_agent_failure_rate, healed.cross_agent_failure_rate)
        ari = _improvement(retention.get("chaos"), retention.get("healed"), higher_is_better=True)
        parts = [(w, v) for w, v in zip(weights, (rli, cafri, ari)) if v is not None]
        if parts:
            composite = sum(w * v for w, v in parts) / sum(w for w, _ in parts)
        overhead = _improvement(chaos.mean_cpu_time_s, healed.mean_cpu_time_s, higher_is_better=True)
    return Scorecard(arms, retention, rli, cafri, ari, composite, overhead, weights)
