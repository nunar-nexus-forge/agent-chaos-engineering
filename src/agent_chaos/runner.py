# SPDX-License-Identifier: Apache-2.0
"""Run a scenario under three arms - baseline (no chaos), chaos, healed (chaos + self-healing)
- with identical seeds, and build a :class:`~agent_chaos.scorecard.Scorecard`."""

from __future__ import annotations

import json
import random
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .healer import HealingConfig, SelfHealer
from .injector import ChaosInjector
from .patterns import RecoveryPattern
from .plan import ChaosPlan
from .scorecard import ARMS, EpisodeResult, Scorecard, build_scorecard


class VirtualClock:
    """A clock that only advances when told to (so injected delays cost no real time)."""

    def __init__(self, start: float = 0.0) -> None:
        self.t = start

    def now(self) -> float:
        return self.t

    def sleep(self, seconds: float) -> None:
        self.t += max(0.0, seconds)

    advance = sleep


@dataclass
class ScenarioContext:
    arm: str
    episode: int
    rng: random.Random
    clock: VirtualClock
    chaos: ChaosInjector
    healer: SelfHealer | None
    agents: tuple[str, ...] = ()
    dependencies: dict[str, tuple[str, ...]] = field(default_factory=dict)
    step_seconds: float = 1.0
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class ScenarioOutcome:
    correct: bool
    steps: int
    step_correct: list[bool] = field(default_factory=list)
    step_times: list[float] = field(default_factory=list)
    failed_agents: list[str] = field(default_factory=list)
    propagated_failures: int = 0
    decisions: int = 0
    correct_decisions: int = 0
    available_steps: int | None = None
    extra: dict[str, Any] = field(default_factory=dict)


ScenarioFn = Callable[[ScenarioContext], ScenarioOutcome]


@dataclass
class RunResult:
    results: list[EpisodeResult]
    scorecard: Scorecard
    plan: dict[str, Any]
    scenario: str
    seed: int
    episodes: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenario": self.scenario,
            "seed": self.seed,
            "episodes": self.episodes,
            "plan": self.plan,
            "scorecard": self.scorecard.to_dict(),
            "results": [r.to_dict() for r in self.results],
        }

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> RunResult:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        results = [EpisodeResult.from_dict(r) for r in data["results"]]
        return cls(
            results,
            build_scorecard(results),
            data.get("plan", {}),
            data.get("scenario", "?"),
            data.get("seed", 0),
            data.get("episodes", 0),
        )


def injection_recovery_latencies(outcome: ScenarioOutcome, records: Sequence[Any]) -> tuple[list[float], int]:
    """Time from each injection to the next correct decision (``None`` -> unrecovered)."""
    latencies: list[float] = []
    unrecovered = 0
    for rec in records:
        recovered_at = None
        for ok, t in zip(outcome.step_correct, outcome.step_times):
            if ok and t > rec.t:
                recovered_at = t
                break
        if recovered_at is None:
            unrecovered += 1
        else:
            latencies.append(recovered_at - rec.t)
    return latencies, unrecovered


class ScenarioRunner:
    def __init__(
        self,
        scenario: ScenarioFn,
        *,
        plan: ChaosPlan,
        episodes: int = 30,
        seed: int = 0,
        agents: Iterable[str] = (),
        dependencies: Mapping[str, Iterable[str]] | None = None,
        config: HealingConfig | None = None,
        patterns: Iterable[RecoveryPattern] | None = None,
        step_seconds: float = 1.0,
        params: Mapping[str, Any] | None = None,
        scorecard_weights: tuple[float, float, float] = (1 / 3, 1 / 3, 1 / 3),
    ) -> None:
        self.scenario = scenario
        self.plan = plan
        self.episodes = episodes
        self.seed = seed
        self.agents = tuple(agents) or tuple(getattr(scenario, "agents", ()))
        deps = dependencies if dependencies is not None else getattr(scenario, "dependencies", None)
        self.dependencies: dict[str, tuple[str, ...]] = {k: tuple(v) for k, v in (deps or {}).items()}
        self.config = config
        self.patterns = list(patterns) if patterns is not None else None
        self.step_seconds = step_seconds
        self.params = dict(params or {})
        self.scorecard_weights = scorecard_weights

    def run_episode(self, arm: str, index: int) -> EpisodeResult:
        episode_seed = self.seed * 100_003 + index
        clock = VirtualClock()
        rng = random.Random(episode_seed)
        chaos = ChaosInjector(
            self.plan,
            seed=episode_seed,
            enabled=arm != "baseline",
            sleep=clock.sleep,
            clock=clock.now,
            agents=self.agents,
            dependencies=self.dependencies,
        )
        chaos.begin_episode(index)
        healer = None
        if arm == "healed":
            healer = SelfHealer(
                self.agents,
                self.config,
                dependencies=self.dependencies,
                clock=clock.now,
                patterns=self.patterns,
            )
        ctx = ScenarioContext(
            arm,
            index,
            rng,
            clock,
            chaos,
            healer,
            self.agents,
            self.dependencies,
            self.step_seconds,
            dict(self.params),
        )
        cpu0 = time.perf_counter()
        t0 = clock.now()
        outcome = self.scenario(ctx)
        cpu = time.perf_counter() - cpu0
        records = chaos.records_for_episode(index)
        latencies, unrecovered = injection_recovery_latencies(outcome, records)
        incidents: list[dict[str, Any]] = []
        if healer is not None:
            for inc in healer.incidents:
                incidents.append(
                    {
                        "agent": inc.agent,
                        "pattern": inc.action.pattern if inc.action else None,
                        "kind": inc.action.kind if inc.action else None,
                        "latency_s": inc.latency_s,
                        "success": inc.success,
                        "propagated_to": list(inc.propagated_to),
                    }
                )
        primary = chaos.primary_fault if arm != "baseline" else None
        return EpisodeResult(
            arm=arm,
            index=index,
            correct=outcome.correct,
            steps=outcome.steps,
            virtual_duration_s=clock.now() - t0,
            cpu_time_s=cpu,
            injections=records,
            incidents=incidents,
            failed_agents=list(outcome.failed_agents),
            propagated_failures=outcome.propagated_failures,
            agents=max(1, len(self.agents)),
            available_steps=outcome.available_steps if outcome.available_steps is not None else outcome.steps,
            decisions=outcome.decisions,
            correct_decisions=outcome.correct_decisions,
            primary_fault=primary.kind.value if primary is not None else None,
            recovery_latencies_s=latencies,
            unrecovered_injections=unrecovered,
            extra=dict(outcome.extra),
        )

    def run(self, arms: Sequence[str] = ARMS) -> RunResult:
        results: list[EpisodeResult] = []
        for arm in arms:
            for i in range(self.episodes):
                results.append(self.run_episode(arm, i))
        name = getattr(self.scenario, "__qualname__", repr(self.scenario))
        module = getattr(self.scenario, "__module__", "")
        return RunResult(
            results,
            build_scorecard(results, weights=self.scorecard_weights),
            self.plan.to_dict(),
            f"{module}:{name}" if module else name,
            self.seed,
            self.episodes,
        )
