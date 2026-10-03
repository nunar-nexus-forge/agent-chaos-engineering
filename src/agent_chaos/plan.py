# SPDX-License-Identifier: Apache-2.0
"""Chaos plans: which faults, with which schedule, under which seed."""

from __future__ import annotations

import enum
import json
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .faults import (
    CyclicDependency,
    DataPoisoning,
    Fault,
    HallucinationLoop,
    InputPerturbation,
    Miscoordination,
    NetworkDelay,
    PerceptionSpoofing,
    StaleMemory,
    ToolError,
)


class ScheduleMode(str, enum.Enum):
    """How faults are distributed across episodes.

    ``round_robin`` arms one *primary* fault per episode, cycling through the plan so that
    every fault kind is exercised equally (the recommended schedule for fair comparisons).
    ``all`` keeps every fault armed in every episode.
    """

    ROUND_ROBIN = "round_robin"
    ALL = "all"


@dataclass
class ChaosPlan:
    faults: list[Fault] = field(default_factory=list)
    name: str = "plan"
    seed: int | None = None
    mode: ScheduleMode = ScheduleMode.ROUND_ROBIN
    force_primary: bool = True
    expected_steps: int = 10
    max_injections_per_episode: int | None = None

    def active_faults(self, episode: int) -> list[Fault]:
        if not self.faults or self.mode == ScheduleMode.ALL:
            return list(self.faults)
        return [self.faults[episode % len(self.faults)]]

    def primary_fault(self, episode: int) -> Fault | None:
        if not self.faults:
            return None
        return self.faults[episode % len(self.faults)]

    # -- presets -----------------------------------------------------------------

    @classmethod
    def standard(cls, *, seed: int | None = None, mode: ScheduleMode = ScheduleMode.ROUND_ROBIN) -> ChaosPlan:
        """The standard operational-fault plan: 50-500 ms network delays, errors in 5 % of
        tool calls, hallucination loops in 3 % of reasoning steps, and cyclic dependencies."""
        return cls(
            [
                NetworkDelay(0.2, min_ms=50, max_ms=500),
                ToolError(0.05),
                HallucinationLoop(0.03),
                CyclicDependency(0.02),
            ],
            name="standard",
            seed=seed,
            mode=mode,
        )

    @classmethod
    def adversarial(
        cls, *, seed: int | None = None, mode: ScheduleMode = ScheduleMode.ROUND_ROBIN
    ) -> ChaosPlan:
        """The adversarial plan: bounded input noise, data poisoning and perception spoofing."""
        return cls(
            [InputPerturbation(0.1, epsilon=0.1), DataPoisoning(0.1, fraction=0.1), PerceptionSpoofing(0.05)],
            name="adversarial",
            seed=seed,
            mode=mode,
        )

    @classmethod
    def full(cls, *, seed: int | None = None, mode: ScheduleMode = ScheduleMode.ROUND_ROBIN) -> ChaosPlan:
        base = cls.standard(seed=seed, mode=mode).faults
        extra: list[Fault] = [StaleMemory(0.05), Miscoordination(0.05)]
        return cls(base + extra + cls.adversarial().faults, name="full", seed=seed, mode=mode)

    @classmethod
    def preset(cls, name: str, **kw: Any) -> ChaosPlan:
        try:
            return PRESETS[name](**kw)
        except KeyError as e:
            raise KeyError(f"unknown preset {name!r}; choose from {', '.join(PRESETS)}") from e

    # -- (de)serialisation ------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "seed": self.seed,
            "mode": self.mode.value,
            "force_primary": self.force_primary,
            "expected_steps": self.expected_steps,
            "max_injections_per_episode": self.max_injections_per_episode,
            "faults": [f.to_dict() for f in self.faults],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ChaosPlan:
        return cls(
            faults=[Fault.from_dict(f) for f in data.get("faults", [])],
            name=data.get("name", "plan"),
            seed=data.get("seed"),
            mode=ScheduleMode(data.get("mode", "round_robin")),
            force_primary=bool(data.get("force_primary", True)),
            expected_steps=int(data.get("expected_steps", 10)),
            max_injections_per_episode=data.get("max_injections_per_episode"),
        )

    def to_json(self, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_json(cls, text: str) -> ChaosPlan:
        return cls.from_dict(json.loads(text))

    def save(self, path: str | Path) -> None:
        Path(path).write_text(self.to_json(), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> ChaosPlan:
        text = Path(path).read_text(encoding="utf-8")
        if str(path).endswith((".yaml", ".yml")):
            import yaml  # optional dependency

            return cls.from_dict(yaml.safe_load(text))
        return cls.from_json(text)

    def describe(self) -> str:
        lines = [
            f"plan {self.name!r}  mode={self.mode.value}  seed={self.seed}  "
            f"force_primary={self.force_primary}"
        ]
        for f in self.faults:
            filters = []
            if f.agents is not None:
                filters.append(f"agents={sorted(f.agents)}")
            if f.targets is not None:
                filters.append(f"targets={sorted(f.targets)}")
            if f.steps is not None:
                filters.append(f"steps={f.steps}")
            lines.append(f"  - {f.kind.value:<22} p={f.probability:<5} {f.params()} {' '.join(filters)}")
        return "\n".join(lines)

    def __iter__(self) -> Iterable[Fault]:  # type: ignore[override]
        return iter(self.faults)


PRESETS = {"standard": ChaosPlan.standard, "adversarial": ChaosPlan.adversarial, "full": ChaosPlan.full}
