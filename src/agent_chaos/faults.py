# SPDX-License-Identifier: Apache-2.0
"""The fault taxonomy.

Each :class:`Fault` describes one kind of cognitive-layer or environmental perturbation,
the *surfaces* it can be injected on (tool calls, reasoning steps, inter-agent messages,
memory reads, raw inputs), a probability, optional agent/target/step filters, and an
:meth:`Fault.apply` method that turns a :class:`FaultContext` into an :class:`Outcome`
(delay, raise, replace, drop or reroute). :class:`agent_chaos.ChaosInjector` decides
when faults fire and executes the outcomes.
"""

from __future__ import annotations

import builtins
import enum
import random
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any, ClassVar


class FaultKind(str, enum.Enum):
    NETWORK_DELAY = "network_delay"
    TOOL_ERROR = "tool_error"
    HALLUCINATION_LOOP = "hallucination_loop"
    CYCLIC_DEPENDENCY = "cyclic_dependency"
    STALE_MEMORY = "stale_memory"
    MISCOORDINATION = "miscoordination"
    INPUT_PERTURBATION = "input_perturbation"
    DATA_POISONING = "data_poisoning"
    PERCEPTION_SPOOFING = "perception_spoofing"


class Surface(str, enum.Enum):
    """Where a fault can be injected."""

    TOOL = "tool"
    REASONING = "reasoning"
    MESSAGE = "message"
    MEMORY = "memory"
    INPUT = "input"


@dataclass
class FaultContext:
    """Everything a fault may look at when deciding what to do."""

    surface: Surface
    agent: str | None = None
    target: str | None = None
    payload: Any = None
    previous: Any = None
    history: Sequence[Any] = ()
    step: int = 0
    episode: int = 0
    rng: random.Random = field(default_factory=random.Random)
    agents: tuple[str, ...] = ()
    dependencies: Mapping[str, tuple[str, ...]] = field(default_factory=dict)


@dataclass
class Outcome:
    """What the injector should do with the intercepted operation."""

    action: str = "pass"  # pass | delay | raise | replace | drop | reroute
    delay_s: float = 0.0
    error: BaseException | None = None
    value: Any = None
    detail: dict[str, Any] = field(default_factory=dict)


@dataclass
class InjectionRecord:
    """Log entry written for every injected fault (post-hoc analysis, scorecards)."""

    index: int
    t: float
    episode: int
    step: int
    fault: FaultKind
    surface: Surface
    agent: str | None
    target: str | None
    action: str
    detail: dict[str, Any] = field(default_factory=dict)
    recovered: bool | None = None
    recovery_pattern: str | None = None
    recovery_latency_s: float | None = None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["fault"] = self.fault.value
        d["surface"] = self.surface.value
        return d


class ChaosToolError(RuntimeError):
    """Raised by :class:`ToolError` when no specific built-in error type is configured."""


# --- helpers -------------------------------------------------------------------


def _is_number(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def map_numbers(obj: Any, fn: Callable[[float], float]) -> Any:
    """Apply ``fn`` to every numeric leaf of a nested structure (lists, tuples, dicts)."""
    if _is_number(obj):
        return fn(obj)
    if isinstance(obj, list):
        return [map_numbers(v, fn) for v in obj]
    if isinstance(obj, tuple):
        return tuple(map_numbers(v, fn) for v in obj)
    if isinstance(obj, dict):
        return {k: map_numbers(v, fn) for k, v in obj.items()}
    return obj


def _make_error(name: str, message: str) -> BaseException:
    cls = getattr(builtins, name, None)
    if isinstance(cls, type) and issubclass(cls, BaseException):
        return cls(message)
    return ChaosToolError(message)


# --- base class ---------------------------------------------------------------


class Fault(ABC):
    kind: ClassVar[FaultKind]
    surfaces: ClassVar[frozenset[Surface]]
    description: ClassVar[str] = ""

    def __init__(
        self,
        probability: float = 0.05,
        *,
        agents: Iterable[str] | None = None,
        targets: Iterable[str] | None = None,
        steps: tuple[int, int] | None = None,
        name: str | None = None,
    ) -> None:
        if not 0.0 <= probability <= 1.0:
            raise ValueError("probability must be within [0, 1]")
        self.probability = probability
        self.agents = frozenset(agents) if agents is not None else None
        self.targets = frozenset(targets) if targets is not None else None
        self.steps = steps
        self.name = name or self.kind.value

    def matches(self, ctx: FaultContext) -> bool:
        if ctx.surface not in self.surfaces:
            return False
        if self.agents is not None and ctx.agent not in self.agents:
            return False
        if self.targets is not None and ctx.target not in self.targets:
            return False
        return not (self.steps is not None and not (self.steps[0] <= ctx.step < self.steps[1]))

    def should_fire(self, ctx: FaultContext) -> bool:
        return self.matches(ctx) and (self.probability >= 1.0 or ctx.rng.random() < self.probability)

    @abstractmethod
    def apply(self, ctx: FaultContext) -> Outcome: ...

    def params(self) -> dict[str, Any]:
        return {}

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind.value,
            "name": self.name,
            "probability": self.probability,
            "agents": sorted(self.agents) if self.agents is not None else None,
            "targets": sorted(self.targets) if self.targets is not None else None,
            "steps": list(self.steps) if self.steps else None,
            "params": self.params(),
        }

    @staticmethod
    def from_dict(data: Mapping[str, Any]) -> Fault:
        cls = FAULT_TYPES[FaultKind(data["kind"])]
        steps = data.get("steps")
        return cls(
            probability=float(data.get("probability", 0.05)),
            agents=data.get("agents"),
            targets=data.get("targets"),
            steps=(int(steps[0]), int(steps[1])) if steps else None,
            name=data.get("name"),
            **dict(data.get("params") or {}),
        )

    def __repr__(self) -> str:
        extra = ", ".join(f"{k}={v!r}" for k, v in self.params().items())
        return f"{type(self).__name__}(p={self.probability}{', ' + extra if extra else ''})"


# --- concrete faults ---------------------------------------------------------------


class NetworkDelay(Fault):
    kind = FaultKind.NETWORK_DELAY
    surfaces = frozenset({Surface.MESSAGE, Surface.TOOL})
    description = "Adds latency to inter-agent messages and tool calls (default 50-500 ms)."

    def __init__(
        self, probability: float = 0.2, *, min_ms: float = 50, max_ms: float = 500, **kw: Any
    ) -> None:
        super().__init__(probability, **kw)
        if min_ms < 0 or max_ms < min_ms:
            raise ValueError("need 0 <= min_ms <= max_ms")
        self.min_ms = min_ms
        self.max_ms = max_ms

    def params(self) -> dict[str, Any]:
        return {"min_ms": self.min_ms, "max_ms": self.max_ms}

    def apply(self, ctx: FaultContext) -> Outcome:
        delay_ms = ctx.rng.uniform(self.min_ms, self.max_ms)
        return Outcome("delay", delay_s=delay_ms / 1000.0, detail={"delay_ms": round(delay_ms, 1)})


class ToolError(Fault):
    kind = FaultKind.TOOL_ERROR
    surfaces = frozenset({Surface.TOOL})
    description = "Makes a tool call raise (TimeoutError / ConnectionError / ChaosToolError)."

    def __init__(
        self,
        probability: float = 0.05,
        *,
        errors: Sequence[str] = ("TimeoutError", "ConnectionError", "ChaosToolError"),
        **kw: Any,
    ) -> None:
        super().__init__(probability, **kw)
        self.errors = tuple(errors)

    def params(self) -> dict[str, Any]:
        return {"errors": list(self.errors)}

    def apply(self, ctx: FaultContext) -> Outcome:
        name = ctx.rng.choice(self.errors)
        return Outcome(
            "raise",
            error=_make_error(name, f"injected {name} in tool {ctx.target!r}"),
            detail={"error": name},
        )


class HallucinationLoop(Fault):
    kind = FaultKind.HALLUCINATION_LOOP
    surfaces = frozenset({Surface.REASONING})
    description = (
        "Replaces a reasoning step's output with ungrounded content "
        "(drift, fabrication or a loop on the previous output)."
    )

    def __init__(
        self, probability: float = 0.03, *, mode: str = "auto", magnitude: float = 5.0, **kw: Any
    ) -> None:
        super().__init__(probability, **kw)
        if mode not in ("auto", "drift", "fabricate", "loop"):
            raise ValueError("mode must be auto, drift, fabricate or loop")
        self.mode = mode
        self.magnitude = magnitude

    def params(self) -> dict[str, Any]:
        return {"mode": self.mode, "magnitude": self.magnitude}

    def apply(self, ctx: FaultContext) -> Outcome:
        mode = self.mode
        if mode == "auto":
            mode = "loop" if (ctx.previous is not None and ctx.rng.random() < 0.5) else "drift"
        value = self.corrupt(ctx.payload, ctx.previous, ctx.rng, mode)
        return Outcome("replace", value=value, detail={"mode": mode})

    def corrupt(self, value: Any, previous: Any, rng: random.Random, mode: str) -> Any:
        if mode == "loop":
            return previous if previous is not None else value
        if mode == "fabricate" or (isinstance(value, str) and mode != "loop"):
            if isinstance(value, str):
                claim = rng.choice(
                    [
                        "As confirmed by the 1987 Geneva protocol",
                        "The tool output clearly stated 42",
                        "Repeating the previous conclusion for emphasis",
                        "Per the system administrator's earlier message",
                    ]
                )
                return f"{value} [{claim}] {value}"
            return map_numbers(value, lambda x: x * self.magnitude)

        def drift(x: float) -> float:
            sign = 1 if rng.random() < 0.5 else -1
            return x + sign * self.magnitude * (abs(x) + 1.0) * rng.uniform(0.5, 1.5)

        if _is_number(value) or isinstance(value, (list, tuple, dict)):
            drifted = map_numbers(value, drift)
            if drifted != value:
                return drifted
        return previous if previous is not None else value


class CyclicDependency(Fault):
    kind = FaultKind.CYCLIC_DEPENDENCY
    surfaces = frozenset({Surface.MESSAGE})
    description = "Reroutes a message so that agents wait on each other (source -> dependent -> source)."

    def __init__(self, probability: float = 0.02, **kw: Any) -> None:
        super().__init__(probability, **kw)

    def apply(self, ctx: FaultContext) -> Outcome:
        source = ctx.agent
        dependents = [a for a, deps in ctx.dependencies.items() if source in deps and a != ctx.target]
        new_target = ctx.rng.choice(dependents) if dependents else (source or ctx.target)
        return Outcome("reroute", value=new_target, detail={"cycle": [source, new_target, source]})


class StaleMemory(Fault):
    kind = FaultKind.STALE_MEMORY
    surfaces = frozenset({Surface.MEMORY})
    description = "Serves an outdated version of a memory entry (or a miss) on read."

    def __init__(
        self, probability: float = 0.05, *, staleness: int = 1, max_staleness: int = 3, **kw: Any
    ) -> None:
        super().__init__(probability, **kw)
        self.staleness = max(1, staleness)
        self.max_staleness = max(self.staleness, max_staleness)

    def params(self) -> dict[str, Any]:
        return {"staleness": self.staleness, "max_staleness": self.max_staleness}

    def apply(self, ctx: FaultContext) -> Outcome:
        history = list(ctx.history)
        if len(history) >= 2:
            back = ctx.rng.randint(self.staleness, min(self.max_staleness, len(history) - 1))
            return Outcome("replace", value=history[-1 - back], detail={"versions_back": back})
        return Outcome("replace", value=None, detail={"versions_back": None, "miss": True})


class Miscoordination(Fault):
    kind = FaultKind.MISCOORDINATION
    surfaces = frozenset({Surface.MESSAGE})
    description = "Drops, misroutes or corrupts an inter-agent message."

    def __init__(self, probability: float = 0.05, *, mode: str = "random", **kw: Any) -> None:
        super().__init__(probability, **kw)
        if mode not in ("random", "drop", "misroute", "corrupt"):
            raise ValueError("mode must be random, drop, misroute or corrupt")
        self.mode = mode

    def params(self) -> dict[str, Any]:
        return {"mode": self.mode}

    def apply(self, ctx: FaultContext) -> Outcome:
        mode = self.mode if self.mode != "random" else ctx.rng.choice(["drop", "misroute", "corrupt"])
        if mode == "drop":
            return Outcome("drop", detail={"mode": mode})
        if mode == "misroute":
            others = [a for a in ctx.agents if a not in (ctx.agent, ctx.target)]
            if others:
                return Outcome("reroute", value=ctx.rng.choice(others), detail={"mode": mode})
            return Outcome("drop", detail={"mode": "drop", "fallback": "no other agents"})
        payload = ctx.payload
        if isinstance(payload, str):
            corrupted: Any = payload[: max(1, len(payload) // 2)] + " …[corrupted]"
        elif isinstance(payload, dict):
            keys = list(payload)
            corrupted = {k: v for k, v in payload.items() if k != (keys[0] if keys else None)}
        else:
            corrupted = map_numbers(payload, lambda x: -x)
        return Outcome("replace", value=corrupted, detail={"mode": mode})


class InputPerturbation(Fault):
    kind = FaultKind.INPUT_PERTURBATION
    surfaces = frozenset({Surface.INPUT, Surface.TOOL})
    description = (
        "Adds bounded adversarial noise (epsilon, uniform or sign-based) to numeric inputs / tool arguments."
    )

    def __init__(
        self, probability: float = 0.1, *, epsilon: float = 0.1, mode: str = "uniform", **kw: Any
    ) -> None:
        super().__init__(probability, **kw)
        if mode not in ("uniform", "sign"):
            raise ValueError("mode must be uniform or sign")
        self.epsilon = epsilon
        self.mode = mode

    def params(self) -> dict[str, Any]:
        return {"epsilon": self.epsilon, "mode": self.mode}

    def apply(self, ctx: FaultContext) -> Outcome:
        rng = ctx.rng

        def perturb(x: float) -> float:
            scale = abs(x) + 1.0
            if self.mode == "sign":
                return x + self.epsilon * scale * (1 if rng.random() < 0.5 else -1)
            return x + rng.uniform(-self.epsilon, self.epsilon) * scale

        return Outcome(
            "replace",
            value=map_numbers(ctx.payload, perturb),
            detail={"epsilon": self.epsilon, "mode": self.mode},
        )


class DataPoisoning(Fault):
    kind = FaultKind.DATA_POISONING
    surfaces = frozenset({Surface.INPUT, Surface.TOOL})
    description = "Corrupts a fraction of the records in a batch (label flips, scaled values)."

    def __init__(
        self, probability: float = 0.1, *, fraction: float = 0.1, scale: float = 10.0, **kw: Any
    ) -> None:
        super().__init__(probability, **kw)
        self.fraction = min(1.0, max(0.0, fraction))
        self.scale = scale

    def params(self) -> dict[str, Any]:
        return {"fraction": self.fraction, "scale": self.scale}

    def _poison(self, record: Any) -> Any:
        if isinstance(record, bool):
            return not record
        if isinstance(record, dict):
            return {
                k: (not v if isinstance(v, bool) else v * self.scale if _is_number(v) else v)
                for k, v in record.items()
            }
        return map_numbers(record, lambda x: x * self.scale)

    def apply(self, ctx: FaultContext) -> Outcome:
        payload = ctx.payload
        if isinstance(payload, (list, tuple)) and payload:
            n = max(1, round(len(payload) * self.fraction))
            idx = set(ctx.rng.sample(range(len(payload)), min(n, len(payload))))
            poisoned = [self._poison(r) if i in idx else r for i, r in enumerate(payload)]
            value: Any = tuple(poisoned) if isinstance(payload, tuple) else poisoned
            return Outcome("replace", value=value, detail={"records": len(idx)})
        return Outcome("replace", value=self._poison(payload), detail={"records": 1})


class PerceptionSpoofing(Fault):
    kind = FaultKind.PERCEPTION_SPOOFING
    surfaces = frozenset({Surface.INPUT})
    description = "Replaces an observation with a spoofed one (fixed value or callable)."

    def __init__(self, probability: float = 0.05, *, spoof: Any = None, **kw: Any) -> None:
        super().__init__(probability, **kw)
        self.spoof = spoof

    def params(self) -> dict[str, Any]:
        return {"spoof": self.spoof if not callable(self.spoof) else None}

    def apply(self, ctx: FaultContext) -> Outcome:
        if callable(self.spoof):
            value = self.spoof(ctx.payload, ctx.rng)
        elif self.spoof is not None:
            value = self.spoof
        else:
            value = map_numbers(ctx.payload, lambda x: -x if x != 0 else 1.0)
            if isinstance(ctx.payload, str):
                value = "[spoofed observation]"
        return Outcome(
            "replace", value=value, detail={"spoof": "callable" if callable(self.spoof) else repr(self.spoof)}
        )


FAULT_TYPES: dict[FaultKind, type[Fault]] = {
    cls.kind: cls
    for cls in (
        NetworkDelay,
        ToolError,
        HallucinationLoop,
        CyclicDependency,
        StaleMemory,
        Miscoordination,
        InputPerturbation,
        DataPoisoning,
        PerceptionSpoofing,
    )
}


def describe_faults() -> list[dict[str, Any]]:
    """Machine-readable description of the taxonomy (used by ``agent-chaos faults``)."""
    return [
        {
            "kind": kind.value,
            "class": cls.__name__,
            "surfaces": sorted(s.value for s in cls.surfaces),
            "description": cls.description,
            "defaults": cls().params(),
            "default_probability": cls().probability,
        }
        for kind, cls in FAULT_TYPES.items()
    ]
