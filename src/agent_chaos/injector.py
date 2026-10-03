# SPDX-License-Identifier: Apache-2.0
"""The :class:`ChaosInjector`: decides when faults fire and applies them at the five surfaces
(tool calls, reasoning steps, messages, memory reads, inputs). Every injection is logged."""

from __future__ import annotations

import copy
import functools
import inspect
import random
import time
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Iterator, Mapping, MutableMapping
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

from .faults import Fault, FaultContext, FaultKind, InjectionRecord, Outcome, Surface
from .plan import ChaosPlan

_NO_REPLACEMENT = object()
F = Callable[..., Any]


@dataclass
class Delivery:
    """Result of passing a message through :meth:`ChaosInjector.message`."""

    delivered: bool
    source: str
    target: str
    content: Any
    delay_s: float = 0.0
    fault: FaultKind | None = None
    original_target: str | None = None
    record: InjectionRecord | None = None


class ChaosInjector:
    """Runs a :class:`~agent_chaos.plan.ChaosPlan` against your code.

    Parameters
    ----------
    plan:
        Which faults may fire. ``None`` means no chaos (every surface passes through).
    seed:
        Seeds the injector's RNG (defaults to ``plan.seed``); the same seed produces the
        same injection decisions, which makes A/B comparisons fair.
    enabled:
        Master switch; see also :meth:`disabled`.
    sleep / clock:
        Injectable time functions. Tests and simulations pass a virtual clock so that
        network delays cost simulated time instead of real time.
    agents / dependencies:
        Known agent names and ``{consumer: [producers]}`` edges; used by misrouting and
        cyclic-dependency faults.
    on_injection:
        Callback receiving every :class:`~agent_chaos.faults.InjectionRecord`.
    """

    def __init__(
        self,
        plan: ChaosPlan | None = None,
        *,
        seed: int | None = None,
        enabled: bool = True,
        sleep: Callable[[float], Any] = time.sleep,
        clock: Callable[[], float] = time.time,
        agents: Iterable[str] = (),
        dependencies: Mapping[str, Iterable[str]] | None = None,
        on_injection: Callable[[InjectionRecord], None] | None = None,
    ) -> None:
        self.plan = plan if plan is not None else ChaosPlan([])
        self.rng = random.Random(seed if seed is not None else self.plan.seed)
        self.enabled = enabled
        self._sleep = sleep
        self._clock = clock
        self.agents = tuple(agents)
        self.dependencies: dict[str, tuple[str, ...]] = {k: tuple(v) for k, v in (dependencies or {}).items()}
        self.on_injection = on_injection
        self.records: list[InjectionRecord] = []
        self.episode = -1
        self.step = 0
        self._active: list[Fault] = []
        self._forced: Fault | None = None
        self._forced_step = 0
        self._forced_done = True
        self._episode_injections = 0
        self._last_output: dict[str, Any] = {}
        self._memory_history: dict[tuple[str, str], list[Any]] = defaultdict(list)
        self.begin_episode(0)

    # -- lifecycle -------------------------------------------------------------

    def begin_episode(self, index: int | None = None) -> None:
        """Start a new episode: arms the schedule's faults and (optionally) a forced injection."""
        self.episode = index if index is not None else self.episode + 1
        self.step = 0
        self._episode_injections = 0
        self._active = self.plan.active_faults(self.episode)
        self._forced = self.plan.primary_fault(self.episode) if self.plan.force_primary else None
        self._forced_step = (
            self.rng.randrange(max(1, self.plan.expected_steps)) if self._forced is not None else 0
        )
        self._forced_done = self._forced is None
        self._last_output.clear()
        self._memory_history.clear()

    def tick(self, step: int | None = None) -> int:
        """Advance the step counter (faults can be windowed by step)."""
        self.step = step if step is not None else self.step + 1
        return self.step

    @property
    def active_faults(self) -> list[Fault]:
        return list(self._active)

    @property
    def primary_fault(self) -> Fault | None:
        return self.plan.primary_fault(self.episode)

    @contextmanager
    def disabled(self) -> Iterator[None]:
        previous = self.enabled
        self.enabled = False
        try:
            yield
        finally:
            self.enabled = previous

    # -- decision -----------------------------------------------------------------

    def decide(
        self,
        surface: Surface | str,
        *,
        agent: str | None = None,
        target: str | None = None,
        payload: Any = None,
        previous: Any = None,
        history: Iterable[Any] = (),
    ) -> tuple[Fault, Outcome] | None:
        """Ask the armed faults whether to inject here. Returns ``(fault, outcome)`` or ``None``."""
        if not self.enabled or not self._active:
            return None
        limit = self.plan.max_injections_per_episode
        if limit is not None and self._episode_injections >= limit:
            return None
        ctx = FaultContext(
            surface=Surface(surface),
            agent=agent,
            target=target,
            payload=payload,
            previous=previous,
            history=tuple(history),
            step=self.step,
            episode=self.episode,
            rng=self.rng,
            agents=self.agents,
            dependencies=self.dependencies,
        )
        fault: Fault | None = None
        if (
            self._forced is not None
            and not self._forced_done
            and self.step >= self._forced_step
            and self._forced.matches(ctx)
        ):
            fault = self._forced
        else:
            for f in self._active:
                if f.should_fire(ctx):
                    fault = f
                    break
        if fault is None:
            return None
        if fault is self._forced:
            self._forced_done = True
        outcome = fault.apply(ctx)
        record = InjectionRecord(
            index=len(self.records),
            t=self._clock(),
            episode=self.episode,
            step=self.step,
            fault=fault.kind,
            surface=ctx.surface,
            agent=agent,
            target=target,
            action=outcome.action,
            detail=dict(outcome.detail),
        )
        self.records.append(record)
        self._episode_injections += 1
        outcome.detail["record_index"] = record.index
        if self.on_injection is not None:
            self.on_injection(record)
        return fault, outcome

    # -- surfaces -----------------------------------------------------------------

    def tool(self, name: str | None = None, agent: str | None = None) -> Callable[[F], F]:
        """Decorator for tool functions: may delay, raise, perturb arguments or replace the result."""

        def decorate(fn: F) -> F:
            tool_name = name or getattr(fn, "__name__", "tool")

            def before(
                args: tuple[Any, ...], kwargs: dict[str, Any]
            ) -> tuple[Any, tuple[Any, ...], dict[str, Any]]:
                payload = {"args": list(args), "kwargs": dict(kwargs)}
                decided = self.decide(Surface.TOOL, agent=agent, target=tool_name, payload=payload)
                if decided is None:
                    return _NO_REPLACEMENT, args, kwargs
                _fault, out = decided
                if out.action == "delay":
                    self._sleep(out.delay_s)
                elif out.action == "raise" and out.error is not None:
                    raise out.error
                elif out.action == "replace":
                    value = out.value
                    if isinstance(value, dict) and set(value) == {"args", "kwargs"}:
                        return _NO_REPLACEMENT, tuple(value["args"]), dict(value["kwargs"])
                    return value, args, kwargs
                return _NO_REPLACEMENT, args, kwargs

            if inspect.iscoroutinefunction(fn):

                @functools.wraps(fn)
                async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                    replacement, a, k = before(args, kwargs)
                    if replacement is not _NO_REPLACEMENT:
                        return replacement
                    return await fn(*a, **k)

                return async_wrapper  # type: ignore[return-value]

            @functools.wraps(fn)
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                replacement, a, k = before(args, kwargs)
                if replacement is not _NO_REPLACEMENT:
                    return replacement
                return fn(*a, **k)

            return wrapper  # type: ignore[return-value]

        return decorate

    def reason(self, agent: str, output: Any, name: str = "reasoning") -> Any:
        """Inline form of :meth:`reasoning_step`: pass a reasoning output through the injector."""
        decided = self.decide(
            Surface.REASONING, agent=agent, target=name, payload=output, previous=self._last_output.get(agent)
        )
        if decided is not None and decided[1].action == "replace":
            output = decided[1].value
        self._last_output[agent] = output
        return output

    def reasoning_step(self, agent: str, name: str | None = None) -> Callable[[F], F]:
        """Decorator for reasoning functions: may replace the output (hallucination loops)."""

        def decorate(fn: F) -> F:
            step_name: str = name or str(getattr(fn, "__name__", "reasoning"))
            if inspect.iscoroutinefunction(fn):

                @functools.wraps(fn)
                async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                    return self.reason(agent, await fn(*args, **kwargs), step_name)

                return async_wrapper  # type: ignore[return-value]

            @functools.wraps(fn)
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                return self.reason(agent, fn(*args, **kwargs), step_name)

            return wrapper  # type: ignore[return-value]

        return decorate

    def message(self, source: str, target: str, content: Any) -> Delivery:
        """Pass an inter-agent message through the injector (delay / drop / reroute / corrupt)."""
        decided = self.decide(Surface.MESSAGE, agent=source, target=target, payload=content)
        delivery = Delivery(True, source, target, content)
        if decided is None:
            return delivery
        fault, out = decided
        delivery.fault = fault.kind
        delivery.record = self.records[-1]
        if out.action == "delay":
            delivery.delay_s = out.delay_s
            self._sleep(out.delay_s)
        elif out.action == "drop":
            delivery.delivered = False
        elif out.action == "reroute":
            delivery.original_target = target
            delivery.target = str(out.value)
        elif out.action == "replace":
            delivery.content = out.value
        return delivery

    def send(self, fn: Callable[[str, str, Any], Any]) -> Callable[[str, str, Any], Any]:
        """Decorator for a ``send(source, target, content)`` function."""

        @functools.wraps(fn)
        def wrapper(source: str, target: str, content: Any) -> Any:
            d = self.message(source, target, content)
            if not d.delivered:
                return None
            return fn(d.source, d.target, d.content)

        return wrapper

    def memory(self, store: MutableMapping[str, Any] | None = None, agent: str = "agent") -> ChaosMemory:
        """Wrap a dict-like memory store; reads may return stale versions or misses."""
        return ChaosMemory(self, store, agent)

    def input(self, agent: str, value: Any, name: str = "input") -> Any:
        """Pass a raw observation / input through the injector (perturbation, poisoning, spoofing)."""
        decided = self.decide(Surface.INPUT, agent=agent, target=name, payload=value)
        if decided is not None and decided[1].action == "replace":
            return decided[1].value
        return value

    # -- bookkeeping ------------------------------------------------------------

    def note_recovery(
        self, record_index: int, *, pattern: str | None, latency_s: float | None, success: bool
    ) -> None:
        rec = self.records[record_index]
        rec.recovered = success
        rec.recovery_pattern = pattern
        rec.recovery_latency_s = latency_s

    def records_for_episode(self, episode: int | None = None) -> list[InjectionRecord]:
        ep = self.episode if episode is None else episode
        return [r for r in self.records if r.episode == ep]

    def summary(self) -> dict[str, Any]:
        return {
            "injections": len(self.records),
            "episodes": len({r.episode for r in self.records}),
            "by_fault": dict(Counter(r.fault.value for r in self.records)),
            "by_surface": dict(Counter(r.surface.value for r in self.records)),
            "by_action": dict(Counter(r.action for r in self.records)),
        }


class ChaosMemory(MutableMapping[str, Any]):
    """Dict-like store whose reads pass through the injector (stale-memory faults)."""

    def __init__(self, injector: ChaosInjector, store: MutableMapping[str, Any] | None, agent: str) -> None:
        self._injector = injector
        self._store: MutableMapping[str, Any] = store if store is not None else {}
        self.agent = agent
        self.stale_reads = 0

    def _read(self, key: str, present: bool) -> tuple[bool, Any]:
        current = self._store.get(key) if present else None
        history = self._injector._memory_history[(self.agent, key)]
        decided = self._injector.decide(
            Surface.MEMORY, agent=self.agent, target=key, payload=current, history=history
        )
        if decided is not None and decided[1].action == "replace":
            self.stale_reads += 1
            value = decided[1].value
            return value is not None, value
        return present, current

    def __getitem__(self, key: str) -> Any:
        present, value = self._read(key, key in self._store)
        if not present:
            raise KeyError(key)
        return value

    def get(self, key: str, default: Any = None) -> Any:  # type: ignore[override]
        present, value = self._read(key, key in self._store)
        return value if present else default

    def __setitem__(self, key: str, value: Any) -> None:
        self._injector._memory_history[(self.agent, key)].append(copy.deepcopy(value))
        self._store[key] = value

    def __delitem__(self, key: str) -> None:
        del self._store[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._store)

    def __len__(self) -> int:
        return len(self._store)

    def __contains__(self, key: object) -> bool:
        return key in self._store

    def history(self, key: str) -> list[Any]:
        return list(self._injector._memory_history[(self.agent, key)])

    def raw(self) -> MutableMapping[str, Any]:
        return self._store
