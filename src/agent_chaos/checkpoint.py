# SPDX-License-Identifier: Apache-2.0
"""Semantic checkpoints: reasoning states captured at decision points, each tagged with a
coherence score, plus the restoration objective used by cognitive rollback."""

from __future__ import annotations

import copy
import itertools
import time
from collections import defaultdict, deque
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any


@dataclass
class SemanticCheckpoint:
    id: int
    agent: str
    state: Any
    coherence: float
    t: float
    step: int = 0
    label: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def score(self, now: float, decay_lambda: float) -> float:
        """Restoration score ``coherence − λ · seconds_since_checkpoint``."""
        return self.coherence - decay_lambda * max(0.0, now - self.t)


class CheckpointStore:
    """Bounded per-agent history of :class:`SemanticCheckpoint` objects.

    * :meth:`latest_valid` - the most recent checkpoint whose coherence meets
      ``validity_threshold`` (semantic checkpointing);
    * :meth:`best` - ``argmax(coherence − λ·age)`` over the history (cognitive rollback),
      balancing semantic validity against the progress lost by going back further.
    """

    def __init__(
        self,
        *,
        capacity_per_agent: int = 20,
        validity_threshold: float = 0.85,
        decay_lambda: float = 0.001,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.capacity = capacity_per_agent
        self.validity_threshold = validity_threshold
        self.decay_lambda = decay_lambda
        self._clock = clock
        self._ids = itertools.count(1)
        self._store: dict[str, deque[SemanticCheckpoint]] = defaultdict(
            lambda: deque(maxlen=capacity_per_agent)
        )
        self.restorations = 0

    def save(
        self,
        agent: str,
        state: Any,
        coherence: float,
        *,
        step: int = 0,
        label: str | None = None,
        **metadata: Any,
    ) -> SemanticCheckpoint:
        cp = SemanticCheckpoint(
            id=next(self._ids),
            agent=agent,
            state=copy.deepcopy(state),
            coherence=float(coherence),
            t=self._clock(),
            step=step,
            label=label,
            metadata=dict(metadata),
        )
        self._store[agent].append(cp)
        return cp

    def history(self, agent: str) -> list[SemanticCheckpoint]:
        return list(self._store.get(agent, ()))

    def latest(self, agent: str) -> SemanticCheckpoint | None:
        hist = self._store.get(agent)
        return hist[-1] if hist else None

    def latest_valid(self, agent: str) -> SemanticCheckpoint | None:
        for cp in reversed(self._store.get(agent, ())):
            if cp.coherence >= self.validity_threshold:
                return cp
        return None

    def best(
        self, agent: str, now: float | None = None, *, valid_only: bool = True
    ) -> SemanticCheckpoint | None:
        hist = self._store.get(agent)
        if not hist:
            return None
        candidates = (
            [cp for cp in hist if cp.coherence >= self.validity_threshold] if valid_only else list(hist)
        )
        if not candidates:
            return None
        t = self._clock() if now is None else now
        return max(candidates, key=lambda cp: (cp.score(t, self.decay_lambda), cp.id))

    def validity(self, agent: str) -> float:
        """Coherence of the best valid checkpoint (0 when none exists)."""
        cp = self.best(agent)
        return cp.coherence if cp is not None else 0.0

    def restore(self, checkpoint: SemanticCheckpoint) -> Any:
        """Return a deep copy of the checkpoint's state (so the caller can mutate it)."""
        self.restorations += 1
        return copy.deepcopy(checkpoint.state)

    def prune(self, agent: str, *, before_step: int | None = None, keep_last: int | None = None) -> int:
        hist = self._store.get(agent)
        if not hist:
            return 0
        items = list(hist)
        if before_step is not None:
            items = [cp for cp in items if cp.step >= before_step]
        if keep_last is not None:
            items = items[-keep_last:]
        removed = len(hist) - len(items)
        hist.clear()
        hist.extend(items)
        return removed

    def agents(self) -> list[str]:
        return [a for a, h in self._store.items() if h]

    def __len__(self) -> int:
        return sum(len(h) for h in self._store.values())
