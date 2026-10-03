# SPDX-License-Identifier: Apache-2.0
"""LangGraph integration.

* :func:`chaos_node` - wrap a node so hallucination-loop faults can corrupt its output;
* :func:`heal_node` - wrap a node with checkpoint / detect / restore (cognitive rollback at
  node granularity);
* :class:`SelfHealingCheckpointSaver` - wrap any LangGraph checkpointer so every checkpoint
  carries a *coherence* score, and ask it for the best valid checkpoint to resume from.
"""

from __future__ import annotations

import builtins
import functools
import inspect
import time
from collections.abc import AsyncIterator, Callable, Iterator
from typing import Any

from ..healer import SelfHealer
from ..injector import ChaosInjector

try:
    from langgraph.checkpoint.base import BaseCheckpointSaver
except ImportError as e:  # pragma: no cover - only without the extra
    raise ImportError(
        "the LangGraph adapter needs langgraph: pip install 'agent-chaos-engineering[langgraph]'"
    ) from e


def chaos_node(
    fn: Callable[..., Any], injector: ChaosInjector, *, agent: str | None = None
) -> Callable[..., Any]:
    """Return a node whose state update passes through the injector's reasoning surface."""
    name: str = agent or str(getattr(fn, "__name__", "node"))
    if inspect.iscoroutinefunction(fn):

        @functools.wraps(fn)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            return injector.reason(name, await fn(*args, **kwargs), name)

        return async_wrapper

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        return injector.reason(name, fn(*args, **kwargs), name)

    return wrapper


def heal_node(
    fn: Callable[..., Any],
    healer: SelfHealer,
    *,
    agent: str | None = None,
    coherence: Callable[[Any], float | list[bool] | tuple[bool, ...]],
    task: str | None = None,
) -> Callable[..., Any]:
    """Wrap a node with the self-healing loop.

    ``coherence(update)`` returns either a ratio in ``[0, 1]`` or a list of constraint
    booleans. Healthy updates are checkpointed; a confirmed anomaly triggers recovery and,
    when the chosen pattern restores a checkpoint, the restored state update is returned
    in place of the node's output.
    """
    name: str = agent or str(getattr(fn, "__name__", "node"))

    def _after(update: Any) -> Any:
        score = coherence(update)
        kwargs: dict[str, Any]
        if isinstance(score, (list, tuple)):
            kwargs = {"constraints": list(score)}
        else:
            kwargs = {"coherence": float(score)}
        signal, action = healer.heal(name, task=task or name, output=None, **kwargs)
        if action is not None and action.kind == "restore" and action.state is not None:
            return action.state
        if not signal.anomalous:
            c = kwargs.get("coherence")
            if c is None:
                cons = kwargs["constraints"]
                c = sum(1 for x in cons if x) / len(cons) if cons else 1.0
            healer.checkpoint(name, update, c)
        return update

    if inspect.iscoroutinefunction(fn):

        @functools.wraps(fn)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            return _after(await fn(*args, **kwargs))

        return async_wrapper

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        return _after(fn(*args, **kwargs))

    return wrapper


class SelfHealingCheckpointSaver(BaseCheckpointSaver):  # type: ignore[misc]
    """Checkpointer extension: tags every checkpoint with a coherence score.

    Wrap any saver (``InMemorySaver``, ``SqliteSaver``, ``PostgresSaver`` ...)::

        saver = SelfHealingCheckpointSaver(InMemorySaver(), coherence=score_state)
        graph = builder.compile(checkpointer=saver)

    After a run, ``saver.best_valid_config(thread_id)`` returns the config of the checkpoint
    maximising ``coherence − λ·age`` among those above the validity threshold. Pass it to
    ``graph.invoke(None, config=...)`` / ``graph.update_state`` to resume from that state
    (LangGraph time travel).
    """

    def __init__(
        self,
        inner: Any,
        *,
        coherence: Callable[[dict[str, Any]], float],
        threshold: float = 0.85,
        decay_lambda: float = 0.001,
        clock: Callable[[], float] = time.time,
    ) -> None:
        super().__init__(serde=getattr(inner, "serde", None))
        self.inner = inner
        self.coherence_fn = coherence
        self.threshold = threshold
        self.decay_lambda = decay_lambda
        self._clock = clock

    # -- delegation -----------------------------------------------------------------

    @property
    def config_specs(self) -> Any:
        return self.inner.config_specs

    def get_tuple(self, config: Any) -> Any:
        return self.inner.get_tuple(config)

    def list(
        self, config: Any, *, filter: Any = None, before: Any = None, limit: Any = None
    ) -> Iterator[Any]:
        return self.inner.list(config, filter=filter, before=before, limit=limit)

    def put(self, config: Any, checkpoint: Any, metadata: Any, new_versions: Any) -> Any:
        return self.inner.put(config, checkpoint, self._tag(checkpoint, metadata), new_versions)

    def put_writes(self, config: Any, writes: Any, task_id: str, task_path: str = "") -> None:
        return self.inner.put_writes(config, writes, task_id, task_path)

    def delete_thread(self, thread_id: str) -> None:
        return self.inner.delete_thread(thread_id)

    async def aget_tuple(self, config: Any) -> Any:
        return await self.inner.aget_tuple(config)

    async def alist(
        self, config: Any, *, filter: Any = None, before: Any = None, limit: Any = None
    ) -> AsyncIterator[Any]:
        async for item in self.inner.alist(config, filter=filter, before=before, limit=limit):
            yield item

    async def aput(self, config: Any, checkpoint: Any, metadata: Any, new_versions: Any) -> Any:
        return await self.inner.aput(config, checkpoint, self._tag(checkpoint, metadata), new_versions)

    async def aput_writes(self, config: Any, writes: Any, task_id: str, task_path: str = "") -> None:
        return await self.inner.aput_writes(config, writes, task_id, task_path)

    async def adelete_thread(self, thread_id: str) -> None:
        return await self.inner.adelete_thread(thread_id)

    def get_next_version(self, current: Any, channel: Any = None) -> Any:
        return self.inner.get_next_version(current, channel)

    # -- coherence -------------------------------------------------------------------

    def _tag(self, checkpoint: Any, metadata: Any) -> Any:
        values = checkpoint.get("channel_values", {}) if isinstance(checkpoint, dict) else {}
        try:
            score = float(self.coherence_fn(values))
        except Exception:
            score = 1.0
        tagged = dict(metadata or {})
        tagged["coherence"] = score
        tagged["coherence_t"] = self._clock()
        return tagged

    def coherence_history(self, thread_id: str) -> builtins.list[tuple[str, float, float]]:
        """``[(checkpoint_id, coherence, timestamp)]`` newest first."""
        out: builtins.list[tuple[str, float, float]] = []
        for t in self.inner.list({"configurable": {"thread_id": thread_id}}):
            md = t.metadata or {}
            if "coherence" in md:
                out.append(
                    (
                        t.config["configurable"]["checkpoint_id"],
                        float(md["coherence"]),
                        float(md.get("coherence_t", 0.0)),
                    )
                )
        return out

    def latest_coherence(self, thread_id: str) -> float | None:
        hist = self.coherence_history(thread_id)
        return hist[0][1] if hist else None

    def needs_rollback(self, thread_id: str) -> bool:
        c = self.latest_coherence(thread_id)
        return c is not None and c < self.threshold

    def best_valid_config(self, thread_id: str) -> Any:
        best = None
        best_score = float("-inf")
        now = self._clock()
        for t in self.inner.list({"configurable": {"thread_id": thread_id}}):
            md = t.metadata or {}
            c = md.get("coherence")
            if c is None or float(c) < self.threshold:
                continue
            score = float(c) - self.decay_lambda * max(0.0, now - float(md.get("coherence_t", now)))
            if score > best_score:
                best, best_score = t, score
        return best.config if best is not None else None
