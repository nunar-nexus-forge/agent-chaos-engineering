# SPDX-License-Identifier: Apache-2.0
"""Record injections and recoveries into the current `MA-Trace <https://pypi.org/project/multi-agent-observability/>`_
episode (requires ``pip install multi-agent-observability``)."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from ..healer import SelfHealer
from ..injector import ChaosInjector


def _chain(previous: Callable[[Any], None] | None, new: Callable[[Any], None]) -> Callable[[Any], None]:
    def both(item: Any) -> None:
        if previous is not None:
            previous(item)
        new(item)

    return both


def bind_ma_trace(injector: ChaosInjector | None = None, healer: SelfHealer | None = None) -> None:
    """Log every injection (warning) and recovery action (info) as ma-trace log events."""
    try:
        import ma_trace as mt
    except ImportError as e:  # pragma: no cover
        raise ImportError("bind_ma_trace needs ma-trace: pip install multi-agent-observability") from e

    if injector is not None:

        def on_injection(rec: Any) -> None:
            data = {k: v for k, v in rec.to_dict().items() if k != "agent"}
            mt.log(
                f"chaos: {rec.fault.value} on {rec.surface.value}", level="warning", agent=rec.agent, **data
            )

        injector.on_injection = _chain(injector.on_injection, on_injection)
    if healer is not None:

        def on_recovery(action: Any) -> None:
            agent = action.agents[0] if action.agents else None
            mt.log(
                f"recovery: {action.describe()}",
                level="info",
                agent=agent,
                pattern=action.pattern,
                kind=action.kind,
            )

        healer.on_recovery = _chain(healer.on_recovery, on_recovery)
