# SPDX-License-Identifier: Apache-2.0
"""Resilience zones and the supervisory control layer (adversarial-resilience architecture).

A :class:`ResilienceZone` groups components (perception, inference, decision, ...) behind
a boundary with its own anomaly detector and checkpoint store. When an anomaly is
confirmed inside a zone the :class:`ZoneSupervisor` isolates *that zone only*, restores its
local checkpoint, blocks dangerous actions while it is isolated, and writes every
event to an audit log.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

from .checkpoint import CheckpointStore, SemanticCheckpoint
from .detection import AnomalyDetector, AnomalySignal


@dataclass
class AuditRecord:
    t: float
    zone: str
    event: str
    detail: dict[str, Any] = field(default_factory=dict)


@dataclass
class ZoneStatus:
    name: str
    healthy: bool = True
    isolated_since: float | None = None
    incidents: int = 0
    restorations: int = 0
    last_reason: str | None = None
    isolated_seconds: float = 0.0


class ResilienceZone:
    def __init__(
        self,
        name: str,
        components: Iterable[str],
        *,
        detector: AnomalyDetector | None = None,
        checkpoints: CheckpointStore | None = None,
        coherence_threshold: float = 0.85,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.name = name
        self.components = list(components)
        self._clock = clock
        self.detector = detector or AnomalyDetector(coherence_threshold=coherence_threshold, clock=clock)
        self.checkpoints = checkpoints or CheckpointStore(validity_threshold=coherence_threshold, clock=clock)
        self.status = ZoneStatus(name)

    def observe(
        self, component: str, task: str | None = None, output: Any = None, **kw: Any
    ) -> AnomalySignal:
        if component not in self.components:
            raise KeyError(f"{component!r} is not part of zone {self.name!r}")
        return self.detector.observe(component, task, output, **kw)

    def checkpoint(self, component: str, state: Any, coherence: float, **meta: Any) -> SemanticCheckpoint:
        return self.checkpoints.save(component, state, coherence, **meta)

    def isolate(self, reason: str) -> None:
        if self.status.healthy:
            self.status.healthy = False
            self.status.isolated_since = self._clock()
        self.status.incidents += 1
        self.status.last_reason = reason

    def restore(self) -> dict[str, Any]:
        """Restore every component to its best valid local checkpoint."""
        states: dict[str, Any] = {}
        for component in self.components:
            cp = self.checkpoints.best(component)
            if cp is not None:
                states[component] = self.checkpoints.restore(cp)
        self.status.restorations += 1
        return states

    def release(self) -> None:
        if not self.status.healthy and self.status.isolated_since is not None:
            self.status.isolated_seconds += self._clock() - self.status.isolated_since
        self.status.healthy = True
        self.status.isolated_since = None
        for component in self.components:
            self.detector.reset(component)

    @property
    def isolated(self) -> bool:
        return not self.status.healthy


@dataclass
class ZoneAction:
    zone: str
    action: str  # none | isolate | restore | release | blocked
    restored: dict[str, Any] = field(default_factory=dict)
    signal: AnomalySignal | None = None


class ZoneSupervisor:
    """Control layer: coordinates zones, enforces policy during incidents, keeps the audit log."""

    def __init__(
        self,
        zones: Iterable[ResilienceZone],
        *,
        dangerous_actions: Iterable[str] = (),
        release_after_healthy: int = 2,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.zones: dict[str, ResilienceZone] = {z.name: z for z in zones}
        self.dangerous_actions = set(dangerous_actions)
        self.release_after_healthy = max(1, release_after_healthy)
        self._clock = clock
        self.audit: list[AuditRecord] = []
        self.lockdown_reason: str | None = None
        self._healthy_streak: dict[str, int] = {}
        self._started = clock()

    # -- lookup -----------------------------------------------------------------

    def zone_of(self, component: str) -> ResilienceZone:
        for z in self.zones.values():
            if component in z.components:
                return z
        raise KeyError(f"no zone contains {component!r}")

    def _log(self, zone: str, event: str, **detail: Any) -> None:
        self.audit.append(AuditRecord(self._clock(), zone, event, detail))

    # -- observation & response ---------------------------------------------------

    def observe(self, component: str, task: str | None = None, output: Any = None, **kw: Any) -> ZoneAction:
        """Observe a component through its zone and react to confirmed anomalies."""
        zone = self.zone_of(component)
        signal = zone.observe(component, task, output, **kw)
        self._log(
            zone.name, "inference", component=component, anomalous=signal.anomalous, reasons=signal.reasons
        )
        return self.report(zone.name, signal)

    def report(self, zone_name: str, signal: AnomalySignal) -> ZoneAction:
        zone = self.zones[zone_name]
        if signal.confirmed:
            if zone.isolated:
                self._healthy_streak[zone_name] = 0
                return ZoneAction(zone_name, "none", signal=signal)
            zone.isolate("; ".join(signal.reasons) or "anomaly confirmed")
            self._log(zone_name, "isolate", reasons=signal.reasons, component=signal.agent)
            restored = zone.restore()
            self._log(zone_name, "restore", components=list(restored))
            self._healthy_streak[zone_name] = 0
            return ZoneAction(zone_name, "isolate", restored=restored, signal=signal)
        if zone.isolated and not signal.anomalous:
            streak = self._healthy_streak.get(zone_name, 0) + 1
            self._healthy_streak[zone_name] = streak
            if streak >= self.release_after_healthy:
                zone.release()
                self._log(zone_name, "release", healthy_observations=streak)
                return ZoneAction(zone_name, "release", signal=signal)
        return ZoneAction(zone_name, "none", signal=signal)

    # -- policy enforcement ---------------------------------------------------------

    def allow(self, component: str, action: str) -> bool:
        """Policy: dangerous actions are blocked during lockdown or while the zone is isolated."""
        zone = self.zone_of(component)
        blocked = action in self.dangerous_actions and (self.lockdown_reason is not None or zone.isolated)
        if blocked:
            self._log(zone.name, "blocked", component=component, action=action)
        return not blocked

    def lockdown(self, reason: str) -> None:
        self.lockdown_reason = reason
        self._log("*", "lockdown", reason=reason)

    def lift_lockdown(self) -> None:
        self.lockdown_reason = None
        self._log("*", "lockdown_lifted")

    # -- reporting -----------------------------------------------------------------

    def availability(self) -> float:
        """Share of elapsed time during which every zone was healthy (1.0 = always)."""
        elapsed = max(1e-9, self._clock() - self._started)
        down = 0.0
        for z in self.zones.values():
            down = max(
                down,
                z.status.isolated_seconds
                + ((self._clock() - z.status.isolated_since) if z.status.isolated_since else 0.0),
            )
        return max(0.0, 1.0 - down / elapsed)

    def statuses(self) -> dict[str, ZoneStatus]:
        return {n: z.status for n, z in self.zones.items()}

    def audit_log(self, events: Sequence[str] | None = None) -> list[AuditRecord]:
        if events is None:
            return list(self.audit)
        return [r for r in self.audit if r.event in events]
