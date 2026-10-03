from __future__ import annotations

import pytest

from agent_chaos.zones import ResilienceZone, ZoneSupervisor


def test_zone_isolation_restore_release(clock):
    perception = ResilienceZone("perception", ["camera"], clock=clock.now)
    inference = ResilienceZone("inference", ["model_a", "model_b"], clock=clock.now)
    inference.detector.min_consecutive = 2
    sup = ZoneSupervisor(
        [perception, inference], dangerous_actions={"brake_release"}, release_after_healthy=2, clock=clock.now
    )
    inference.checkpoint("model_a", {"w": 1}, 0.99)
    inference.checkpoint("model_b", {"w": 2}, 0.98)
    assert sup.observe("model_a", coherence=0.95).action == "none"
    clock.advance(1)
    assert sup.observe("model_a", coherence=0.2).action == "none"
    clock.advance(1)
    act = sup.observe("model_a", coherence=0.2)
    assert act.action == "isolate" and act.restored == {"model_a": {"w": 1}, "model_b": {"w": 2}}
    assert inference.isolated and not perception.isolated and inference.status.incidents == 1
    assert (
        not sup.allow("model_a", "brake_release")
        and sup.allow("camera", "brake_release")
        and sup.allow("model_a", "log")
    )
    assert sup.observe("model_a", coherence=0.2).action == "none"  # still confirmed, already isolated
    clock.advance(1)
    assert sup.observe("model_a", coherence=0.99).action == "none"
    clock.advance(1)
    assert sup.observe("model_a", coherence=0.99).action == "release"
    assert not inference.isolated and inference.status.isolated_seconds == 2.0
    assert 0 < sup.availability() < 1
    events = [r.event for r in sup.audit_log(["isolate", "restore", "release", "blocked"])]
    assert events == ["isolate", "restore", "blocked", "release"]
    sup.lockdown("attack in progress")
    assert not sup.allow("camera", "brake_release") and sup.lockdown_reason
    sup.lift_lockdown()
    assert sup.allow("camera", "brake_release")
    assert set(sup.statuses()) == {"perception", "inference"}
    with pytest.raises(KeyError):
        sup.zone_of("nobody")
    with pytest.raises(KeyError):
        perception.observe("model_a")
