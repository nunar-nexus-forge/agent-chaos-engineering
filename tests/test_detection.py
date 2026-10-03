from __future__ import annotations

import math

from agent_chaos.detection import (
    AnomalyDetector,
    PathDiscrepancyDetector,
    ReasoningVarianceMonitor,
    as_vector,
    default_distance,
)


def test_vector_helpers():
    assert (
        as_vector(3) == [3.0]
        and as_vector([1, 2.5]) == [1.0, 2.5]
        and as_vector("x") is None
        and as_vector(True) is None
    )
    assert (
        default_distance(1.0, 3.0) == 2.0
        and default_distance("a", "a") == 0.0
        and default_distance("a", "b") == 1.0
    )
    assert default_distance([1, 2], [2, 4]) == 1.5


def test_reasoning_variance_uses_peer_mean():
    m = ReasoningVarianceMonitor(sigma=1.5, history=10)
    m.record("a", "t1", 10.0)
    m.record("b", "t1", 12.0)
    m.record("c", "t1", 14.0)
    assert m.variance("a") == 3.0  # |10 - mean(12,14)|
    assert m.variance("b") == 0.0  # |12 - mean(10,14)|
    assert m.peers("a", "t1") == {"b": 12.0, "c": 14.0}
    m.record("solo", "t2", 1.0)
    assert m.variance("solo") is None
    m.record("x", "t3", "yes")
    m.record("y", "t3", "no")
    assert m.variance("x") == 1.0
    for v in (1.0, 1.0, 1.0, 1.0):
        m.update_history("a", v)
    assert m.threshold("a") == 1.0 and m.z_score("a", 1.0) == 0.0 and math.isinf(m.z_score("a", 2.0))
    assert ReasoningVarianceMonitor().threshold("nobody") is None


def test_detector_coherence_and_temporal_filter(clock):
    d = AnomalyDetector(coherence_threshold=0.85, min_consecutive=2, clock=clock.now)
    s1 = d.observe("a", coherence=0.95)
    assert not s1.anomalous and s1.step == 0 and d.consecutive("a") == 0
    s2 = d.observe("a", constraints=[True, False, False])
    assert s2.anomalous and not s2.confirmed and s2.consecutive == 1 and s2.coherence == 1 / 3
    assert s2.coherence_degradation > 0.5 and "coherence" in s2.reasons[0]
    clock.advance(2.0)
    s3 = d.observe("a", coherence=0.5)
    assert s3.confirmed and s3.consecutive == 2 and d.streak_duration("a") == 2.0
    assert "CONFIRMED" in str(s3) and d.anomalous_agents() == ["a"] and d.last("a") is s3
    s4 = d.observe("a", coherence=0.99)
    assert not s4.anomalous and d.consecutive("a") == 0 and d.streak_duration("a") == 0.0
    d.reset("a")
    assert d.observe("a", constraints={}).coherence == 1.0
    assert d.observe("b", constraints={"x": True, "y": False}).coherence == 0.5


def test_detector_variance_after_warmup():
    d = AnomalyDetector(variance_sigma=1.5, min_consecutive=1, warmup=3)
    for step in range(5):  # stable phase: a and b agree
        d.observe("a", task=f"t{step}", output=10.0 + step)
        d.observe("b", task=f"t{step}", output=10.0 + step + 0.1)
    d.observe("b", task="t9", output=50.0)
    sig = d.observe("a", task="t9", output=10.0)
    assert (
        sig.anomalous
        and sig.variance is not None
        and sig.variance_threshold is not None
        and sig.z_score > 1.5
    )
    assert "reasoning variance" in sig.reasons[0]


def test_path_discrepancy():
    p = PathDiscrepancyDetector(threshold=0.2)
    assert p.discrepancy([1.0]) == 0.0
    d, flag = p.check([10.0, 10.5])
    assert d < 0.2 and not flag
    d2, flag2 = p.check([0.95, 0.05])
    assert flag2 and d2 > 1.0
    assert p.check(["a", "a"]) == (0.0, False) and p.check(["a", "b"]) == (1.0, True)
    assert p.discrepancy([[1.0, 1.0], [1.0, 3.0]]) > 0
