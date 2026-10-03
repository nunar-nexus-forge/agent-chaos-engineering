from __future__ import annotations

import pytest

from agent_chaos.checkpoint import CheckpointStore
from agent_chaos.consensus import (
    ConsensusWeights,
    TrustWeightedDecision,
    cross_path_consistency,
    weighted_consensus,
    weighted_vote,
)


def test_checkpoint_store_latest_valid_and_best(clock):
    store = CheckpointStore(capacity_per_agent=3, validity_threshold=0.85, decay_lambda=0.01, clock=clock.now)
    a = store.save("x", {"v": 1}, 0.95, step=1)
    assert store.latest("x") is a
    clock.advance(10)
    b = store.save("x", {"v": 2}, 0.90, step=2)
    clock.advance(10)
    c = store.save("x", {"v": 3}, 0.50, step=3, label="bad")
    assert store.latest("x") is c and store.latest_valid("x") is b
    # best: a -> 0.95 - 0.01*20 = 0.75 ; b -> 0.90 - 0.01*10 = 0.80  => b
    assert store.best("x") is b and store.validity("x") == 0.90
    assert store.best("x", valid_only=False) is b
    state = store.restore(b)
    state["v"] = 99
    assert b.state == {"v": 2} and store.restorations == 1
    store.save("x", {"v": 4}, 0.99, step=4)  # capacity 3 evicts a
    assert [cp.step for cp in store.history("x")] == [2, 3, 4] and len(store) == 3
    assert store.prune("x", keep_last=1) == 2 and store.history("x")[0].step == 4
    assert store.best("nobody") is None and store.validity("nobody") == 0.0 and store.agents() == ["x"]
    assert store.prune("nobody") == 0


def test_consensus_weights_update():
    w = ConsensusWeights(["a", "b"], alpha=0.1, threshold=0.85)
    assert w.weights == {"a": 0.5, "b": 0.5}
    w.update({"a": 0.95, "b": 0.5})
    assert w["a"] > w["b"] and abs(sum(w.weights.values()) - 1.0) < 1e-9 and w.updates == 1
    w.ensure("c")
    assert "c" in w.weights
    raw = ConsensusWeights(["a"], normalise=False, min_weight=0.5)
    raw.update({"a": 0.0})
    assert raw["a"] == max(0.5, 1.0 * (1 + 0.1 * (0 - 0.85)))
    w.reset()
    assert w.influence()["a"] == pytest.approx(1 / 3)


def test_weighted_consensus_and_vote():
    assert weighted_consensus({"a": 1.0, "b": 3.0}, {"a": 1, "b": 1}) == 2.0
    assert weighted_consensus({"a": [1.0, 0.0], "b": [3.0, 2.0]}, {"a": 3, "b": 1}) == [1.5, 0.5]
    with pytest.raises(ValueError):
        weighted_consensus({"a": "x"}, {"a": 1})
    assert weighted_vote({"a": "yes", "b": "no", "c": "yes"}, {"a": 1, "b": 5, "c": 1}) == ("no", 5 / 7)
    with pytest.raises(ValueError):
        weighted_vote({}, {})


def test_trust_weighted_decision_suppresses_outlier():
    fusion = TrustWeightedDecision(["p1", "p2", "p3"], discrepancy_threshold=0.5)
    ok = fusion.decide({"p1": 0.80, "p2": 0.82, "p3": 0.79})
    assert ok.consistent and not ok.suppressed and abs(ok.value - 0.8033) < 1e-3
    attacked = fusion.decide({"p1": 0.80, "p2": 0.82, "p3": 0.05})
    assert (
        attacked.flagged_adversarial and attacked.suppressed == ["p3"] and abs(attacked.value - 0.81) < 1e-9
    )
    fusion.feedback(0.81, {"p1": 0.80, "p2": 0.82, "p3": 0.05})
    assert fusion.trust.weights["p3"] < fusion.trust.weights["p1"]
    votes = TrustWeightedDecision(["a", "b"]).decide({"a": "fraud", "b": "fraud"})
    assert votes.value == "fraud"
    with pytest.raises(ValueError):
        fusion.decide({})


def test_cross_path_consistency():
    assert cross_path_consistency([0.5, 0.6])
    assert not cross_path_consistency([0.95, 0.05])
    assert cross_path_consistency([0.9])
