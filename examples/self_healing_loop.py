# SPDX-License-Identifier: Apache-2.0
"""Minimal self-healing loop around a hand-written "agent" that occasionally hallucinates."""

from __future__ import annotations

import random

from agent_chaos import HealingConfig, SelfHealer

rng = random.Random(3)
healer = SelfHealer(["analyst"], HealingConfig(min_consecutive=1))
state = {"estimate": 50.0}


def analyst_step(state: dict, observation: float) -> dict:
    if rng.random() < 0.15:  # a hallucination: wildly off
        return {"estimate": observation * 9}
    return {"estimate": 0.7 * observation + 0.3 * state["estimate"]}


for t in range(12):
    observation = 50.0 + t
    proposal = analyst_step(state, observation)
    constraints = [abs(proposal["estimate"] - observation) / observation < 0.25]
    signal, action = healer.heal(
        "analyst", task=f"t{t}", output=proposal["estimate"], constraints=constraints
    )
    if action is not None and action.kind == "restore":
        proposal = action.state
        print(f"t={t:2d} anomaly -> {action.describe()}")
    elif signal.anomalous:
        # one recovery per incident: while the incident opened above is still open, heal() does
        # not recover again; it resolves at the next healthy output (see docs/recovery-patterns.md)
        incident = healer.open_incident("analyst")
        why = f"incident #{incident.id} still open" if incident else "no valid checkpoint"
        print(f"t={t:2d} anomaly ({why}; waiting for a healthy output)")
    else:
        healer.checkpoint("analyst", proposal, coherence=1.0)
    state = proposal
    print(f"t={t:2d} estimate={state['estimate']:.1f}")

print(healer.report())
