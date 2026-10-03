# SPDX-License-Identifier: Apache-2.0
"""A small, deterministic multi-agent pipeline used by the demo, the CLI default scenario
and the tests. It is a *simulation harness*, not a benchmark: five cooperating agents track
a drifting hidden quantity, faults corrupt their sensing/reasoning/messaging, and the
self-healer (when enabled) checkpoints, detects and recovers.

Topology::

    sensor --> analyst_a --\\
                            aggregator --> executor
    sensor --> analyst_b --/
"""

from __future__ import annotations

from typing import Any

from .runner import ScenarioContext, ScenarioOutcome

AGENTS: tuple[str, ...] = ("sensor", "analyst_a", "analyst_b", "aggregator", "executor")
DEPENDENCIES: dict[str, tuple[str, ...]] = {
    "analyst_a": ("sensor",),
    "analyst_b": ("sensor",),
    "aggregator": ("analyst_a", "analyst_b"),
    "executor": ("aggregator",),
}
ANALYSTS = ("analyst_a", "analyst_b")


def pipeline_scenario(ctx: ScenarioContext) -> ScenarioOutcome:
    p = ctx.params
    steps = int(p.get("steps", 8))
    tolerance = float(p.get("tolerance", 0.15))
    drift = float(p.get("drift", 0.03))
    noise = float(p.get("noise", 0.02))
    success_threshold = float(p.get("success_threshold", 0.75))

    rng, chaos, healer, clock = ctx.rng, ctx.chaos, ctx.healer, ctx.clock
    truth0 = rng.uniform(20.0, 100.0)
    memory = {a: chaos.memory({}, agent=a) for a in ANALYSTS}
    step_correct: list[bool] = []
    step_times: list[float] = []
    failed: set[str] = set()
    propagated = 0
    decisions = correct_decisions = available = 0
    last_reading: float | None = None
    checkpoints_taken = restores = 0

    @chaos.tool("read_sensor", agent="sensor")
    def read_sensor(truth: float) -> float:
        return truth * (1.0 + rng.gauss(0.0, noise))

    for t in range(steps):
        chaos.tick(t)
        truth_t = truth0 * (1.0 + drift * t)
        step_times.append(clock.now())

        # --- perception: the sensor reads the world (tool surface) ---------------------
        try:
            reading: float | None = float(read_sensor(truth_t))
            last_reading = reading
        except Exception:
            reading = last_reading  # tool error -> fall back to the last good reading
            failed.add("sensor")
        if reading is None:
            step_correct.append(False)
            clock.advance(ctx.step_seconds)
            continue

        # --- messaging: sensor -> analysts (message surface) ----------------------------
        inputs: dict[str, float] = {}
        for a in ANALYSTS:
            d = chaos.message("sensor", a, reading)
            if d.delivered and d.target in ANALYSTS:
                inputs.setdefault(
                    d.target, float(d.content) if isinstance(d.content, (int, float)) else reading
                )

        # --- reasoning: analysts estimate (reasoning + memory surfaces) -----------------
        estimates: dict[str, float] = {}
        wrong_analysts: set[str] = set()
        for a in ANALYSTS:
            x = inputs.get(a)
            prev = memory[a].get("estimate")
            if x is None and prev is None:
                continue
            raw = x if prev is None else (prev if x is None else 0.7 * x + 0.3 * prev)
            est = chaos.reason(a, raw)
            if not isinstance(est, (int, float)):
                est = raw
            est = float(est)
            if healer is not None:
                constraints = [
                    0.0 < est < 500.0,
                    (abs(est - x) / max(x, 1e-9) < 0.25) if x is not None else True,
                    (abs(est - prev) / max(abs(prev), 1e-9) < 0.5) if prev is not None else True,
                ]
                accuracy = max(0.0, 1.0 - abs(est - x) / max(x, 1e-9)) if x is not None else 0.5
                signal, action = healer.heal(
                    a, task=f"step{t}", output=est, constraints=constraints, accuracy=accuracy
                )
                if action is not None and action.kind == "restore" and action.state is not None:
                    est = float(action.state["estimate"])
                    restores += 1
                elif not signal.anomalous:
                    healer.checkpoint(
                        a, {"estimate": est}, coherence=sum(constraints) / len(constraints), step=t
                    )
                    checkpoints_taken += 1
            memory[a]["estimate"] = est
            estimates[a] = est
            if abs(est - truth_t) / truth_t >= tolerance:
                wrong_analysts.add(a)
                failed.add(a)

        # --- aggregation: trust/influence-weighted consensus ----------------------------
        quarantined = set(healer.quarantined) if healer is not None else set()
        usable: dict[str, float] = {}
        for a, est in estimates.items():
            if a in quarantined:
                continue
            d = chaos.message(a, "aggregator", est)
            if d.delivered and d.target == "aggregator":
                usable[a] = float(d.content) if isinstance(d.content, (int, float)) else est
        if not usable:
            step_correct.append(False)
            clock.advance(ctx.step_seconds)
            continue
        weights = healer.influence() if healer is not None else dict.fromkeys(usable, 1.0)
        total = sum(weights.get(a, 0.0) for a in usable) or 1.0
        aggregate = sum(est * weights.get(a, 0.0) for a, est in usable.items()) / total

        # --- decision: executor acts on the aggregate (message surface) -----------------
        d = chaos.message("aggregator", "executor", aggregate)
        if not d.delivered or d.target != "executor":
            step_correct.append(False)
            clock.advance(ctx.step_seconds)
            continue
        decision = float(d.content) if isinstance(d.content, (int, float)) else aggregate
        available += 1
        decisions += 1
        ok = abs(decision - truth_t) / truth_t < tolerance
        correct_decisions += int(ok)
        step_correct.append(ok)
        if not ok:
            failed.update({"aggregator", "executor"})
            if wrong_analysts and len(wrong_analysts) < len(ANALYSTS):
                propagated += 2  # aggregator and executor failed because an upstream analyst did
        clock.advance(ctx.step_seconds)

    correct = decisions > 0 and (correct_decisions / decisions) >= success_threshold
    return ScenarioOutcome(
        correct=correct,
        steps=steps,
        step_correct=step_correct,
        step_times=step_times,
        failed_agents=sorted(failed),
        propagated_failures=propagated,
        decisions=decisions,
        correct_decisions=correct_decisions,
        available_steps=available,
        extra={"checkpoints": checkpoints_taken, "restores": restores, "truth0": round(truth0, 3)},
    )


pipeline_scenario.agents = AGENTS  # type: ignore[attr-defined]
pipeline_scenario.dependencies = DEPENDENCIES  # type: ignore[attr-defined]


def describe() -> dict[str, Any]:
    return {
        "agents": list(AGENTS),
        "dependencies": {k: list(v) for k, v in DEPENDENCIES.items()},
        "doc": __doc__,
    }
