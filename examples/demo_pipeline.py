# SPDX-License-Identifier: Apache-2.0
"""Run the bundled five-agent pipeline under baseline / chaos / healed arms and print the
resilience scorecard. No API keys, no network - everything is a seeded simulation.

    python examples/demo_pipeline.py [--episodes 30] [--preset standard|adversarial|full]
"""

from __future__ import annotations

import argparse

from agent_chaos import ChaosPlan, HealingConfig, ScenarioRunner
from agent_chaos.sim import pipeline_scenario


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=30)
    ap.add_argument("--preset", default="standard", choices=["standard", "adversarial", "full"])
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    plan = ChaosPlan.preset(args.preset, seed=args.seed)
    print(plan.describe(), "\n")
    runner = ScenarioRunner(
        pipeline_scenario,
        plan=plan,
        episodes=args.episodes,
        seed=args.seed,
        config=HealingConfig(min_consecutive=1),
        params={"steps": 8, "success_threshold": 0.75},
    )
    result = runner.run()
    print(result.scorecard.to_markdown())
    healed = [r for r in result.results if r.arm == "healed"]
    incidents = [i for r in healed for i in r.incidents]
    if incidents:
        print(f"\nfirst healed incident: {incidents[0]}")
    result.save("chaos-report.json")
    print(
        "\nfull per-episode results written to chaos-report.json (re-render with: agent-chaos report chaos-report.json)"
    )


if __name__ == "__main__":
    main()
