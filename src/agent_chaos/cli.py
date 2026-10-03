# SPDX-License-Identifier: Apache-2.0
"""``agent-chaos`` command line interface (distribution ``agent-chaos-engineering``)."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from typing import Any

from ._version import __version__
from .faults import describe_faults
from .plan import PRESETS, ChaosPlan
from .runner import RunResult, ScenarioRunner
from .scorecard import ARMS


def _resolve(spec: str) -> Any:
    if ":" not in spec:
        raise SystemExit(f"--scenario must look like 'package.module:function', got {spec!r}")
    module_name, _, attr = spec.partition(":")
    module = importlib.import_module(module_name)
    try:
        return getattr(module, attr)
    except AttributeError as e:
        raise SystemExit(f"{spec!r} not found") from e


def _parse_params(items: list[str] | None) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for item in items or []:
        if "=" not in item:
            raise SystemExit(f"--param expects key=value, got {item!r}")
        k, _, v = item.partition("=")
        try:
            out[k] = json.loads(v)
        except json.JSONDecodeError:
            out[k] = v
    return out


def cmd_faults(args: argparse.Namespace) -> int:
    rows = describe_faults()
    if args.json:
        print(json.dumps(rows, indent=2, default=str))
        return 0
    for r in rows:
        print(f"{r['kind']:<22} surfaces={','.join(r['surfaces']):<24} p={r['default_probability']}")
        print(f"    {r['description']}")
        if r["defaults"]:
            print(f"    defaults: {r['defaults']}")
    return 0


def cmd_presets(args: argparse.Namespace) -> int:
    if args.json:
        print(json.dumps({name: fn().to_dict() for name, fn in PRESETS.items()}, indent=2, default=str))
        return 0
    for name, fn in PRESETS.items():
        print(fn().describe().replace(f"plan {name!r}", f"preset {name!r}"))
        print()
    return 0


def cmd_plan(args: argparse.Namespace) -> int:
    plan = ChaosPlan.preset(args.preset, seed=args.seed)
    text = plan.to_json()
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write(text)
        print(f"wrote {args.out}", file=sys.stderr)
    else:
        print(text)
    return 0


def _load_plan(args: argparse.Namespace) -> ChaosPlan:
    if args.plan:
        return ChaosPlan.load(args.plan)
    return ChaosPlan.preset(args.preset, seed=args.seed)


def cmd_run(args: argparse.Namespace) -> int:
    sys.path.insert(0, os.getcwd())
    scenario = _resolve(args.scenario)
    plan = _load_plan(args)
    arms = tuple(a.strip() for a in args.arms.split(",") if a.strip())
    for arm in arms:
        if arm not in ARMS:
            raise SystemExit(f"unknown arm {arm!r}; choose from {', '.join(ARMS)}")
    runner = ScenarioRunner(
        scenario, plan=plan, episodes=args.episodes, seed=args.seed, params=_parse_params(args.param)
    )
    result = runner.run(arms)
    if args.json:
        result.save(args.json)
        print(f"wrote {args.json}", file=sys.stderr)
    if args.format == "json":
        print(result.scorecard.to_json())
    else:
        print(
            f"scenario: {result.scenario}   plan: {plan.name}   "
            f"episodes/arm: {args.episodes}   seed: {args.seed}\n"
        )
        print(result.scorecard.to_markdown())
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    result = RunResult.load(args.file)
    if args.format == "json":
        print(result.scorecard.to_json())
    else:
        print(f"scenario: {result.scenario}   episodes/arm: {result.episodes}   seed: {result.seed}\n")
        print(result.scorecard.to_markdown())
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="agent-chaos", description="Chaos engineering and self-healing for multi-agent systems."
    )
    p.add_argument("--version", action="version", version=f"agent-chaos {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("faults", help="list the fault taxonomy")
    s.add_argument("--json", action="store_true")
    s.set_defaults(func=cmd_faults)

    s = sub.add_parser("presets", help="describe the built-in chaos plans")
    s.add_argument("--json", action="store_true")
    s.set_defaults(func=cmd_presets)

    s = sub.add_parser("plan", help="export a preset as an editable JSON plan")
    s.add_argument("--preset", choices=list(PRESETS), default="standard")
    s.add_argument("--seed", type=int, default=None)
    s.add_argument("-o", "--out")
    s.set_defaults(func=cmd_plan)

    s = sub.add_parser(
        "run", help="run a scenario under baseline / chaos / healed arms and print the scorecard"
    )
    s.add_argument(
        "--scenario",
        default="agent_chaos.sim:pipeline_scenario",
        help="'package.module:function' taking a ScenarioContext",
    )
    s.add_argument("--preset", choices=list(PRESETS), default="standard")
    s.add_argument("--plan", help="JSON (or YAML) plan file; overrides --preset")
    s.add_argument("--episodes", type=int, default=30)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--arms", default=",".join(ARMS))
    s.add_argument("--param", action="append", help="scenario parameter key=value (JSON values accepted)")
    s.add_argument("--json", help="write the full results (per-episode) to this file")
    s.add_argument("--format", choices=["markdown", "json"], default="markdown")
    s.set_defaults(func=cmd_run)

    s = sub.add_parser("report", help="re-render a saved results file")
    s.add_argument("file")
    s.add_argument("--format", choices=["markdown", "json"], default="markdown")
    s.set_defaults(func=cmd_report)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
