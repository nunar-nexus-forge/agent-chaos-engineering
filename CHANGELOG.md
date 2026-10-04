# Changelog

All notable changes to `agent-chaos-engineering` are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.1.0] - 2026-10-03

First release on PyPI: the initial version of 27 September 2026 (*Added*) together with
the changes made to it before publishing.

### Added
- Fault taxonomy: network delay, tool error, hallucination loop, cyclic dependency, stale memory, miscoordination, input perturbation, data poisoning, perception spoofing.
- `ChaosPlan` with round-robin / all schedules, forced primary injections, JSON/YAML persistence and three presets (operational faults, adversarial faults, everything).
- `ChaosInjector` with tool, reasoning, message, memory and input surfaces (sync + async), deterministic seeding, virtual clocks and full injection logs.
- Detection: reasoning-variance and semantic-coherence monitors with temporal filtering; redundant-path discrepancy detector.
- `CheckpointStore` (semantic checkpoints, restoration objective) and the four recovery patterns with their matching criteria and cost-based selection.
- `SelfHealer` orchestrator: dependency graph, failure context, incidents, quarantine with probation, consensus weights, reporting.
- Resilience zones, `ZoneSupervisor` control layer with audit log and lockdown, `TrustWeightedDecision` fusion with cross-path consistency.
- `ScenarioRunner` (baseline / chaos / healed arms with identical seeds) and the resilience `Scorecard` (accuracy retention, recovery latency, CAFR, per-fault success, decision integrity, availability, overhead, RLI/CAFRI/ARI/composite).
- Built-in simulated pipeline scenario, the `agent-chaos` CLI, LangGraph (`chaos_node`, `heal_node`, `SelfHealingCheckpointSaver`) and LangChain (`chaos_tool`) adapters, and a bridge to `multi-agent-observability` (MA-Trace).

### Changed
- The operational-fault preset and its constructor are now `standard` / `ChaosPlan.standard()`;
  the bundled plan file is `examples/plans/standard.json`.
- README, `NOTICE`, `CITATION.cff` and the package metadata now describe the software only;
  the README no longer lists CrewAI among the supported frameworks (there is no CrewAI adapter).

### Fixed
- CI caches uv by `pyproject.toml` (the lock file is not committed), which newer `setup-uv` releases require.
- README install line named the companion package `ma-trace`; the distribution is `multi-agent-observability`.
- `docs/recovery-patterns.md` now describes `affected_agents` as the number of degraded agents (the
  code never restricted it to downstream agents).
