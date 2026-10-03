# SPDX-License-Identifier: Apache-2.0
"""agent-chaos: chaos engineering and self-healing recovery patterns for multi-agent systems.

Break things::

    from agent_chaos import ChaosInjector, ChaosPlan
    chaos = ChaosInjector(ChaosPlan.standard(seed=1), agents=["planner", "executor"])

    @chaos.tool("search", agent="executor")          # 5 % of calls raise, some are delayed
    def search(q): ...

    @chaos.reasoning_step("planner")                  # 3 % of outputs become hallucinations
    def plan(goal): ...

    delivery = chaos.message("planner", "executor", plan)   # may be dropped / rerouted / delayed

Heal them::

    from agent_chaos import SelfHealer
    healer = SelfHealer(agents=["planner", "executor"], dependencies={"executor": ["planner"]})
    healer.checkpoint("planner", state, coherence=1.0)
    signal, action = healer.heal("planner", task="t1", output=answer, constraints=[...])

Measure::

    from agent_chaos import ScenarioRunner
    runner = ScenarioRunner(my_scenario, plan=ChaosPlan.standard(), episodes=30)
    print(runner.run().scorecard.to_markdown())
"""

from __future__ import annotations

from ._version import __version__
from .checkpoint import CheckpointStore, SemanticCheckpoint
from .consensus import (
    ConsensusWeights,
    FusedDecision,
    TrustWeightedDecision,
    cross_path_consistency,
    weighted_consensus,
    weighted_vote,
)
from .detection import (
    AnomalyDetector,
    AnomalySignal,
    CoherenceMonitor,
    PathDiscrepancyDetector,
    ReasoningVarianceMonitor,
)
from .faults import (
    FAULT_TYPES,
    ChaosToolError,
    CyclicDependency,
    DataPoisoning,
    Fault,
    FaultContext,
    FaultKind,
    HallucinationLoop,
    InjectionRecord,
    InputPerturbation,
    Miscoordination,
    NetworkDelay,
    Outcome,
    PerceptionSpoofing,
    StaleMemory,
    Surface,
    ToolError,
    describe_faults,
)
from .healer import HealingConfig, Incident, SelfHealer
from .injector import ChaosInjector, ChaosMemory, Delivery
from .patterns import (
    DEFAULT_PATTERNS,
    AdaptiveConsensusReset,
    CognitiveRollback,
    FailureContext,
    PatternProfile,
    PatternSelector,
    QuarantineIsolation,
    RecoveryAction,
    RecoveryPattern,
    SemanticCheckpointing,
)
from .plan import PRESETS, ChaosPlan, ScheduleMode
from .runner import RunResult, ScenarioContext, ScenarioOutcome, ScenarioRunner, VirtualClock
from .scorecard import ARMS, ArmSummary, EpisodeResult, Scorecard, build_scorecard, summarise_arm
from .zones import AuditRecord, ResilienceZone, ZoneAction, ZoneStatus, ZoneSupervisor

__all__ = [
    "ARMS",
    "DEFAULT_PATTERNS",
    "FAULT_TYPES",
    "PRESETS",
    "AdaptiveConsensusReset",
    "AnomalyDetector",
    "AnomalySignal",
    "ArmSummary",
    "AuditRecord",
    "ChaosInjector",
    "ChaosMemory",
    "ChaosPlan",
    "ChaosToolError",
    "CheckpointStore",
    "CognitiveRollback",
    "CoherenceMonitor",
    "ConsensusWeights",
    "CyclicDependency",
    "DataPoisoning",
    "Delivery",
    "EpisodeResult",
    "FailureContext",
    "Fault",
    "FaultContext",
    "FaultKind",
    "FusedDecision",
    "HallucinationLoop",
    "HealingConfig",
    "Incident",
    "InjectionRecord",
    "InputPerturbation",
    "Miscoordination",
    "NetworkDelay",
    "Outcome",
    "PathDiscrepancyDetector",
    "PatternProfile",
    "PatternSelector",
    "PerceptionSpoofing",
    "QuarantineIsolation",
    "ReasoningVarianceMonitor",
    "RecoveryAction",
    "RecoveryPattern",
    "ResilienceZone",
    "RunResult",
    "ScenarioContext",
    "ScenarioOutcome",
    "ScenarioRunner",
    "ScheduleMode",
    "Scorecard",
    "SelfHealer",
    "SemanticCheckpoint",
    "SemanticCheckpointing",
    "StaleMemory",
    "Surface",
    "ToolError",
    "TrustWeightedDecision",
    "VirtualClock",
    "ZoneAction",
    "ZoneStatus",
    "ZoneSupervisor",
    "__version__",
    "build_scorecard",
    "cross_path_consistency",
    "describe_faults",
    "summarise_arm",
    "weighted_consensus",
    "weighted_vote",
]
