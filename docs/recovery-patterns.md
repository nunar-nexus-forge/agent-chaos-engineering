# Recovery patterns in depth

## Inputs to selection

`SelfHealer.failure_context(agent)` derives a `FailureContext` from the monitors:

| field | how it is computed |
|---|---|
| `failure_duration_s` | time since the agent's current anomaly streak began |
| `degraded_agents` | the agent plus every agent currently in an anomaly streak |
| `affected_agents` | number of degraded agents (the agent plus every agent currently in an anomaly streak) |
| `error_propagation_depth` | longest chain of degraded consumers starting at the agent |
| `coherence_degradation` | drop of the latest coherence below the agent's baseline (or below the threshold) |
| `checkpoint_validity` | coherence of the best valid checkpoint (0 if none) |
| `cascading_risk` | `0.5 · downstream share + 0.5 · severity` where severity blends the coherence drop and the variance z-score (heuristic) |
| `system_can_operate_degraded` | at least `min_healthy_agents` healthy agents remain after removing the degraded ones |
| `reasoning_variance_sigma` | z-score of the agent's reasoning variance against its own history |
| `alternative_agents_available` | any healthy agent outside the degraded set |

## Selection

1. Patterns whose criteria match are candidates; if none match, all patterns are.
2. The candidate with the lowest `recovery_time + w_impact · affected + w_accuracy · (1 − accuracy_retention)` wins.
   `recovery_time` and `accuracy_retention` come from `PatternProfile` (defaults: starting
   values for each pattern's expected recovery time and accuracy retention) and can be
   recalibrated for your system; `w_impact` (default 1.0 s per affected agent) and
   `w_accuracy` (default 10 s per 100 % retention loss) encode organisational priorities.

`RecoveryAction.alternatives` lists every candidate with its cost so decisions are explainable.

## What each pattern returns

| pattern | `action.kind` | payload |
|---|---|---|
| semantic checkpointing | `restore` | `action.state` = last valid checkpoint state of the agent |
| cognitive rollback | `restore` | `action.states` = best checkpoint state per degraded agent |
| quarantine isolation | `quarantine` | `action.agents` = agents now quarantined; `healer.influence()` gives them weight 0 |
| adaptive consensus reset | `reweight` | `action.weights` = new normalised influence weights |
| (no valid checkpoint) | `none` | `action.success = False` |

The application is up to you: write the restored state back into your agent, skip
quarantined agents when aggregating, use the weights in your consensus step.

## Resolution and latency

An incident opens when `recover()` is called and resolves at the next non-anomalous
observation of the agent (`Incident.latency_s`). While an incident is open, `heal()` applies
no further recovery to that agent even if later outputs are still anomalous (one recovery per
incident); call `recover(agent)` to apply another pattern explicitly, or
`resolve(agent, success=False)` to close the incident. Quarantined agents are released after
`probation_observations` consecutive healthy observations.
