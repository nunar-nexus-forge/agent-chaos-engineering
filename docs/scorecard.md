# The resilience scorecard

`ScenarioRunner` runs one scenario function under three arms with identical per-episode
seeds, so the chaos and healed arms see the same injection decisions:

| arm | chaos | self-healer |
|---|---|---|
| `baseline` | off | none |
| `chaos` | on | none |
| `healed` | on | `SelfHealer` |

## Per-arm metrics

| metric | definition |
|---|---|
| task success (accuracy) | share of episodes the scenario reports as `correct` |
| accuracy retention | arm accuracy / baseline accuracy |
| recovery latency | mean time from an injection to the next correct decision step (virtual clock); `n/a` when nothing recovered |
| recovery rate | recovered injections / all injections |
| cross-agent failure rate (CAFR) | propagated failures / (agents × steps), averaged over episodes |
| decision integrity | correct decisions / decisions (all steps, under attack) |
| system availability | steps with a valid output / steps |
| recovery success by fault | share of correct episodes among those whose *primary* fault was that kind |
| compute overhead | (healed cpu time − chaos cpu time) / chaos cpu time |

## Improvement metrics (healed vs chaos)

* **RLI** = (RL_chaos − RL_healed) / RL_chaos
* **CAFRI** = (CAFR_chaos − CAFR_healed) / CAFR_chaos
* **ARI** = (AR_healed − AR_chaos) / AR_chaos, with AR = accuracy retention
* **composite reliability** = weighted mean of the three (weights default to ⅓ each)

## Writing a scenario

```python
from agent_chaos import ScenarioContext, ScenarioOutcome


def my_scenario(ctx: ScenarioContext) -> ScenarioOutcome:
    chaos, healer, rng, clock = ctx.chaos, ctx.healer, ctx.rng, ctx.clock
    step_ok, times = [], []
    for t in range(10):
        chaos.tick(t)
        times.append(clock.now())
        ...  # run one step through chaos.tool / .message / .reason
        if healer:
            ...  # checkpoint / heal
        step_ok.append(decision_is_correct)
        clock.advance(ctx.step_seconds)
    return ScenarioOutcome(
        correct=sum(step_ok) / len(step_ok) >= 0.75,
        steps=10,
        step_correct=step_ok,
        step_times=times,
        decisions=10,
        correct_decisions=sum(step_ok),
    )


my_scenario.agents = ("a", "b")  # optional: lets the runner build the healer
my_scenario.dependencies = {"b": ["a"]}
```

`step_correct`/`step_times` drive the recovery-latency metric; `propagated_failures` and
`failed_agents` drive CAFR; `available_steps` drives availability.
