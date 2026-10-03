from __future__ import annotations

import pytest

from agent_chaos.faults import FaultKind, ToolError
from agent_chaos.plan import PRESETS, ChaosPlan, ScheduleMode


def test_presets_and_schedule():
    plan = ChaosPlan.standard(seed=3)
    assert [f.kind for f in plan.faults] == [
        FaultKind.NETWORK_DELAY,
        FaultKind.TOOL_ERROR,
        FaultKind.HALLUCINATION_LOOP,
        FaultKind.CYCLIC_DEPENDENCY,
    ]
    assert plan.mode is ScheduleMode.ROUND_ROBIN and plan.seed == 3
    assert plan.active_faults(1) == [plan.faults[1]] and plan.primary_fault(5) is plan.faults[1]
    all_mode = ChaosPlan.standard(mode=ScheduleMode.ALL)
    assert all_mode.active_faults(7) == all_mode.faults
    assert ChaosPlan().active_faults(0) == [] and ChaosPlan().primary_fault(0) is None
    assert set(PRESETS) == {"standard", "adversarial", "full"}
    assert len(ChaosPlan.full().faults) == 9
    with pytest.raises(KeyError):
        ChaosPlan.preset("nope")
    assert "network_delay" in plan.describe()
    assert list(iter(plan)) == plan.faults


def test_json_roundtrip(tmp_path):
    plan = ChaosPlan(
        [ToolError(0.5, agents=["x"])],
        name="mine",
        seed=9,
        mode=ScheduleMode.ALL,
        force_primary=False,
        expected_steps=4,
        max_injections_per_episode=2,
    )
    path = tmp_path / "plan.json"
    plan.save(path)
    back = ChaosPlan.load(path)
    assert back.name == "mine" and back.seed == 9 and back.mode is ScheduleMode.ALL and not back.force_primary
    assert back.expected_steps == 4 and back.max_injections_per_episode == 2
    assert back.faults[0].agents == {"x"} and back.faults[0].probability == 0.5


def test_yaml_roundtrip(tmp_path):
    pytest.importorskip("yaml")
    import yaml

    plan = ChaosPlan.adversarial(seed=1)
    path = tmp_path / "plan.yaml"
    path.write_text(yaml.safe_dump(plan.to_dict()))
    assert [f.kind for f in ChaosPlan.load(path).faults] == [f.kind for f in plan.faults]
