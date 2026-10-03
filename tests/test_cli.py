from __future__ import annotations

import json

from agent_chaos.cli import main


def test_faults_and_presets(capsys):
    assert main(["faults"]) == 0
    assert "hallucination_loop" in capsys.readouterr().out
    assert main(["faults", "--json"]) == 0
    assert len(json.loads(capsys.readouterr().out)) == 9
    assert main(["presets"]) == 0
    assert "preset 'standard'" in capsys.readouterr().out
    assert main(["presets", "--json"]) == 0
    assert set(json.loads(capsys.readouterr().out)) == {"standard", "adversarial", "full"}


def test_plan_export_and_run(tmp_path, capsys):
    plan_file = tmp_path / "plan.json"
    assert main(["plan", "--preset", "adversarial", "--seed", "4", "-o", str(plan_file)]) == 0
    assert json.loads(plan_file.read_text())["seed"] == 4
    capsys.readouterr()
    assert main(["plan"]) == 0
    assert json.loads(capsys.readouterr().out)["name"] == "standard"
    out_file = tmp_path / "results.json"
    assert (
        main(
            [
                "run",
                "--episodes",
                "2",
                "--seed",
                "1",
                "--plan",
                str(plan_file),
                "--param",
                "steps=4",
                "--json",
                str(out_file),
            ]
        )
        == 0
    )
    out = capsys.readouterr().out
    assert "task success" in out and out_file.exists()
    assert main(["run", "--episodes", "1", "--arms", "chaos", "--format", "json"]) == 0
    assert "arms" in json.loads(capsys.readouterr().out)
    assert main(["report", str(out_file)]) == 0
    assert "accuracy" in capsys.readouterr().out
    assert main(["report", str(out_file), "--format", "json"]) == 0
    assert json.loads(capsys.readouterr().out)["arms"]["chaos"]["episodes"] == 2


def test_run_errors(capsys):
    import pytest

    with pytest.raises(SystemExit):
        main(["run", "--scenario", "nomodule", "--episodes", "1"])
    with pytest.raises(SystemExit):
        main(["run", "--arms", "bogus", "--episodes", "1"])
    with pytest.raises(SystemExit):
        main(["run", "--param", "novalue", "--episodes", "1"])
    with pytest.raises(SystemExit):
        main(["run", "--scenario", "agent_chaos.sim:missing", "--episodes", "1"])
