import importlib
import importlib.util
from pathlib import Path

import pytest


def cli():
    assert importlib.util.find_spec("vlm_orchestrator.harness.g1.cli"), "G1 CLI missing"
    return importlib.import_module("vlm_orchestrator.harness.g1.cli")


def test_cli_rejects_free_prompt_and_unknown_skill():
    module = cli()
    with pytest.raises(SystemExit):
        module.main(["run", "--skill", "bottle_to_right_table", "--prompt", "invented"])
    with pytest.raises(SystemExit):
        module.main(["run", "--skill", "invented"])


def test_skills_needs_no_vlm_or_control(capsys):
    profile = Path(__file__).resolve().parents[1] / "configs/g1/workstation.yaml"
    assert cli().main(["--profile", str(profile), "skills"]) == 0
    assert "bottle_to_right_table" in capsys.readouterr().out


def test_status_and_observe_do_not_claim(tmp_path, monkeypatch, capsys):
    from test_g1_runner import Executor

    e = Executor()
    module = cli()
    monkeypatch.setattr(module, "G1Client", lambda *args: e.client())
    assert module.main(["status"]) == 0
    assert module.main(["observe", "--output", str(tmp_path / "camera.jpg")]) == 0
    assert (tmp_path / "camera.jpg").exists()
    assert not e.calls
