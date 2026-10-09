import pytest

from test_g1_handover_profile import ROOT
from vlm_orchestrator.harness.g1.cli import main


def test_autopilot_cli_rejects_disabled_template_before_contact(capsys):
    with pytest.raises(SystemExit) as exc:
        main(
            [
                "--profile",
                str(ROOT / "configs/g1/handover.example.yaml"),
                "autopilot",
                "--max-cycles",
                "3",
                "--vlm-model",
                "openai/gpt-5.6-sol",
                "--vlm-base-url",
                "https://api.genon.ai/v1",
            ]
        )
    assert exc.value.code == 2
    assert "Verified stationary handover profile" in capsys.readouterr().err
