import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

import repeat_native_turns as batch


@pytest.mark.parametrize('exit_code,outcome,expected', [
    (0, 'passed_control_checks', 0),
    (17, 'passed_control_checks', 1),
    (0, 'failed', 1),
    (17, None, 1),
])
def test_batch_propagates_driver_failure_but_preserves_failed_turn_controls(
    monkeypatch, tmp_path, exit_code, outcome, expected,
):
    monkeypatch.setattr(sys, 'argv', ['batch', '--native', str(tmp_path/'native'),
                        '--profile', str(tmp_path/'profile'), '--output', str(tmp_path/'results')])

    def run(argv, **kwargs):
        folder = Path(argv[argv.index('--output')+1])
        folder.mkdir()
        if outcome is not None:
            (folder/'result.json').write_text(json.dumps(dict(outcome=outcome, turn_success=False)))
        return SimpleNamespace(returncode=exit_code)

    monkeypatch.setattr(batch.subprocess, 'run', run)
    assert batch.main() == expected
    result = json.loads((tmp_path/'results/batch-results.json').read_text())
    assert result['control_checks_passed'] == (expected == 0)
    assert len(result['cases']) == 3
