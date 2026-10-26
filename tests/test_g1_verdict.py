import json

import numpy as np
import pytest

from vlm_orchestrator.harness.g1 import cli
from vlm_orchestrator.harness.g1.verdict import (
    FAILURE_REASONS,
    SAFETY_STOP,
    append_verdict,
    next_trial,
    pending_trials,
    remaining_trials,
    request_session_end,
    run_console,
)


def make_trial(d, i, episode_end="handover", n_frames=3):
    np.savez(
        d / f"trial_{i:03d}.npz",
        wall=np.zeros(2),
        episode_end=episode_end,
        args="{}",
        model="m1",
        wall_start=10.0 + i,
        wall_stop=20.0 + i,
    )
    blobs = [b"ab"] * n_frames
    np.savez(
        d / f"trial_{i:03d}_frames.npz",
        jpeg=np.frombuffer(b"".join(blobs), np.uint8),
        offsets=np.cumsum([0] + [len(b) for b in blobs]).astype(np.int64),
        t_recv=np.zeros(n_frames),
        names=np.array(["x"] * n_frames, dtype=str),
    )


def lines(d, name="results.jsonl"):
    return [json.loads(s) for s in (d / name).read_text().splitlines()]


def test_append_verdict_writes_lerobot_line(tmp_path):
    make_trial(tmp_path, 0)
    line = append_verdict(tmp_path, 0, "success", "", "")
    assert list(line) == ["trial", "verdict", "reason", "note", "episode_end",
                          "wall_start", "wall_stop", "frames", "model"]
    assert line == {"trial": 0, "verdict": "success", "reason": "", "note": "",
                    "episode_end": "handover", "wall_start": 10.0, "wall_stop": 20.0,
                    "frames": 3, "model": "m1"}
    assert lines(tmp_path) == [line]


def test_pending_lists_unscored_trials(tmp_path):
    for i in range(3):
        make_trial(tmp_path, i)
    append_verdict(tmp_path, 1, "fail", FAILURE_REASONS[0], "n")
    assert pending_trials(tmp_path) == [0, 2]
    append_verdict(tmp_path, 0, "redo", "", "")
    assert pending_trials(tmp_path) == [2]


def test_redo_extends_remaining_trials(tmp_path):
    for i in range(2):
        make_trial(tmp_path, i)
    append_verdict(tmp_path, 0, "success", "", "")
    append_verdict(tmp_path, 1, "redo", "", "")
    assert remaining_trials(tmp_path, 2) == 1
    assert lines(tmp_path, "redo.jsonl")[0]["verdict"] == "redo"
    assert [r["trial"] for r in lines(tmp_path)] == [0]  # no redo in results.jsonl


def test_watchdog_default_reason_and_double_scoring(tmp_path):
    make_trial(tmp_path, 0, episode_end="watchdog: arm speed")
    assert SAFETY_STOP == "safety stop (watchdog)" and SAFETY_STOP in FAILURE_REASONS
    assert append_verdict(tmp_path, 0, "fail", None, "")["reason"] == SAFETY_STOP
    with pytest.raises(ValueError):
        append_verdict(tmp_path, 0, "success", "", "")
    with pytest.raises(FileNotFoundError):
        append_verdict(tmp_path, 5, "success", "", "")


def test_next_trial(tmp_path):
    assert next_trial(tmp_path / "missing") == 0
    assert next_trial(tmp_path) == 0
    make_trial(tmp_path, 0)
    make_trial(tmp_path, 1)
    assert next_trial(tmp_path) == 2
    (tmp_path / "redo.jsonl").write_text(json.dumps({"trial": 4}) + "\n")
    assert next_trial(tmp_path) == 5


def test_end_command_writes_request_file(tmp_path):
    request_session_end(tmp_path)
    assert (tmp_path / "end_session.request").exists()


def test_console_parses_fail_reason_and_note(tmp_path, capsys):
    make_trial(tmp_path, 0)
    make_trial(tmp_path, 1)
    make_trial(tmp_path, 2, episode_end="watchdog: x")
    run_console(tmp_path, iter(["f 3 note text", "r", "f", "e"]), poll_s=0)
    got = lines(tmp_path)
    assert got[0]["reason"] == FAILURE_REASONS[2] and got[0]["note"] == "note text"
    assert lines(tmp_path, "redo.jsonl")[0]["trial"] == 1
    assert got[1]["reason"] == SAFETY_STOP
    assert (tmp_path / "end_session.request").exists()


def test_cli_verdict_subcommand(tmp_path, monkeypatch):
    make_trial(tmp_path, 0)
    monkeypatch.setattr("sys.stdin", iter(["s\n"]))
    assert cli.main(["verdict", "--session-dir", str(tmp_path)]) == 0
    assert lines(tmp_path)[0]["verdict"] == "success"
