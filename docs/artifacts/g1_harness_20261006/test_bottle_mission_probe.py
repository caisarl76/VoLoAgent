"""A coordinator completion cannot bypass missing independent evidence."""

from dataclasses import dataclass
import json
import os
from pathlib import Path
from types import SimpleNamespace

import bottle_mission_probe as probe


@dataclass
class ReadyStatus:
    controller_running: bool = True
    policy_ready: bool = True


@dataclass
class CompletedMission:
    outcome: str = "completed"
    reason: str = "scripted coordinator completion"


def test_completed_coordinator_with_missing_truth_is_failed(tmp_path, monkeypatch):
    native = tmp_path / "native"
    for relative in ["gear_sonic/scripts/run_vla_inference.py",
                     "gear_sonic_deploy/target/release/g1_deploy_onnx_ref"]:
        path = native / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("test stub; never executed\n")
    root = Path(__file__).parents[3]
    output = tmp_path / "evidence"
    monkeypatch.setattr("sys.argv", ["probe", "--native", str(native),
        "--profile", str(root / "configs/g1/workstation.yaml"),
        "--scene", str(Path(__file__).parent / "bottle-measured-scene/scene.xml"),
        "--output", str(output), "--env-file", str(tmp_path / "unused.env"),
        "--monitor-mode", "simulator_truth_control"])
    socket = SimpleNamespace(setsockopt=lambda *a: None, bind=lambda *a: None,
                             send_string=lambda *a: None, close=lambda: None)
    monkeypatch.setattr(probe.zmq, "Context", lambda: SimpleNamespace(socket=lambda *a: socket, term=lambda: None))
    monkeypatch.setattr(probe.pty, "openpty", lambda: (os.open(os.devnull, os.O_RDONLY), os.open(os.devnull, os.O_RDONLY)))
    monkeypatch.setattr(probe.time, "sleep", lambda *a: None)
    monkeypatch.setattr(probe.os, "killpg", lambda *a: None)
    created_endpoints = []

    class Process:
        pid = 0
        returncode = None

        def __init__(self, argv, **kwargs):
            log = kwargs["stdout"]
            log.write("Server is ready and listening\nInit Done\n")
            log.flush()
            if "--harness-endpoint" in argv:
                endpoint = Path(argv[argv.index("--harness-endpoint") + 1][6:])
                endpoint.touch()
                created_endpoints.append(endpoint)

        def poll(self):
            return self.returncode

        def wait(self, *args):
            self.returncode = 0
            return 0

    monkeypatch.setattr(probe.subprocess, "Popen", Process)
    monkeypatch.setattr(probe, "G1Client", lambda *a: SimpleNamespace(get_status=lambda: ReadyStatus(), close=lambda: None))
    monkeypatch.setattr(probe, "HarnessRunner", lambda *a: SimpleNamespace(run=lambda *a: CompletedMission()))
    try:
        exit_code = probe.main()
    finally:
        for path in created_endpoints:
            path.unlink()
    result = json.loads((output / "result.json").read_text())
    assert result["mission"]["outcome"] == "completed"
    assert "bottle-ground-truth.jsonl" in result["reason"]
    assert result["outcome"] == "failed"
    assert result["placement_validated"] is False
    assert exit_code == 1
