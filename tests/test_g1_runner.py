from dataclasses import replace
import importlib
import importlib.util
from pathlib import Path
import threading
import time

from vlm_orchestrator.harness.g1.contract import (
    Execution,
    Lease,
    Response,
    SkillCall,
    Status,
)
from vlm_orchestrator.harness.g1.registry import load_profile
from test_g1_monitor import snapshot


PROFILE = load_profile(
    Path(__file__).resolve().parents[1] / "configs/g1/workstation.yaml"
)


class Executor:
    def __init__(self):
        self.phase = "IDLE"
        self.epoch = 0
        self.execution = None
        self.owner = None
        self.calls = []
        self.heartbeats = 0
        self.reboot = False
        self.profile_mismatch = False

    def status(self):
        return Status(
            "boot2" if self.reboot else "boot",
            self.phase,
            "bottle_to_right_table" if self.execution else None,
            self.execution,
            self.epoch,
            True,
            True,
            "PLANNER" if self.phase != "MANIPULATING" else "POSE",
            self.phase != "MANIPULATING",
            self.phase != "MANIPULATING",
            0.0,
            1,
            0.0,
            "frame",
            "wrong" if self.profile_mismatch else PROFILE.registry_sha256,
            "127.0.0.1",
            15558,
            PROFILE.checkpoint,
            self.owner,
            None,
        )

    def client(self):
        return LocalClient(self)


class LocalClient:
    def __init__(self, executor):
        self.e = executor

    def get_status(self):
        if self.e.phase == "RESETTING":
            self.e.phase = "COMPLETED"
        return self.e.status()

    def observe(self):
        s = snapshot(
            str(time.monotonic()), time.monotonic(), self.e.epoch, self.e.execution
        )
        return replace(s, runtime_id=self.e.status().runtime_id)

    def request(self, method, params, **kwargs):
        self.e.calls.append(method)
        if method == "claim_control":
            self.e.owner = kwargs["session_id"]
            result = Lease("boot", self.e.owner, "lease", time.monotonic() + 2)
        elif method == "heartbeat":
            self.e.heartbeats += 1
            result = self.e.status()
        elif method == "release_control":
            self.e.owner = None
            result = self.e.status()
        else:
            if method == "start_manipulation":
                self.e.phase = "MANIPULATING"
                self.e.execution = "mission"
                self.e.epoch += 1
            elif method == "pause_manipulation":
                self.e.phase = "PAUSED"
                self.e.epoch += 1
            elif method == "reset_standing":
                self.e.phase = "RESETTING"
                self.e.epoch += 1
            elif method == "cancel":
                self.e.phase = "INTERRUPTED"
                self.e.epoch += 1
            result = Execution(
                "boot",
                "mission",
                "bottle_to_right_table",
                self.e.phase,
                self.e.epoch,
                time.monotonic(),
                None,
            )
        return Response("r", self.e.status().runtime_id, result)

    def close(self):
        pass


class Monitor:
    def __init__(self, outcome="complete", delay=0.0):
        self.outcome = outcome
        self.delay = delay
        self.raw_response = "{}"

    def begin(self, execution, initial):
        self.execution = execution

    def check(self, frame):
        time.sleep(self.delay)
        from vlm_orchestrator.harness.g1.monitor import CompletionDecision

        return CompletionDecision(
            self.outcome,
            "placed",
            frame.frame_id,
            frame.execution_id,
            frame.inference_epoch,
            frame.received_at,
            time.monotonic(),
        )


def runner(e, tmp_path, monitor=None):
    assert importlib.util.find_spec("vlm_orchestrator.harness.g1.runner"), (
        "Mission runner missing"
    )
    cls = importlib.import_module("vlm_orchestrator.harness.g1.runner").HarnessRunner
    return cls(PROFILE, e.client, monitor or Monitor(), tmp_path)


def test_success_pauses_then_resets(tmp_path):
    e = Executor()
    result = runner(e, tmp_path).run(SkillCall("bottle_to_right_table", {}))
    assert result.outcome == "completed"
    ordered = [m for m in e.calls if m != "heartbeat"]
    assert ordered == [
        "claim_control",
        "start_manipulation",
        "pause_manipulation",
        "reset_standing",
        "release_control",
    ]


def test_failure_preserves_hands_and_does_not_reset(tmp_path):
    e = Executor()
    result = runner(e, tmp_path, Monitor("failure")).run(
        SkillCall("bottle_to_right_table", {})
    )
    assert result.outcome == "interrupted"
    assert "cancel" in e.calls and "reset_standing" not in e.calls


def test_vlm_delay_does_not_delay_heartbeat(tmp_path):
    e = Executor()
    runner(e, tmp_path, Monitor(delay=0.7)).run(SkillCall("bottle_to_right_table", {}))
    assert e.heartbeats >= 2


def test_runtime_reboot_interrupts_mission(tmp_path):
    e = Executor()
    timer = threading.Timer(0.1, lambda: setattr(e, "reboot", True))
    timer.start()
    result = runner(e, tmp_path, Monitor(delay=0.4)).run(
        SkillCall("bottle_to_right_table", {})
    )
    timer.join()
    assert result.outcome != "completed"
    assert "reset_standing" not in e.calls


def test_runtime_profile_mismatch_aborts(tmp_path):
    e = Executor()
    e.profile_mismatch = True
    result = runner(e, tmp_path).run(SkillCall("bottle_to_right_table", {}))
    assert result.outcome == "fault"
    assert not e.calls


def test_terminal_mission_does_not_recycle(tmp_path):
    e = Executor()
    runner(e, tmp_path).run(SkillCall("bottle_to_right_table", {}))
    assert e.calls.count("start_manipulation") == 1


def test_bad_later_skill_prevents_claim(tmp_path):
    e = Executor()
    import pytest

    with pytest.raises(ValueError):
        runner(e, tmp_path).run_sequence(
            [SkillCall("bottle_to_right_table", {}), SkillCall("invented", {})]
        )
    assert not e.calls
