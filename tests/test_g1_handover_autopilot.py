from dataclasses import replace
import importlib
import threading
import time

import pytest

from test_g1_handover_profile import handover_profile
from test_g1_handover_monitor import reply
from test_g1_runner import Executor, LocalClient
from vlm_orchestrator.harness.g1.contract import Execution, Response
from vlm_orchestrator.harness.g1.handover_monitor import HandoverMonitor
from vlm_orchestrator.harness.g1.registry import load_profile


class HandoverExecutor(Executor):
    def __init__(self, profile):
        super().__init__()
        self.profile, self.index = profile, 1
        self.hand_open, self.capability = True, True
        self.camera_age = 0.0

    def status(self):
        self.index += 1
        return replace(
            super().status(),
            registry_sha256=self.profile.registry_sha256,
            checkpoint_expected=self.profile.checkpoint,
            telemetry_index=self.index,
            ready_return_enabled=self.capability,
            right_hand_open=self.hand_open,
            observation_age_s=self.camera_age,
        )

    def client(self):
        return HandoverClient(self)


class HandoverClient(LocalClient):
    def request(self, method, params, **kwargs):
        if method == "reset_ready":
            self.e.calls.append(method)
            self.e.epoch += 1
            self.e.execution = (
                "ready" if not params["execution_id"] else params["execution_id"]
            )
            self.e.phase = "RESETTING"
            return Response(
                "r",
                "boot",
                Execution(
                    "boot",
                    self.e.execution,
                    "reset_ready",
                    "RESETTING",
                    self.e.epoch,
                    time.monotonic(),
                    None,
                ),
            )
        return super().request(method, params, **kwargs)


def build(tmp_path, *, vlm=None, monitor=None, verified=True):
    mod = importlib.util.find_spec("vlm_orchestrator.harness.g1.autopilot")
    assert mod is not None, "Stationary handover autopilot missing"
    cls = importlib.import_module(mod.name).HandoverRunner
    profile = load_profile(handover_profile(tmp_path, verified=verified))
    e = HandoverExecutor(profile)
    m = monitor or HandoverMonitor(
        vlm or (lambda *_: reply("yes")), time.monotonic, profile.limits
    )
    return cls(profile, e.client, m, tmp_path / "evidence"), e, m


def test_three_cycles_share_lease_and_return_ready(tmp_path):
    r, e, _ = build(tmp_path)
    result = r.run_autopilot(3)
    assert result.outcome == "completed", result.reason
    assert e.calls.count("claim_control") == e.calls.count("release_control") == 1
    assert (
        e.calls.count("start_manipulation") == e.calls.count("pause_manipulation") == 3
    )
    assert e.calls.count("reset_ready") == 4
    assert "reset_standing" not in e.calls


def test_fresh_cached_pretransition_frames_wait_for_new_capture(tmp_path):
    class CachedClient(HandoverClient):
        def observe(self):
            frame = super().observe()
            return replace(
                frame,
                received_at=frame.received_at - 0.1,
                source_timestamp=frame.source_timestamp - 0.1,
                age_s=0.1,
            )

    r, e, _ = build(tmp_path)
    r.client_factory = lambda: CachedClient(e)
    result = r.run_autopilot(1)
    assert result.outcome == "completed", result.reason
    assert e.calls.count("reset_ready") == 2


def test_empty_hand_occlusion_waits_without_reset(tmp_path):
    r, e, m = build(tmp_path)
    observations = ["unknown", "no", "yes", "yes"]
    checks = []

    def vision(*_):
        if m.phase == "wait_empty":
            checks.append(e.calls.count("reset_ready"))
            return reply(observations.pop(0))
        return reply("yes")

    m.vlm_call_fn = vision
    assert r.run_autopilot(1).outcome == "completed"
    assert checks == [1, 1, 1, 1]
    assert e.calls.count("reset_ready") == 2


def test_roll_failure_interrupts_without_ready_return(tmp_path):
    r, e, m = build(tmp_path)
    m.vlm_call_fn = lambda *_: (
        reply("failure", "tilted palm roll; rescued")
        if m.phase == "wait_empty"
        else reply("yes")
    )
    result = r.run_autopilot(3)
    assert result.outcome == "interrupted"
    assert "roll" in result.reason
    assert e.calls.count("start_manipulation") == 1
    assert e.calls.count("reset_ready") == 1
    assert "cancel" in e.calls


@pytest.mark.parametrize(
    "field,value",
    [
        ("cycle_id", 99),
        ("phase", "old"),
        ("runtime_id", "old"),
        ("execution_id", "old"),
        ("inference_epoch", 99),
        ("captured_at", 0.0),
    ],
)
def test_old_phase_or_cycle_decision_interrupts(tmp_path, field, value):
    class WrongMonitor(HandoverMonitor):
        def check(self, frame):
            return replace(super().check(frame), **{field: value})

    m = WrongMonitor(lambda *_: reply("yes"), time.monotonic)
    r, e, _ = build(tmp_path, monitor=m)
    result = r.run_autopilot(1)
    assert result.outcome == "interrupted"
    assert "start_manipulation" not in e.calls


@pytest.mark.parametrize("closed", [False, True])
def test_waiting_does_not_block_ownership_checks(tmp_path, closed):
    r, e, m = build(tmp_path)
    if closed:
        e.hand_open = False

        def vision(*_):
            if m.phase == "pick_offer":
                time.sleep(0.4)
            return reply("yes")
    else:

        def vision(*_):
            if m.phase == "ready":
                time.sleep(0.4)
                return reply("no", "Bottle in person's hand, not on desk")
            return reply("yes")

    m.vlm_call_fn = vision
    timer = threading.Timer(0.15, lambda: setattr(e, "reboot", True))
    timer.start()
    result = r.run_autopilot(1)
    timer.join()
    assert result.outcome == "interrupted"
    assert "pause_manipulation" not in e.calls
    assert e.calls.count("reset_ready") == 1


@pytest.mark.parametrize("verified,capability", [(False, True), (True, False)])
def test_unverified_profile_or_missing_native_capability_never_claims(
    tmp_path, verified, capability
):
    r, e, _ = build(tmp_path, verified=verified)
    e.capability = capability
    if not verified:
        with pytest.raises(ValueError):
            r.run_autopilot(1)
    else:
        assert r.run_autopilot(1).outcome == "fault"
    assert not e.calls


def test_startup_does_not_open_a_hand_that_is_holding_a_bottle(tmp_path):
    r, e, m = build(tmp_path, vlm=lambda *_: reply("no", "Bottle still in robot hand"))
    timer = threading.Timer(0.15, lambda: setattr(e, "reboot", True))
    timer.start()
    result = r.run_autopilot(1)
    timer.join()
    assert result.outcome == "interrupted"
    assert "reset_ready" not in e.calls


def test_stale_camera_interrupts_during_pending_vision(tmp_path):
    def vision(*_):
        time.sleep(0.2)
        return reply("yes")

    r, e, _ = build(tmp_path, vlm=vision)
    timer = threading.Timer(0.1, lambda: setattr(e, "camera_age", 0.6))
    timer.start()
    result = r.run_autopilot(1)
    timer.join()
    assert result.outcome == "interrupted"
    assert "camera" in result.reason.lower()
    assert "reset_ready" not in e.calls


@pytest.mark.parametrize("cycles", [0, -1, True, 1.5])
def test_bad_cycle_limit_rejected(tmp_path, cycles):
    r, e, _ = build(tmp_path)
    with pytest.raises(ValueError):
        r.run_autopilot(cycles)
    assert not e.calls
