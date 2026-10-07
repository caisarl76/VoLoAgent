"""Cross-repository IPC checks. The native fixture has no robot-action socket."""

from contextlib import contextmanager
import json
import os
from pathlib import Path
import subprocess
import time

import pytest

from vlm_orchestrator.harness.g1.client import G1Client
from vlm_orchestrator.harness.g1.contract import SkillCall
from vlm_orchestrator.harness.g1.monitor import G1CompletionMonitor
from vlm_orchestrator.harness.g1.registry import load_profile
from vlm_orchestrator.harness.g1.runner import HarnessRunner


ROOT = Path(__file__).resolve().parents[1]
PROFILE_PATH = ROOT / "configs/g1/workstation.yaml"
PROFILE = load_profile(PROFILE_PATH)


@contextmanager
def executor(native_runtime, tmp_path, scenario="success", profile_path=PROFILE_PATH):
    checkout, python = native_runtime
    endpoint = "ipc://" + str(tmp_path / "runtime.sock")
    evidence = tmp_path / "native.jsonl"
    with (tmp_path / "native.log").open("w") as log:
        env = {**os.environ, "PYTHONPATH": str(checkout)}
        process = subprocess.Popen(
            [
                str(python),
                str(checkout / "gear_sonic/tests/harness_fake_runtime.py"),
                "--endpoint",
                endpoint,
                "--profile",
                str(profile_path),
                "--scenario",
                scenario,
                "--evidence",
                str(evidence),
            ],
            cwd=checkout,
            env=env,
            stdout=log,
            stderr=log,
        )
        try:
            deadline = time.monotonic() + 5
            while not Path(endpoint[6:]).exists():
                assert process.poll() is None, (tmp_path / "native.log").read_text()
                assert time.monotonic() < deadline, "Native fixture did not start"
                time.sleep(0.02)
            yield endpoint, evidence
        finally:
            process.terminate()
            process.wait(timeout=5)


def events(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


@pytest.mark.parametrize("sequence", [False, True])
def test_subskill_hold_cross_process(native_runtime, tmp_path, sequence):
    from test_g1_subskill_handoff import candidate_profile

    profile_path = candidate_profile(tmp_path)
    profile = load_profile(profile_path)
    with executor(native_runtime, tmp_path, "subskill_handoff", profile_path) as (
        endpoint,
        evidence,
    ):
        monitor = G1CompletionMonitor(
            profile.require_skill("pick_bottle_and_hold"),
            lambda *_: (
                '{"status":"complete","action":"next","reason":"test-only skill result"}'
            ),
            time.monotonic,
            profile.limits,
        )
        calls = [SkillCall("pick_bottle_and_hold", {})]
        if sequence:
            calls.append(SkillCall("place_held_bottle_on_stool", {}))
        result = HarnessRunner(
            profile, lambda: G1Client(endpoint), monitor, tmp_path / "missions"
        ).run_sequence(calls)
        assert result.outcome == "completed", result.reason
        recorded = events(evidence)
        mission = events(result.evidence_dir / "events.jsonl")
        prompts = [e for e in recorded if e["event"] == "prompt"]
        assert [e["prompt"] for e in prompts] == [
            profile.require_skill(c.skill_id).prompt for c in calls
        ]
        hold = next(e for e in recorded if e["event"] == "hold")
        assert hold["open_hands"] is False
        assert hold["left"] == pytest.approx([0.3] * 7)
        assert hold["right"] == pytest.approx([0.4] * 7)
        assert any(e["event"] == "late_result_rejected" for e in recorded)
        requests = [e for e in mission if e["event"] == "request"]
        assert all(e["method"] != "cancel" for e in requests)
        if sequence:
            assert (
                hold["at"]
                < prompts[1]["at"]
                < next(e["at"] for e in recorded if e["event"] == "reset")
            )
            assert len([e for e in recorded if e["event"] == "reset"]) == 1
            assert len([e for e in requests if e["method"] == "start_manipulation"]) > 2
        else:
            assert not any(e["event"] == "reset" for e in recorded)
        assert {e["publisher"] for e in recorded if "publisher" in e} == {"memory-only"}


def test_mirrored_contract_and_golden_fixtures(native_runtime):
    checkout, _ = native_runtime
    assert (ROOT / "tests/fixtures/g1_rpc_v1.json").read_bytes() == (
        checkout / "gear_sonic/tests/fixtures/g1_rpc_v1.json"
    ).read_bytes()
    for local, native in [
        ("contract.py", "harness_contract.py"),
        ("registry.py", "harness_profile.py"),
    ]:
        ours = (ROOT / "vlm_orchestrator/harness/g1" / local).read_text()
        theirs = (checkout / "gear_sonic/utils/inference" / native).read_text()
        assert ours.replace("from .contract", "from .harness_contract") == theirs


@pytest.mark.parametrize(
    "scenario", ["success", "late_result", "frozen_camera", "failed_ack"]
)
def test_bottle_mission_cross_process(native_runtime, tmp_path, scenario):
    with executor(native_runtime, tmp_path, scenario) as (endpoint, evidence):
        monitor = G1CompletionMonitor(
            PROFILE.require_skill("bottle_to_right_table"),
            lambda *_: (
                '{"status":"complete","action":"next","reason":"placed and released"}'
            ),
            time.monotonic,
            PROFILE.limits,
        )
        result = HarnessRunner(
            PROFILE, lambda: G1Client(endpoint), monitor, tmp_path / "missions"
        ).run(SkillCall("bottle_to_right_table", {}))
        recorded = events(evidence)
        mission = events(result.evidence_dir / "events.jsonl")
        assert len([e for e in mission if e["event"] == "mission_result"]) == 1
        assert {e["publisher"] for e in recorded if "publisher" in e} == {"memory-only"}
        assert [e["prompt"] for e in recorded if e["event"] == "prompt"] == [
            "pick drink bottle and place it on the right table"
        ]
        if scenario in {"success", "late_result"}:
            assert result.outcome == "completed", result.reason
            decisions = [e for e in mission if e["event"] == "monitor_decision"]
            assert [e["outcome"] for e in decisions] == ["in_progress", "complete"]
            assert decisions[0]["frame_id"] != decisions[1]["frame_id"]
            terminal = next(
                e
                for e in mission
                if e["event"] == "status" and e["phase"] == "COMPLETED"
            )
            assert terminal["hold_confirmed"] and terminal["planner_reference_active"]
            assert terminal["inference_epoch"] > decisions[-1]["inference_epoch"]
            assert any(e["event"] == "measured_settled" for e in recorded)
            if scenario == "late_result":
                assert any(e["event"] == "late_result_rejected" for e in recorded)
        else:
            assert result.outcome != "completed"
            assert not any(e["event"] == "measured_settled" for e in recorded)


def test_coordinator_loss_expires_lease_and_preserves_hands(native_runtime, tmp_path):
    with executor(native_runtime, tmp_path) as (endpoint, evidence):
        client = G1Client(endpoint)
        status = client.get_status()
        lease = client.request(
            "claim_control",
            {"registry_sha256": status.registry_sha256},
            session_id="lost-agent",
        ).result
        started = client.request(
            "start_manipulation",
            {"skill_id": "bottle_to_right_table"},
            session_id="lost-agent",
            lease_id=lease.lease_id,
        ).result
        assert started.phase == "MANIPULATING"
        client.close()  # No heartbeat or cleanup from this owner.
        time.sleep(PROFILE.limits.lease_s + 0.3)
        observer = G1Client(endpoint)
        try:
            status = observer.get_status()
            assert status.phase == "INTERRUPTED" and status.owner_session_id is None
            assert status.hold_confirmed and status.reason == "lease_expired"
            holds = [e for e in events(evidence) if e["event"] == "hold"]
            assert holds and all(not e["open_hands"] for e in holds)
        finally:
            observer.close()


def test_bottle_then_bounded_reposition(native_runtime, tmp_path):
    with executor(native_runtime, tmp_path, "locomotion") as (endpoint, evidence):
        monitor = G1CompletionMonitor(
            PROFILE.require_skill("bottle_to_right_table"),
            lambda *_: '{"status":"complete","action":"next","reason":"released"}',
            time.monotonic,
        )
        runner = HarnessRunner(
            PROFILE, lambda: G1Client(endpoint), monitor, tmp_path / "missions"
        )
        runner.locomotion_enabled = True
        plan = json.loads((ROOT / "configs/g1/bottle_then_reposition.json").read_text())
        result = runner.run_sequence([SkillCall(**call) for call in plan["skills"]])
        assert result.outcome == "completed", result.reason
        phases = [e["phase"] for e in events(evidence) if e["event"] == "phase"]
        assert (
            "WALKING" in phases
            and "TURNING" in phases
            and phases.count("RESETTING") == 1
        )


def test_coordinator_loss_mid_walk_stops_movement(native_runtime, tmp_path):
    with executor(native_runtime, tmp_path, "locomotion") as (endpoint, evidence):
        client = G1Client(endpoint)
        status = client.get_status()
        lease = client.request(
            "claim_control",
            {"registry_sha256": status.registry_sha256},
            session_id="lost-agent",
        ).result
        reset = client.request(
            "reset_standing",
            {"execution_id": "", "open_hands": False},
            session_id="lost-agent",
            lease_id=lease.lease_id,
        ).result
        deadline = time.monotonic() + 3
        while client.get_status().phase != "COMPLETED":
            assert time.monotonic() < deadline
            client.request(
                "heartbeat", {}, session_id="lost-agent", lease_id=lease.lease_id
            )
            time.sleep(0.05)
        started = client.request(
            "walk_for",
            {"direction": "forward", "duration_s": 5.0, "speed_mps": 0.2},
            session_id="lost-agent",
            lease_id=lease.lease_id,
        ).result
        assert started.phase == "WALKING" and started.execution_id != reset.execution_id
        client.close()
        time.sleep(PROFILE.limits.lease_s + 0.3)
        observer = G1Client(endpoint)
        try:
            status = observer.get_status()
            assert status.phase == "INTERRUPTED" and status.hold_confirmed
            assert any(
                e["event"] == "hold" and not e["open_hands"] for e in events(evidence)
            )
        finally:
            observer.close()


def test_operator_takeover_pauses_owned_mission(native_runtime, tmp_path):
    with executor(native_runtime, tmp_path, "operator_pause") as (endpoint, evidence):
        monitor = G1CompletionMonitor(
            PROFILE.require_skill("bottle_to_right_table"),
            lambda *_: '{"status":"complete","action":"next","reason":"released"}',
            time.monotonic,
        )
        result = HarnessRunner(
            PROFILE, lambda: G1Client(endpoint), monitor, tmp_path / "missions"
        ).run(SkillCall("bottle_to_right_table", {}))
        assert result.outcome == "interrupted"
        recorded = events(evidence)
        index = next(
            i for i, e in enumerate(recorded) if e["event"] == "operator_override"
        )
        assert not any(e.get("kind") == "pose" for e in recorded[index:])
        assert not any(e["event"] == "reset" for e in recorded)
        assert any(e["event"] == "late_result_rejected" for e in recorded)
