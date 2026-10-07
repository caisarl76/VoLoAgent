"""Behavior of optional hold completion; no robot or policy service involved."""

from dataclasses import replace
import json
from pathlib import Path
import time

import pytest
import yaml

from test_g1_runner import Executor, LocalClient, Monitor, PROFILE
from vlm_orchestrator.harness.g1.client import RPCError
from vlm_orchestrator.harness.g1.contract import Response, SkillCall
from vlm_orchestrator.harness.g1.registry import load_profile
from vlm_orchestrator.harness.g1.runner import HarnessRunner


def candidate_profile(tmp_path):
    data = yaml.safe_load(
        (
            Path(__file__).resolve().parents[1] / "configs/g1/workstation.yaml"
        ).read_text()
    )
    data["skills"] = {
        "pick_bottle_and_hold": {
            "prompt": "pick the bottle and hold it",
            "completion_criteria": "bottle visibly held above the source table",
            "completion_action": "hold",
        },
        "place_held_bottle_on_stool": {
            "prompt": "place the held bottle on the green stool",
            "completion_criteria": "released bottle visibly rests on the stool",
        },
    }
    path = tmp_path / "candidate-test-only.yaml"
    path.write_text(yaml.safe_dump(data))
    return path


class HandoffExecutor(Executor):
    def __init__(self, profile):
        super().__init__()
        self.profile = profile
        self.ack = True
        self.hold_age = 0.0
        self.bad_epoch = False
        self.not_ready = 0
        self.start_error = None
        self.started = 0
        self.skill = None
        self.ack_polls = 0
        self.confirm_after = 0
        self.after_rejection_observe_delay = 0.0
        self.next_observe_delay = 0.0

    def status(self):
        s = replace(
            super().status(),
            registry_sha256=self.profile.registry_sha256,
            skill_id=self.skill,
            telemetry_index=1 + self.started,
        )
        if self.phase == "PAUSED":
            self.ack_polls += 1
            confirmed = self.ack and self.ack_polls > self.confirm_after
            s = replace(
                s,
                hold_confirmed=confirmed,
                planner_reference_active=confirmed,
                telemetry_age_s=self.hold_age,
                telemetry_index=2 + self.started,
                inference_epoch=self.epoch + int(self.bad_epoch),
            )
        return s

    def client(self):
        return HandoffClient(self)


class HandoffClient(LocalClient):
    def observe(self):
        delay, self.e.next_observe_delay = self.e.next_observe_delay, 0.0
        time.sleep(delay)
        return super().observe()

    def request(self, method, params, **kwargs):
        if method == "start_manipulation" and self.e.started:
            if self.e.start_error:
                self.e.calls.append(method)
                raise RPCError(
                    self.e.start_error, "ambiguous handoff transport failure"
                )
            if self.e.not_ready:
                self.e.not_ready -= 1
                self.e.next_observe_delay = self.e.after_rejection_observe_delay
                self.e.calls.append(method)
                return Response(
                    "r", "boot", None, {"code": "NOT_READY", "message": "worker busy"}
                )
        if method == "start_manipulation":
            self.e.started += 1
            self.e.skill = params["skill_id"]
        response = super().request(method, params, **kwargs)
        if method in {
            "start_manipulation",
            "pause_manipulation",
            "reset_standing",
            "cancel",
        }:
            response = replace(
                response, result=replace(response.result, skill_id=self.e.skill)
            )
        return response


def setup(tmp_path):
    profile = load_profile(candidate_profile(tmp_path))
    profile = replace(
        profile,
        limits=replace(profile.limits, planner_deadline_s=0.08, policy_deadline_s=0.08),
    )
    e = HandoffExecutor(profile)
    return e, HarnessRunner(profile, e.client, Monitor(), tmp_path / "missions")


def test_default_completion_keeps_legacy_reset():
    assert (
        PROFILE.require_skill("bottle_to_right_table").completion_action
        == "reset_standing"
    )


@pytest.mark.parametrize("value", ["open_hands", [], {}, None, True])
def test_malformed_completion_action_rejected(tmp_path, value):
    path = candidate_profile(tmp_path)
    data = yaml.safe_load(path.read_text())
    data["skills"]["pick_bottle_and_hold"]["completion_action"] = value
    path.write_text(yaml.safe_dump(data))
    with pytest.raises(ValueError):
        load_profile(path)


def test_hold_completes_without_reset_or_cancel(tmp_path):
    e, runner = setup(tmp_path)
    e.confirm_after = 3
    result = runner.run(SkillCall("pick_bottle_and_hold", {}))
    assert result.outcome == "completed", result.reason
    assert e.ack_polls > 3
    assert "reset_standing" not in e.calls and "cancel" not in e.calls
    assert e.calls[-1] == "release_control"
    records = [
        json.loads(line)
        for line in (result.evidence_dir / "events.jsonl").read_text().splitlines()
    ]
    assert any(
        r["event"] == "skill_complete" and r["completion_action"] == "hold"
        for r in records
    )


def test_pick_to_place_resets_only_after_placement(tmp_path):
    e, runner = setup(tmp_path)
    result = runner.run_sequence(
        [
            SkillCall("pick_bottle_and_hold", {}),
            SkillCall("place_held_bottle_on_stool", {}),
        ]
    )
    assert result.outcome == "completed", result.reason
    assert [m for m in e.calls if m != "heartbeat"] == [
        "claim_control",
        "start_manipulation",
        "pause_manipulation",
        "start_manipulation",
        "pause_manipulation",
        "reset_standing",
        "release_control",
    ]
    records = [
        json.loads(line)
        for line in (result.evidence_dir / "events.jsonl").read_text().splitlines()
    ]
    placement = next(
        r
        for r in records
        if r["event"] == "skill_complete" and r["completion_action"] == "reset_standing"
    )
    assert placement["phase"] == "COMPLETED"


@pytest.mark.parametrize("condition", ["no_ack", "stale_feedback", "wrong_epoch"])
def test_unconfirmed_hold_blocks_next_skill(tmp_path, condition):
    e, runner = setup(tmp_path)
    if condition == "no_ack":
        e.ack = False
    elif condition == "stale_feedback":
        e.hold_age = 0.51
    else:
        e.bad_epoch = True
    result = runner.run_sequence(
        [
            SkillCall("pick_bottle_and_hold", {}),
            SkillCall("place_held_bottle_on_stool", {}),
        ]
    )
    assert result.outcome == "interrupted", result.reason
    assert e.started == 1 and "reset_standing" not in e.calls


def test_handoff_waits_for_previous_worker(tmp_path):
    e, runner = setup(tmp_path)
    e.not_ready = 1
    result = runner.run_sequence(
        [
            SkillCall("pick_bottle_and_hold", {}),
            SkillCall("place_held_bottle_on_stool", {}),
        ]
    )
    assert result.outcome == "completed", result.reason
    assert e.calls.count("start_manipulation") == 3 and e.started == 2


def test_busy_handoff_is_bounded(tmp_path):
    e, runner = setup(tmp_path)
    e.not_ready = 10000
    before = time.monotonic()
    result = runner.run_sequence(
        [
            SkillCall("pick_bottle_and_hold", {}),
            SkillCall("place_held_bottle_on_stool", {}),
        ]
    )
    assert result.outcome == "interrupted" and e.started == 1
    assert time.monotonic() - before < 1
    assert "reset_standing" not in e.calls


def test_ambiguous_handoff_failure_never_retries(tmp_path):
    e, runner = setup(tmp_path)
    e.start_error = "TIMEOUT"
    result = runner.run_sequence(
        [
            SkillCall("pick_bottle_and_hold", {}),
            SkillCall("place_held_bottle_on_stool", {}),
        ]
    )
    assert result.outcome == "fault"
    assert e.calls.count("start_manipulation") == 2 and e.started == 1


def test_slow_observation_cannot_start_after_handoff_deadline(tmp_path):
    e, runner = setup(tmp_path)
    e.not_ready = 1
    e.after_rejection_observe_delay = 0.1
    result = runner.run_sequence(
        [
            SkillCall("pick_bottle_and_hold", {}),
            SkillCall("place_held_bottle_on_stool", {}),
        ]
    )
    assert result.outcome == "interrupted"
    assert e.started == 1 and e.calls.count("start_manipulation") == 2
