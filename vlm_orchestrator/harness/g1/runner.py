"""Bounded mission sequencing, evidence and independent ownership heartbeat."""

from __future__ import annotations

import base64
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import queue
import threading
import time
from typing import Callable, Literal
import uuid

from .client import G1Client, RPCError
from .contract import Execution, Lease, SkillCall
from .monitor import G1CompletionMonitor
from .registry import G1Profile, validate_skill_call


@dataclass(frozen=True)
class MissionResult:
    execution_id: str
    outcome: Literal["completed", "interrupted", "fault"]
    reason: str
    evidence_dir: Path


class MissionInterrupted(RuntimeError):
    pass


class HarnessRunner:
    def __init__(
        self,
        profile: G1Profile,
        client_factory: Callable[[], G1Client],
        monitor: G1CompletionMonitor | None,
        evidence_dir: Path,
    ):
        self.profile, self.client_factory, self.monitor = (
            profile,
            client_factory,
            monitor,
        )
        self.evidence_dir = Path(evidence_dir)
        self._log_lock = threading.Lock()
        self.locomotion_enabled = False

    def _log(self, event, **data):
        entry = dict(at=time.monotonic(), event=event, **data)
        with self._log_lock, (self._mission_dir / "events.jsonl").open("a") as stream:
            stream.write(json.dumps(entry, allow_nan=False, default=str) + "\n")

    def _rpc(self, method, params):
        rid = uuid.uuid4().hex
        result = self._client.request(
            method,
            params,
            session_id=self._session_id,
            lease_id=self._lease.lease_id if self._lease else None,
            request_id=rid,
        )
        self._log("request", request_id=rid, method=method, params=params)
        if result.error:
            raise RPCError(result.error["code"], result.error["message"])
        return result.result

    def _heartbeat(self):
        client = self.client_factory()
        try:
            if client.get_status().runtime_id != self._lease.runtime_id:
                raise MissionInterrupted("Executor restarted")
            while not self._stop.is_set():
                response = client.request(
                    "heartbeat",
                    {},
                    session_id=self._session_id,
                    lease_id=self._lease.lease_id,
                )
                if response.error or response.runtime_id != self._lease.runtime_id:
                    raise MissionInterrupted("Ownership heartbeat rejected")
                if self._stop.wait(self.profile.limits.heartbeat_s):
                    break
        except Exception as exc:
            self._heartbeat_error.put(exc)
        finally:
            client.close()

    def _check(self):
        if not self._heartbeat_error.empty():
            raise MissionInterrupted(f"Heartbeat failed: {self._heartbeat_error.get()}")
        status = self._client.get_status()
        if (
            status.runtime_id != self._lease.runtime_id
            or status.owner_session_id != self._session_id
        ):
            raise MissionInterrupted("Executor restarted or operator took control")
        self._log("status", **asdict(status))
        if status.phase in {"FAULT", "INTERRUPTED"}:
            raise MissionInterrupted(status.reason or status.phase)
        return status

    def _wait_complete(self, execution, timeout):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            status = self._check()
            if status.execution_id != execution.execution_id:
                raise MissionInterrupted("Execution changed")
            if status.phase == "COMPLETED":
                if not status.hold_confirmed:
                    raise MissionInterrupted(
                        "Completion without a confirmed planner hold"
                    )
                return
            time.sleep(0.02)
        raise MissionInterrupted("Motion completion deadline exceeded")

    def _manipulate(self, call):
        if self.monitor is None:
            raise ValueError("Manipulation requires a configured VLM monitor")
        initial = self._client.observe()
        execution = self._rpc("start_manipulation", {"skill_id": call.skill_id})
        if not isinstance(execution, Execution):
            raise ValueError("Expected manipulation execution")
        self._execution = execution
        skill = self.profile.require_skill(call.skill_id)
        if isinstance(self.monitor, G1CompletionMonitor):
            self.monitor.skill = skill
        self.monitor.begin(execution, initial)
        self._log("skill_start", prompt=skill.prompt, **asdict(execution))
        deadline = execution.started_at + self.profile.limits.manipulation_deadline_s
        next_check = time.monotonic()
        pending = None
        decisions = queue.Queue(maxsize=1)
        while time.monotonic() < deadline:
            status = self._check()
            if (
                status.execution_id != execution.execution_id
                or status.inference_epoch != execution.inference_epoch
                or status.phase != "MANIPULATING"
            ):
                raise MissionInterrupted("Manipulation state changed")
            if pending is not None and not pending.is_alive():
                decision, error = decisions.get_nowait()
                pending = None
                if error:
                    raise MissionInterrupted(f"Completion monitor error: {error}")
                self._log(
                    "monitor_decision",
                    **asdict(decision),
                    raw_response=self.monitor.raw_response,
                )
                if (
                    decision.execution_id != execution.execution_id
                    or decision.inference_epoch != execution.inference_epoch
                    or time.monotonic() - decision.captured_at
                    > self.profile.limits.decision_expiry_s
                ):
                    raise MissionInterrupted("Stale completion decision")
                if decision.outcome in {"failure", "unavailable"}:
                    raise MissionInterrupted(
                        decision.reason or "Operator inspection required"
                    )
                if decision.outcome == "complete":
                    paused = self._rpc(
                        "pause_manipulation", {"execution_id": execution.execution_id}
                    )
                    if not isinstance(paused, Execution) or paused.phase != "PAUSED":
                        raise MissionInterrupted(
                            "Pause did not acknowledge action invalidation"
                        )
                    reset = self._rpc(
                        "reset_standing",
                        {"execution_id": execution.execution_id, "open_hands": True},
                    )
                    self._execution = reset
                    self._wait_complete(reset, self.profile.limits.reset_deadline_s)
                    return
            if pending is None and time.monotonic() >= next_check:
                frame = self._client.observe()
                if frame.received_at >= execution.started_at:
                    path = self._mission_dir / (
                        hashlib.sha256(frame.frame_id.encode()).hexdigest() + ".jpg"
                    )
                    path.write_bytes(
                        base64.b64decode(frame.jpeg_rgb_b64, validate=True)
                    )
                    self._log(
                        "monitor_frame",
                        frame_id=frame.frame_id,
                        received_at=frame.received_at,
                        path=str(path),
                    )

                    def check(frame=frame):
                        try:
                            decisions.put((self.monitor.check(frame), None))
                        except Exception as exc:
                            decisions.put((None, exc))

                    pending = threading.Thread(target=check, daemon=True)
                    pending.start()
                    next_check = (
                        time.monotonic() + self.profile.limits.monitor_interval_s
                    )
            time.sleep(0.02)
        raise MissionInterrupted("Manipulation deadline exceeded")

    def _cancel_and_release(self):
        if self._lease is None:
            return
        try:
            status = self._client.get_status()
            if (
                status.runtime_id != self._lease.runtime_id
                or status.owner_session_id != self._session_id
            ):
                return
            if self._execution and status.phase not in {
                "COMPLETED",
                "INTERRUPTED",
                "FAULT",
            }:
                self._rpc("cancel", {"execution_id": self._execution.execution_id})
            deadline = time.monotonic() + self.profile.limits.planner_deadline_s
            while time.monotonic() < deadline:
                status = self._client.get_status()
                if status.hold_confirmed:
                    self._rpc("release_control", {})
                    return
                time.sleep(0.02)
        except Exception as exc:
            self._log("cleanup_unconfirmed", reason=str(exc))

    def run(self, call: SkillCall) -> MissionResult:
        return self.run_sequence([call])

    def run_sequence(self, calls: list[SkillCall]) -> MissionResult:
        if not calls or len(calls) > 64:
            raise ValueError("Sequence requires 1 to 64 skills")
        for call in calls:
            validate_skill_call(
                self.profile, call, locomotion_enabled=self.locomotion_enabled
            )
        self._session_id = uuid.uuid4().hex
        self._mission_dir = self.evidence_dir / self._session_id
        self._mission_dir.mkdir(parents=True, exist_ok=False)
        self._client, self._lease, self._execution = self.client_factory(), None, None
        self._stop, self._heartbeat_error = threading.Event(), queue.Queue()
        heartbeat = None
        outcome, reason = "fault", "Mission did not start"
        self._log(
            "mission_start",
            session_id=self._session_id,
            registry_sha256=self.profile.registry_sha256,
            checkpoint_expected=self.profile.checkpoint,
            skills=[asdict(c) for c in calls],
        )
        try:
            status = self._client.get_status()
            if status.registry_sha256 != self.profile.registry_sha256:
                raise ValueError("Executor profile mismatch")
            if not status.controller_running or status.owner_session_id is not None:
                raise ValueError("Operator must prepare a running paused executor")
            if (
                any(c.skill_id in {"walk_for", "turn_by"} for c in calls)
                and not status.locomotion_enabled
            ):
                raise ValueError("Native executor has locomotion disabled")
            if (
                any(c.skill_id in self.profile.skills for c in calls)
                and self.monitor is None
            ):
                raise ValueError("Manipulation requires a configured VLM monitor")
            ready_deadline = time.monotonic() + self.profile.limits.prewarm_deadline_s
            while not status.policy_ready and any(
                c.skill_id in self.profile.skills for c in calls
            ):
                if time.monotonic() >= ready_deadline:
                    raise TimeoutError("Policy prewarm unavailable")
                time.sleep(0.1)
                status = self._client.get_status()
            self._lease = self._rpc(
                "claim_control", {"registry_sha256": self.profile.registry_sha256}
            )
            if not isinstance(self._lease, Lease):
                raise ValueError("Expected control lease")
            heartbeat = threading.Thread(target=self._heartbeat, daemon=True)
            heartbeat.start()
            for call in calls:
                if call.skill_id in self.profile.skills:
                    self._manipulate(call)
                else:
                    params = (
                        {"execution_id": "", **call.params}
                        if call.skill_id == "reset_standing"
                        else call.params
                    )
                    self._execution = self._rpc(call.skill_id, params)
                    self._wait_complete(
                        self._execution,
                        self.profile.limits.reset_deadline_s
                        if call.skill_id == "reset_standing"
                        else self.profile.limits.turn_deadline_s,
                    )
            outcome, reason = (
                "completed",
                "All skills completed with confirmed planner hold",
            )
        except (MissionInterrupted, KeyboardInterrupt) as exc:
            outcome, reason = "interrupted", str(exc) or "Operator cancelled"
        except Exception as exc:
            outcome, reason = "fault", str(exc)
        finally:
            self._cancel_and_release()
            self._stop.set()
            if heartbeat:
                heartbeat.join(1.5)
            self._client.close()
        result = MissionResult(
            self._execution.execution_id if self._execution else "",
            outcome,
            reason,
            self._mission_dir,
        )
        self._log("mission_result", **asdict(result))
        (self._mission_dir / "result.json").write_text(
            json.dumps(asdict(result), default=str, indent=2) + "\n"
        )
        return result
