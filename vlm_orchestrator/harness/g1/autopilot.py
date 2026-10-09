"""One-lease stationary bottle handover coordinator."""

import base64
from dataclasses import asdict
import hashlib
import queue
import threading
import time

from .contract import Execution, SkillCall
from .handover_monitor import HandoverMonitor
from .runner import HarnessRunner, MissionInterrupted


class HandoverRunner(HarnessRunner):
    def _cancel_and_release(self):
        # A stopped autopilot must mark a paused offer interrupted before release.
        self._held_execution = None
        super()._cancel_and_release()

    def _validate_executor(self, status):
        if not status.ready_return_enabled or status.phase not in {"IDLE", "COMPLETED"}:
            raise ValueError(
                "Prepared native handover ready-return capability required"
            )

    def _wait_phase(self, phase, cycle, execution, native_phase, deadline=None):
        not_before = time.monotonic()
        initial = self._client.observe()
        self.monitor.begin_phase(phase, cycle, execution, initial, not_before)
        self._log(
            "handover_phase", handover_phase=phase, cycle_id=cycle, **asdict(execution)
        )
        next_check, pending, pending_frame = 0.0, None, None
        decisions = queue.Queue(maxsize=1)
        while deadline is None or time.monotonic() < deadline:
            status = self._check()
            if (
                (status.execution_id or "") != execution.execution_id
                or status.inference_epoch != execution.inference_epoch
                or status.phase != native_phase
            ):
                raise MissionInterrupted("Handover execution state changed")
            if (
                status.observation_age_s is None
                or not 0
                <= status.observation_age_s
                <= self.profile.limits.camera_max_age_s
            ):
                raise MissionInterrupted("Handover camera stream expired")
            if (
                not status.controller_running
                or status.telemetry_age_s is None
                or not 0
                <= status.telemetry_age_s
                <= self.profile.limits.feedback_max_age_s
                or (native_phase != "MANIPULATING" and not self._fresh_hold(status))
            ):
                raise MissionInterrupted("Handover lost fresh control or planner hold")
            if pending is not None and not pending.is_alive():
                decision, error = decisions.get_nowait()
                checked_frame = pending_frame
                pending_frame = None
                pending = None
                if error:
                    raise MissionInterrupted(f"Handover monitor error: {error}")
                self._log(
                    "handover_decision",
                    **asdict(decision),
                    raw_response=self.monitor.raw_response,
                )
                now = time.monotonic()
                if (
                    checked_frame is None
                    or decision.runtime_id != execution.runtime_id
                    or decision.execution_id != execution.execution_id
                    or decision.inference_epoch != execution.inference_epoch
                    or decision.cycle_id != cycle
                    or decision.phase != phase
                    or decision.frame_id != checked_frame.frame_id
                    or decision.captured_at != checked_frame.received_at
                    or not not_before
                    <= decision.captured_at
                    <= decision.decided_at
                    <= now
                    or now - decision.captured_at
                    > self.profile.limits.decision_expiry_s
                ):
                    raise MissionInterrupted("Stale or mismatched handover decision")
                if decision.outcome in {"failure", "unavailable"}:
                    raise MissionInterrupted(
                        decision.reason or "Handover inspection required"
                    )
                if decision.outcome == "complete":
                    if phase != "pick_offer" or status.right_hand_open is True:
                        return status
                    self.monitor.streak = 0
                    self._log("offering_not_open", cycle_id=cycle)
            if pending is None and time.monotonic() >= next_check:
                frame = self._client.observe()
                age = time.monotonic() - frame.received_at
                if not 0 <= age <= self.profile.limits.camera_max_age_s:
                    raise MissionInterrupted("Handover camera frame expired")
                if (
                    frame.received_at < not_before
                    or frame.frame_id in self.monitor.seen
                ):
                    # The native cache can legitimately repeat a fresh capture.
                    # Await a post-transition capture without counting old evidence.
                    next_check = time.monotonic() + 0.02
                    time.sleep(0.02)
                    continue
                pending_frame = frame
                path = self._mission_dir / (
                    hashlib.sha256(frame.frame_id.encode()).hexdigest() + ".jpg"
                )
                path.write_bytes(base64.b64decode(frame.jpeg_rgb_b64, validate=True))
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
                next_check = time.monotonic() + self.profile.limits.monitor_interval_s
            time.sleep(0.02)
        raise MissionInterrupted("Handover manipulation deadline exceeded")

    def _return_ready(self, execution_id):
        previous = self._check()
        self._held_execution = None
        reset = self._rpc("reset_ready", {"execution_id": execution_id})
        if (
            not isinstance(reset, Execution)
            or reset.runtime_id != previous.runtime_id
            or reset.phase != "RESETTING"
            or reset.inference_epoch <= previous.inference_epoch
        ):
            raise MissionInterrupted("Ready return did not invalidate previous actions")
        self._execution = reset
        self._wait_complete(reset, self.profile.limits.reset_deadline_s)
        return reset

    def _cycles(self, max_cycles):
        deadline = time.monotonic() + self.profile.limits.planner_deadline_s
        while True:
            status = self._check()
            if self._fresh_hold(status):
                break
            if time.monotonic() >= deadline:
                raise MissionInterrupted(
                    "Initial hold acknowledgement deadline exceeded"
                )
            time.sleep(0.02)
        initial = Execution(
            status.runtime_id,
            status.execution_id or "",
            "reset_ready",
            status.phase,
            status.inference_epoch,
            time.monotonic(),
            None,
        )
        # Do not open/reposition a hand that was already supporting an object.
        self._wait_phase("startup_empty", 1, initial, status.phase)
        ready = self._return_ready("")
        cycle = 1
        while max_cycles is None or cycle <= max_cycles:
            self._wait_phase("ready", cycle, ready, "COMPLETED")
            skill = self.profile.handover.skill_id
            execution, _ = self._start_manipulation(SkillCall(skill, {}))
            if not isinstance(execution, Execution):
                raise MissionInterrupted("Expected handover policy execution")
            self._execution = execution
            status = self._wait_phase(
                "pick_offer",
                cycle,
                execution,
                "MANIPULATING",
                execution.started_at + self.profile.limits.manipulation_deadline_s,
            )
            paused = self._rpc(
                "pause_manipulation", {"execution_id": execution.execution_id}
            )
            if (
                not isinstance(paused, Execution)
                or paused.phase != "PAUSED"
                or paused.runtime_id != execution.runtime_id
                or paused.execution_id != execution.execution_id
                or paused.inference_epoch <= execution.inference_epoch
            ):
                raise MissionInterrupted(
                    "Handover pause did not invalidate policy actions"
                )
            self._execution = paused
            self._wait_paused_hold(paused, status.telemetry_index)
            self._held_execution = paused
            self._wait_phase("wait_empty", cycle, paused, "PAUSED")
            ready = self._return_ready(paused.execution_id)
            self._log("handover_cycle_complete", cycle_id=cycle)
            cycle += 1
        return f"Completed {cycle - 1} handover cycles with confirmed ready hold"

    def run_autopilot(self, max_cycles=None):
        if max_cycles is not None and (type(max_cycles) is not int or max_cycles < 1):
            raise ValueError("max_cycles must be a positive integer or None")
        h = self.profile.handover
        if (
            h is None
            or not h.checkpoint_verified
            or not h.ready_pose_reviewed
            or not isinstance(self.monitor, HandoverMonitor)
            or self.locomotion_enabled
        ):
            raise ValueError(
                "Verified stationary handover profile and phase monitor required"
            )
        return self._run_session(
            [SkillCall(h.skill_id, {})], lambda _: self._cycles(max_cycles)
        )
