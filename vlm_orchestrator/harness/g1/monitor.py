"""Fresh-frame completion confirmation around the existing VLM handler."""

from __future__ import annotations

import base64
from dataclasses import dataclass
import io
import queue
import threading
from typing import Callable, Literal

import numpy as np
from PIL import Image

from vlm_orchestrator.failure_handlers.vlm import VLMFailureHandler
from vlm_orchestrator.strategies.base import SessionState
from vlm_orchestrator.vlm.api import encode_image_b64, parse_json

from .contract import Execution, ObservationSnapshot
from .registry import HarnessLimits, SkillDefinition


@dataclass(frozen=True)
class CompletionDecision:
    outcome: Literal["in_progress", "complete", "failure", "unavailable"]
    reason: str
    frame_id: str
    execution_id: str
    inference_epoch: int
    captured_at: float
    decided_at: float


def decode_image(snapshot):
    with Image.open(
        io.BytesIO(base64.b64decode(snapshot.jpeg_rgb_b64, validate=True))
    ) as image:
        return np.asarray(image.convert("RGB"))


class G1CompletionMonitor:
    def __init__(
        self,
        skill: SkillDefinition,
        vlm_call_fn: Callable,
        now_fn: Callable[[], float],
        limits: HarnessLimits | None = None,
    ):
        self.skill, self.vlm_call_fn, self.now = skill, vlm_call_fn, now_fn
        self.limits = limits or HarnessLimits()
        self.execution = None
        self.streak = 0
        self.seen = set()
        self.last_check = None
        self.raw_response = None
        self._call_thread = None
        self._call_deadline = 0.0
        self._handler = VLMFailureHandler(
            self._bounded_call,
            self._image_message,
            lambda obs: obs["image"],
            lambda _: None,
            check_interval=1,
            recovery_mode="replan",
            primary_label="G1 ego camera",
        )

    def _bounded_call(self, system, content):
        if self._call_thread is not None and self._call_thread.is_alive():
            raise TimeoutError("Previous VLM call still running")
        replies = queue.Queue(maxsize=1)

        def call():
            try:
                replies.put((self.vlm_call_fn(system, content), None))
            except Exception as exc:
                replies.put((None, exc))

        self._call_thread = threading.Thread(target=call, daemon=True)
        self._call_thread.start()
        remaining = min(
            self.limits.monitor_deadline_s, self._call_deadline - self.now()
        )
        if remaining <= 0:
            raise TimeoutError("Image decision expired")
        try:
            raw, error = replies.get(timeout=remaining)
        except queue.Empty as exc:
            raise TimeoutError("VLM call deadline exceeded") from exc
        if error:
            raise error
        if not isinstance(raw, str) or not isinstance(parse_json(raw), dict):
            raise ValueError("VLM response must be a JSON object")
        self.raw_response = raw
        return raw

    def _image_message(self, text, initial, current, extra, **kwargs):
        content = [
            {
                "type": "text",
                "text": f"{text}\nCompletion criterion: {self.skill.completion_criteria}",
            }
        ]
        for label, image in [("BEFORE", initial), ("NOW", current)]:
            content.append({"type": "text", "text": label})
            content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{encode_image_b64(image)}"
                    },
                }
            )
        return content

    def begin(self, execution: Execution, initial: ObservationSnapshot) -> None:
        if initial.runtime_id != execution.runtime_id:
            raise ValueError("Initial image belongs to a different executor")
        self.execution = execution
        self.state = SessionState(
            original_instruction=self.skill.prompt,
            subgoals=[self.skill.prompt],
            initial_image=decode_image(initial),
        )
        self._handler.on_episode_start({}, self.state)
        self.streak, self.seen, self.last_check = 0, set(), None

    def _decision(self, snapshot, outcome, reason):
        return CompletionDecision(
            outcome,
            reason,
            snapshot.frame_id,
            snapshot.execution_id or "",
            snapshot.inference_epoch,
            snapshot.received_at,
            self.now(),
        )

    def check(self, snapshot: ObservationSnapshot) -> CompletionDecision:
        now = self.now()
        if self.execution is None:
            raise RuntimeError("Monitor mission not started")
        valid = (
            snapshot.runtime_id == self.execution.runtime_id
            and snapshot.execution_id == self.execution.execution_id
            and snapshot.inference_epoch == self.execution.inference_epoch
            and snapshot.camera_key == "ego_view"
            and self.execution.started_at <= snapshot.received_at <= now
            and now - snapshot.received_at <= self.limits.camera_max_age_s
            and snapshot.frame_id not in self.seen
        )
        if not valid:
            self.streak = 0
            return self._decision(
                snapshot, "unavailable", "stale_or_repeated_frame_or_execution"
            )
        if (
            self.last_check is not None
            and now - self.last_check < self.limits.monitor_interval_s
        ):
            return self._decision(snapshot, "in_progress", "check_not_due")
        self.seen.add(snapshot.frame_id)
        self.last_check = now
        self.raw_response = None
        self._call_deadline = snapshot.received_at + self.limits.decision_expiry_s
        self.state.episode_step += 1
        self.state.infer_count += 1
        self.state.vlm_check_result = None
        try:
            result = self._handler.step({"image": decode_image(snapshot)}, self.state)
        except (ValueError, OSError, TypeError):
            self.streak = 0
            return self._decision(snapshot, "unavailable", "invalid_image_or_response")
        if (
            self.now() - snapshot.received_at > self.limits.decision_expiry_s
            or self.raw_response is None
        ):
            self.streak = 0
            return self._decision(
                snapshot, "unavailable", "vlm_error_or_expired_decision"
            )
        check = self.state.vlm_check_result
        if (
            result is not None
            and result.status == "complete"
            and result.action == "next"
        ):
            self.streak += 1
            return self._decision(
                snapshot,
                "complete" if self.streak >= 2 else "in_progress",
                result.reason,
            )
        self.streak = 0
        if result is not None and (
            result.status == "failure" or result.action == "replan"
        ):
            return self._decision(
                snapshot, "failure", result.reason or "operator_inspection_required"
            )
        if check and check.get("reason") == "parse_failure":
            return self._decision(snapshot, "unavailable", "parse_failure")
        return self._decision(
            snapshot, "in_progress", check.get("reason", "") if check else ""
        )
