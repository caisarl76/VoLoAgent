"""Re-arm checks before a trial: the case is empty, then a bottle stands upright with no hand."""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Callable, Literal

from vlm_orchestrator.vlm.api import encode_image_b64

from .contract import ObservationSnapshot
from .monitor import G1CompletionMonitor, decode_image
from .registry import HarnessLimits

# Wording from docs/artifacts/g1_vlm_upright_check_20261026/upright_check.py (17/17), plus case_empty.
PROMPT = (
    "You see the head camera of a humanoid robot looking down at a black case. Answer only with JSON "
    '{"case_empty": true|false, "bottle_upright_on_case": true|false, "hand_at_bottle": true|false}. '
    "bottle_upright_on_case is true only if a "
    "drink bottle stands upright on top of the black case (not lying down, not held, not on the floor). "
    "hand_at_bottle is true if any robot or human hand touches or holds a bottle. "
    "case_empty is true only if nothing stands or lies on top of the black case."
)
FIELDS = ("case_empty", "bottle_upright_on_case", "hand_at_bottle")


@dataclass(frozen=True)
class RearmDecision:
    outcome: Literal["yes", "no", "unknown", "failure"]
    reason: str
    frame_id: str
    step: str
    complete: bool = False  # second "yes" from a distinct frame


class RearmMonitor(G1CompletionMonitor):
    """Reuses the monitor's bounded VLM call; sends the current frame only, never a BEFORE image.

    The base's _begin and _image_message must never be reached from this subclass (skill is None).
    A ValueError from the VLM client is labelled "unknown" (malformed_reply), not "failure".
    """

    def __init__(
        self,
        vlm_call_fn: Callable,
        now_fn: Callable[[], float],
        limits: HarnessLimits | None = None,
    ):
        super().__init__(None, vlm_call_fn, now_fn, limits)
        self.step = None

    def begin(self, step: Literal["empty", "bottle"], initial: ObservationSnapshot) -> None:
        if step not in ("empty", "bottle"):
            raise ValueError(f"Unknown re-arm step: {step!r}")
        with self._session_lock:
            self.step, self.streak, self.seen = step, 0, set()

    def _check(self, snapshot: ObservationSnapshot) -> RearmDecision:
        def out(outcome, reason, complete=False):
            return RearmDecision(outcome, reason, snapshot.frame_id, self.step, complete)

        if self.step is None:
            raise RuntimeError("Monitor step not started")
        if self.now() - snapshot.received_at > self.limits.camera_max_age_s:
            return out("unknown", "stale_frame")
        if snapshot.frame_id in self.seen:  # not a new answer: leave the streak alone
            return out("unknown", "repeated_frame")
        self.seen.add(snapshot.frame_id)
        self._call_deadline = snapshot.received_at + self.limits.decision_expiry_s
        try:
            image_url = f"data:image/jpeg;base64,{encode_image_b64(decode_image(snapshot))}"
        except (ValueError, OSError, TypeError):
            self.streak = 0
            return out("unknown", "invalid_image")
        content = [  # image first, as in the validated upright_check run
            {"type": "image_url", "image_url": {"url": image_url}},
            {"type": "text", "text": PROMPT},
        ]
        try:
            raw = self._bounded_call("", content)
        except ValueError:  # _bounded_call: the reply is not a JSON object
            self.streak = 0
            return out("unknown", "malformed_reply")
        except Exception as exc:  # timeout or VLM error
            self.streak = 0
            return out("failure", f"{type(exc).__name__}: {exc}")
        try:
            answer = json.loads(raw[raw.index("{") : raw.rindex("}") + 1])
            case_empty, upright, hand = (answer[k] for k in FIELDS)
            if not all(isinstance(v, bool) for v in (case_empty, upright, hand)):
                raise ValueError
        except (ValueError, KeyError, TypeError):
            self.streak = 0
            return out("unknown", "malformed_reply")
        if (case_empty and not upright) if self.step == "empty" else (upright and not hand):
            self.streak += 1
            return out("yes", self.step, self.streak >= 2)
        self.streak = 0
        return out("no", self.step)
