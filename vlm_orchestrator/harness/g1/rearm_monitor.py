"""VLM checks around a trial: before it, the case is empty, then a bottle stands upright with no hand ("empty",
"bottle"); after it, the bottle is off the robot's hand ("taken", R24) before the arm returns to ready."""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Callable, Literal

from vlm_orchestrator.vlm.api import encode_image_b64

from .contract import ObservationSnapshot
from .monitor import G1CompletionMonitor, decode_image
from .registry import HarnessLimits

# Verbatim PROMPT of docs/artifacts/g1_vlm_upright_check_20261026/upright_check.py (17/17). Adding case_empty to
# it degraded the upright answers, so the empty case is asked separately with EMPTY_PROMPT. A bare "case_empty"
# question answered false on all 17 frames (the lid's handle and labels); naming them as part of the case
# separated the 17 frames (2026-10-26, Qwen3-VL-2B on the local server).
PROMPT = ("You see the head camera of a humanoid robot looking down at a black case. Answer only with JSON "
          '{"bottle_upright_on_case": true|false, "hand_at_bottle": true|false}. bottle_upright_on_case is true only if a '
          "drink bottle stands upright on top of the black case (not lying down, not held, not on the floor). "
          "hand_at_bottle is true if any robot or human hand touches or holds a bottle.")
EMPTY_PROMPT = (
    "You see the head camera of a humanoid robot looking down at a black case. The case has a built-in carrying "
    "handle and white labels on its lid; they are part of the case. Answer only with JSON "
    '{"object_on_case": true|false}. object_on_case is true if a bottle or any other loose object stands or lies '
    "on top of the black case."
)
# R24: tested on candidate-A trial frames (on-hand 16/17 true, empty-reach 27/30 false); offer-pose empty-hand frames
# are still to collect on the robot.
TAKEN_PROMPT = ('You see the head camera of a humanoid robot. Answer only with JSON {"bottle_on_robot_hand": true|false}. '
                "bottle_on_robot_hand is true if a drink bottle rests on or is held by the robot's own hand (the black and silver "
                "metal gripper), including while a person touches it; false if the robot's hand is empty or not visible.")


@dataclass(frozen=True)
class RearmDecision:
    outcome: Literal["yes", "no", "unknown", "failure"]
    reason: str
    frame_id: str
    step: str
    complete: bool = False  # second "yes" from a distinct frame
    answers: dict | None = None  # {"upright_check": {"raw", "parsed"}, "object_on_case": {...}} for the evidence log


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
        def recorded(system, content):  # keeps a reply the base rejects as malformed, for the evidence
            self._last_raw = vlm_call_fn(system, content)
            return self._last_raw

        super().__init__(None, recorded, now_fn, limits)
        self.step, self._last_raw = None, None

    def begin(self, step: Literal["empty", "bottle", "taken"], initial: ObservationSnapshot) -> None:
        if step not in ("empty", "bottle", "taken"):
            raise ValueError(f"Unknown re-arm step: {step!r}")
        with self._session_lock:
            self.step, self.streak, self.seen = step, 0, set()

    def _check(self, snapshot: ObservationSnapshot) -> RearmDecision:
        answers = {}

        def out(outcome, reason, complete=False):
            return RearmDecision(outcome, reason, snapshot.frame_id, self.step, complete, answers)

        def ask(name, prompt, fields):
            """Returns (parsed values, None) or (None, failing decision); a failure resets the streak."""
            content = [  # image first, as in the validated upright_check run
                {"type": "image_url", "image_url": {"url": image_url}},
                {"type": "text", "text": prompt},
            ]
            self._last_raw = None
            try:
                raw = self._bounded_call("", content)
            except ValueError:  # _bounded_call: the reply is not a JSON object
                self.streak = 0
                answers[name] = {"raw": self._last_raw, "parsed": None}
                return None, out("unknown", f"{name}: malformed_reply")
            except Exception as exc:  # timeout or VLM error
                self.streak = 0
                return None, out("failure", f"{name}: {type(exc).__name__}: {exc}")
            answers[name] = {"raw": raw, "parsed": None}
            try:
                parsed = json.loads(raw[raw.index("{") : raw.rindex("}") + 1])
                values = tuple(parsed[k] for k in fields)
                if not all(isinstance(v, bool) for v in values):
                    raise ValueError
            except (ValueError, KeyError, TypeError):
                self.streak = 0
                return None, out("unknown", f"{name}: malformed_reply")
            answers[name]["parsed"] = parsed
            return values, None

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
        if self.step == "taken":  # one call; yes iff bottle_on_robot_hand is exactly False
            values, bad = ask("taken", TAKEN_PROMPT, ("bottle_on_robot_hand",))
            if bad:
                return bad
            ok = values[0] is False
        else:
            values, bad = ask("upright_check", PROMPT, ("bottle_upright_on_case", "hand_at_bottle"))
            if bad:
                return bad
            upright, hand = values
            ok = (upright and not hand) if self.step == "bottle" else (not upright and not hand)
            if ok and self.step == "empty":  # an obvious "no" above costs a single call
                values, bad = ask("object_on_case", EMPTY_PROMPT, ("object_on_case",))
                if bad:
                    return bad
                ok = not values[0]
        if ok:
            self.streak += 1
            return out("yes", self.step, self.streak >= 2)
        self.streak = 0
        return out("no", self.step)
