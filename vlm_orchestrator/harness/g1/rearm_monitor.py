"""Re-arm checks before a trial: the case is empty, then a bottle stands upright with no hand."""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Callable, Literal

from vlm_orchestrator.vlm.api import encode_image_b64

from .contract import ObservationSnapshot
from .monitor import G1CompletionMonitor, decode_image
from .registry import HarnessLimits

# Verbatim PROMPT of docs/artifacts/g1_vlm_upright_check_20261026/upright_check.py (17/17). Adding case_empty to
# it degraded the upright answers, so case_empty is asked separately with EMPTY_PROMPT.
PROMPT = ("You see the head camera of a humanoid robot looking down at a black case. Answer only with JSON "
          '{"bottle_upright_on_case": true|false, "hand_at_bottle": true|false}. bottle_upright_on_case is true only if a '
          "drink bottle stands upright on top of the black case (not lying down, not held, not on the floor). "
          "hand_at_bottle is true if any robot or human hand touches or holds a bottle.")
EMPTY_PROMPT = (
    'Answer only with JSON {"case_empty": true|false}. '
    "case_empty is true only if nothing stands or lies on top of the black case."
)


@dataclass(frozen=True)
class RearmDecision:
    outcome: Literal["yes", "no", "unknown", "failure"]
    reason: str
    frame_id: str
    step: str
    complete: bool = False  # second "yes" from a distinct frame
    answers: dict | None = None  # {"upright_check": {"raw", "parsed"}, "case_empty": {...}} for the evidence log


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
        answers = {}

        def out(outcome, reason, complete=False):
            return RearmDecision(outcome, reason, snapshot.frame_id, self.step, complete, answers)

        def ask(name, prompt, fields):
            """Returns (parsed values, None) or (None, failing decision); a failure resets the streak."""
            content = [  # image first, as in the validated upright_check run
                {"type": "image_url", "image_url": {"url": image_url}},
                {"type": "text", "text": prompt},
            ]
            try:
                raw = self._bounded_call("", content)
            except ValueError:  # _bounded_call: the reply is not a JSON object
                self.streak = 0
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
        values, bad = ask("upright_check", PROMPT, ("bottle_upright_on_case", "hand_at_bottle"))
        if bad:
            return bad
        upright, hand = values
        if self.step == "bottle":
            ok = upright and not hand
        else:
            ok = not upright and not hand
            if ok:  # an obvious "no" above costs a single call
                values, bad = ask("case_empty", EMPTY_PROMPT, ("case_empty",))
                if bad:
                    return bad
                ok = values[0]
        if ok:
            self.streak += 1
            return out("yes", self.step, self.streak >= 2)
        self.streak = 0
        return out("no", self.step)
