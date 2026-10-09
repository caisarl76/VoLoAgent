"""Phase judgments for stationary handover, with cycle-persistent failures."""

from dataclasses import dataclass

from vlm_orchestrator.vlm.api import parse_json

from .monitor import CompletionDecision, G1CompletionMonitor, decode_image
from .registry import SkillDefinition


CRITERIA = {
    "ready": (
        "Is a drink bottle clearly on the reachable desk working surface, available "
        "to pick? A bottle in a person's hand or on the robot hand is NO. "
        "Occlusion or uncertain reachability is UNKNOWN."
    ),
    "pick_offer": (
        "Has the robot lifted the bottle from the desk and offered it resting "
        "stably on its open RIGHT palm? The palm must face up and be level enough "
        "that the bottle is supported without rolling. Compare frames for stability. "
        "Closed fingers, a bottle still on the desk, an unstable tilt or a bottle "
        "supported by a person are not a completed offer."
    ),
    "wait_empty": (
        "Was the bottle deliberately taken from the previously offered RIGHT hand "
        "by the person, and is that robot hand now clearly visible and empty, with "
        "no human contact obstructing its return? A still-supported bottle is NO; "
        "an occluded hand or unclear removal is UNKNOWN. Empty alone does not prove "
        "successful removal: check earlier frames for rolling or falling."
    ),
}
SYSTEM = (
    "Inspect chronological robot ego-camera frames for a bottle handover. "
    "Answer the supplied phase criterion. Any visible bottle slipping, rolling "
    "off the robot palm, falling, collision or human rescue is FAILURE, even "
    "if the person subsequently catches it and the robot hand becomes empty. "
    "Do not infer stable support from open fingers alone. Unknown is not success. "
    'Return ONLY JSON {"outcome":"yes|no|unknown|failure","reason":"..."}.'
)


@dataclass(frozen=True)
class HandoverDecision(CompletionDecision):
    runtime_id: str
    cycle_id: int
    phase: str


class HandoverMonitor(G1CompletionMonitor):
    def __init__(self, vlm_call_fn, now_fn, limits=None):
        super().__init__(SkillDefinition("bottle_handover", "handover", "phase criterion"),
                         vlm_call_fn, now_fn, limits)
        self.cycle_id = 0
        self.failure = None

    def begin_phase(self, phase, cycle_id, execution, initial, not_before):
        if phase not in CRITERIA or type(cycle_id) is not int or cycle_id < 1:
            raise ValueError("Invalid handover phase/cycle")
        if cycle_id < self.cycle_id or initial.runtime_id != execution.runtime_id:
            raise ValueError("Old cycle or wrong initial runtime")
        if cycle_id != self.cycle_id:
            self.failure = None
        self.phase, self.cycle_id, self.execution = phase, cycle_id, execution
        self.initial, self.previous, self.not_before = initial, None, not_before
        self.streak, self.seen, self.last_check = 0, set(), None

    def _decision(self, snapshot, outcome, reason):
        return HandoverDecision(outcome, reason, snapshot.frame_id,
            snapshot.execution_id or "", snapshot.inference_epoch,
            snapshot.received_at, self.now(), snapshot.runtime_id,
            self.cycle_id, self.phase)

    def check(self, snapshot):
        if self.execution is None:
            raise RuntimeError("Handover phase not started")
        now = self.now()
        if not (
            snapshot.runtime_id == self.execution.runtime_id
            and snapshot.execution_id == self.execution.execution_id
            and snapshot.inference_epoch == self.execution.inference_epoch
            and snapshot.camera_key == "ego_view"
            and self.not_before <= snapshot.received_at <= now
            and now - snapshot.received_at <= self.limits.camera_max_age_s
            and snapshot.frame_id not in self.seen
        ):
            self.streak = 0
            return self._decision(snapshot, "unavailable", "stale_or_repeated_frame_or_execution")
        if self.failure:
            return self._decision(snapshot, "failure", self.failure)
        if self.last_check is not None and now-self.last_check < self.limits.monitor_interval_s:
            return self._decision(snapshot, "in_progress", "check_not_due")
        self.seen.add(snapshot.frame_id)
        self.last_check, self.raw_response = now, None
        self._call_deadline = snapshot.received_at + self.limits.decision_expiry_s
        content = [{"type": "text", "text": CRITERIA[self.phase]}]
        frames = [("PHASE START", self.initial)]
        if self.previous is not None:
            frames.append(("PREVIOUS", self.previous))
        frames.append(("NOW", snapshot))
        try:
            for label, frame in frames:
                # Validate the image before exposing it to the provider.
                decode_image(frame)
                content.extend([
                    {"type": "text", "text": f"{label}: {frame.source_timestamp:.3f}s"},
                    {"type": "image_url", "image_url": {
                        "url": "data:image/jpeg;base64," + frame.jpeg_rgb_b64}},
                ])
            raw = self._bounded_call(SYSTEM, content)
            data = parse_json(raw)
            if (not isinstance(data, dict) or data.keys() != {"outcome", "reason"}
                or type(data["outcome"]) is not str
                or data["outcome"] not in {"yes", "no", "unknown", "failure"}
                or type(data["reason"]) is not str
                or self.now()-snapshot.received_at > self.limits.decision_expiry_s):
                raise ValueError("Invalid or expired handover judgment")
        except Exception as exc:
            self.streak = 0
            return self._decision(snapshot, "unavailable", f"vision_unavailable:{type(exc).__name__}")
        self.previous = snapshot
        if data["outcome"] == "failure":
            self.streak = 0
            self.failure = data["reason"] or "Observed handover failure"
            return self._decision(snapshot, "failure", self.failure)
        self.streak = self.streak+1 if data["outcome"] == "yes" else 0
        return self._decision(snapshot, "complete" if self.streak >= 2 else "in_progress", data["reason"])
