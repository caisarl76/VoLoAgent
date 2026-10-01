import base64
import importlib
import importlib.util
import io
import json

from PIL import Image
import pytest

from vlm_orchestrator.harness.g1.contract import Execution, ObservationSnapshot
from vlm_orchestrator.harness.g1.registry import SkillDefinition


def snapshot(frame, now=1.0, epoch=2, execution="mission"):
    image = io.BytesIO()
    Image.new("RGB", (8, 8)).save(image, format="JPEG")
    return ObservationSnapshot(
        "boot",
        execution,
        epoch,
        "ego_view",
        now,
        frame,
        now,
        0.0,
        base64.b64encode(image.getvalue()).decode(),
    )


def monitor(responses):
    assert importlib.util.find_spec("vlm_orchestrator.harness.g1.monitor"), (
        "G1 monitor missing"
    )
    cls = importlib.import_module(
        "vlm_orchestrator.harness.g1.monitor"
    ).G1CompletionMonitor
    clock = [1.0]
    iterator = iter(responses)

    def vlm(*args):
        result = next(iterator)
        if isinstance(result, Exception):
            raise result
        if callable(result):
            return result(clock)
        return result

    m = cls(
        SkillDefinition("bottle", "exact trained prompt", "Bottle released on table"),
        vlm,
        lambda: clock[0],
    )
    m.begin(
        Execution("boot", "mission", "bottle", "MANIPULATING", 2, 1.0, None),
        snapshot("initial", 0.9),
    )
    return m, clock


COMPLETE = json.dumps(
    {"status": "complete", "action": "next", "reason": "placed and released"}
)


def test_two_distinct_complete_frames_required():
    m, clock = monitor([COMPLETE, COMPLETE])
    assert m.check(snapshot("one")).outcome == "in_progress"
    clock[0] = 2.0
    assert m.check(snapshot("two", 2.0)).outcome == "complete"


def test_repeated_frame_cannot_confirm():
    m, clock = monitor([COMPLETE, COMPLETE])
    assert m.check(snapshot("one")).outcome == "in_progress"
    clock[0] = 2.0
    assert m.check(snapshot("one", 2.0)).outcome == "unavailable"
    clock[0] = 3.0
    assert m.check(snapshot("two", 3.0)).outcome == "in_progress"


@pytest.mark.parametrize(
    "raw",
    [
        "bad-json",
        "[]",
        "null",
        json.dumps({"status": "complete", "action": "continue", "confidence": 1.0}),
        RuntimeError("API down"),
    ],
)
def test_parse_error_or_api_error_clears_streak(raw):
    m, clock = monitor([COMPLETE, raw, COMPLETE])
    m.check(snapshot("one"))
    clock[0] = 2.0
    assert m.check(snapshot("two", 2.0)).outcome != "complete"
    clock[0] = 3.0
    assert m.check(snapshot("three", 3.0)).outcome == "in_progress"


def test_stale_or_previous_epoch_cannot_complete():
    m, clock = monitor([COMPLETE])
    assert m.check(snapshot("one", epoch=1)).outcome == "unavailable"
    assert m.check(snapshot("one", execution="old")).outcome == "unavailable"
    clock[0] = 2.0
    assert m.check(snapshot("one", 1.0)).outcome == "unavailable"


def test_late_decision_discarded():
    def late(clock):
        clock[0] += 10.1
        return COMPLETE

    m, _ = monitor([late])
    assert m.check(snapshot("one")).outcome == "unavailable"


def test_replan_instruction_never_reaches_policy():
    m, _ = monitor(
        [
            json.dumps(
                {
                    "status": "failure",
                    "action": "replan",
                    "instruction": "invented grasp",
                }
            )
        ]
    )
    decision = m.check(snapshot("one"))
    assert decision.outcome == "failure"
    assert not hasattr(decision, "instruction")


def test_image_message_uses_jpeg_mime():
    m, _ = monitor([COMPLETE])
    content = m._image_message("criterion", decode_initial(), decode_initial(), None)
    assert all(
        item["image_url"]["url"].startswith("data:image/jpeg;base64,")
        for item in content
        if item["type"] == "image_url"
    )


def decode_initial():
    import numpy as np

    return np.zeros((4, 4, 3), np.uint8)
