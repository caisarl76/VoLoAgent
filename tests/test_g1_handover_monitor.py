from dataclasses import replace
import importlib
import json

import pytest

from test_g1_monitor import snapshot
from vlm_orchestrator.harness.g1.contract import Execution
from vlm_orchestrator.harness.g1.registry import HarnessLimits


def make(responses, phase="pick_offer"):
    module = importlib.util.find_spec("vlm_orchestrator.harness.g1.handover_monitor")
    assert module is not None, "Phase-specific handover monitor missing"
    cls = importlib.import_module(module.name).HandoverMonitor
    clock = [1.0]
    replies = iter(responses)

    def call(*_):
        result = next(replies)
        return result(clock) if callable(result) else result

    m = cls(call, lambda: clock[0], HarnessLimits(monitor_interval_s=0.01))
    execution = Execution(
        "boot", "mission", "bottle_handover", "MANIPULATING", 2, 0.9, None
    )
    m.begin_phase(phase, 1, execution, snapshot("initial", 0.9), 1.0)
    return m, clock, execution


def reply(outcome, reason="test observation"):
    return json.dumps(dict(outcome=outcome, reason=reason))


@pytest.mark.parametrize(
    "phase", ["startup_empty", "ready", "pick_offer", "wait_empty"]
)
def test_two_distinct_yes_frames_required_for_each_phase(phase):
    m, clock, _ = make([reply("yes"), reply("yes")], phase)
    assert m.check(snapshot("a")).outcome == "in_progress"
    clock[0] += 0.1
    d = m.check(snapshot("b", clock[0]))
    assert d.outcome == "complete"
    assert (d.runtime_id, d.cycle_id, d.phase) == ("boot", 1, phase)


@pytest.mark.parametrize("middle", ["no", "unknown"])
def test_occlusion_or_still_held_bottle_clears_confirmation(middle):
    m, clock, _ = make(
        [reply("yes"), reply(middle), reply("yes"), reply("yes")], "wait_empty"
    )
    outcomes = []
    for i in range(4):
        clock[0] = 1.0 + i * 0.1
        outcomes.append(m.check(snapshot(str(i), clock[0])).outcome)
    assert outcomes == ["in_progress"] * 3 + ["complete"]


def test_roll_then_human_rescue_is_failure_even_in_empty_phase():
    m, clock, execution = make(
        [reply("failure", "Bottle rolling off tilted palm; human rescue"), reply("yes")]
    )
    assert m.check(snapshot("roll")).outcome == "failure"
    clock[0] = 1.1
    paused = replace(execution, phase="PAUSED", inference_epoch=3)
    m.begin_phase("wait_empty", 1, paused, snapshot("rescue", 1.1, epoch=3), 1.1)
    assert m.check(snapshot("empty", 1.1, epoch=3)).outcome == "failure"
    clock[0] = 1.2
    m.begin_phase("ready", 2, paused, snapshot("new", 1.2, epoch=3), 1.2)
    assert m.check(snapshot("new-bottle", 1.2, epoch=3)).outcome == "in_progress"


@pytest.mark.parametrize(
    "bad",
    [
        lambda s: replace(s, runtime_id="old"),
        lambda s: replace(s, inference_epoch=1),
        lambda s: replace(s, execution_id="old"),
        lambda s: replace(s, received_at=0.4),
        lambda s: replace(s, camera_key="other"),
    ],
)
def test_stale_or_wrong_identity_cannot_confirm(bad):
    m, _, _ = make([reply("yes")])
    assert m.check(bad(snapshot("bad"))).outcome == "unavailable"


def test_repeated_frame_clears_confirmation():
    m, clock, _ = make([reply("yes"), reply("yes")])
    m.check(snapshot("a"))
    clock[0] = 1.1
    assert m.check(snapshot("a", 1.1)).outcome == "unavailable"
    assert m.check(snapshot("b", 1.1)).outcome == "in_progress"


@pytest.mark.parametrize(
    "raw",
    [
        "bad-json",
        "[]",
        '{"outcome":[],"reason":"bad"}',
        '{"outcome":"yes","reason":"x","extra":1}',
        '{"outcome":"YES","reason":"x"}',
    ],
)
def test_malformed_response_rejected(raw):
    m, _, _ = make([raw])
    assert m.check(snapshot("a")).outcome == "unavailable"


def test_expired_vision_result_rejected():
    def delayed(clock):
        clock[0] += 11.0
        return reply("yes")

    m, _, _ = make([delayed])
    assert m.check(snapshot("a")).outcome == "unavailable"


def test_wait_empty_keeps_offer_image_but_uses_fresh_paused_confirmations():
    import base64
    import io
    from PIL import Image

    m, clock, execution = make([reply("yes"), reply("yes")])
    image = io.BytesIO()
    Image.new("RGB", (8, 8), (200, 0, 0)).save(image, format="JPEG")
    anchor = replace(
        snapshot("offering"), jpeg_rgb_b64=base64.b64encode(image.getvalue()).decode()
    )
    m.check(anchor)
    clock[0] = 1.1
    anchor = replace(
        anchor, frame_id="offer-confirmed", received_at=1.1, source_timestamp=1.1
    )
    assert m.check(anchor).outcome == "complete"
    clock[0] = 1.2
    paused = replace(execution, phase="PAUSED", inference_epoch=3)
    m.begin_phase("wait_empty", 1, paused, snapshot("already-empty", 1.2, epoch=3), 1.2)
    contents = []

    def vision(_, content):
        contents.extend(content)
        return reply("yes")

    m.vlm_call_fn = vision
    assert m.check(snapshot("empty-1", 1.2, epoch=3)).outcome == "in_progress"
    assert any(
        part.get("image_url", {}).get("url")
        == "data:image/jpeg;base64," + anchor.jpeg_rgb_b64
        for part in contents
    )
    clock[0] = 1.3
    assert m.check(snapshot("empty-2", 1.3, epoch=3)).outcome == "complete"
