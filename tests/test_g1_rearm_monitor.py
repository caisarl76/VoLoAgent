import json
import time
from dataclasses import replace

import pytest

from test_g1_monitor import snapshot
from vlm_orchestrator.harness.g1.registry import HarnessLimits
from vlm_orchestrator.harness.g1.rearm_monitor import PROMPT, RearmMonitor


def answer(empty=False, upright=False, hand=False):
    return json.dumps(
        {"case_empty": empty, "bottle_upright_on_case": upright, "hand_at_bottle": hand}
    )


def make(replies, step="empty", limits=None):
    queue, calls = list(replies), []

    def vlm(system, content):
        calls.append(content)
        r = queue.pop(0)
        if isinstance(r, Exception):
            raise r
        return r

    m = RearmMonitor(vlm, lambda: 1.0, limits)
    m.begin(step, snapshot("init"))
    return m, calls


def run(m, *frames):
    return [m.check(snapshot(f)) for f in frames]


def test_rearm_requires_empty_case_before_bottle():
    m, _ = make([answer(upright=True)] * 3)
    ds = run(m, "a", "b", "c")
    assert [d.outcome for d in ds] == ["no"] * 3 and not any(d.complete for d in ds)


def test_empty_step_completes_on_second_distinct_frame():
    m, _ = make([answer(empty=True)] * 2)
    ds = run(m, "a", "b")
    assert [d.outcome for d in ds] == ["yes", "yes"]
    assert [d.complete for d in ds] == [False, True]
    assert ds[1].step == "empty" and ds[1].frame_id == "b"


def test_bottle_step_rejects_hand_at_bottle():
    m, _ = make([answer(upright=True, hand=True)] * 2, "bottle")
    ds = run(m, "a", "b")
    assert [d.outcome for d in ds] == ["no", "no"] and not ds[1].complete


def test_bottle_step_completes_without_hand():
    m, _ = make([answer(upright=True)] * 2, "bottle")
    assert run(m, "a", "b")[1].complete


def test_two_distinct_frames_required():
    m, calls = make([answer(empty=True)] * 2)
    ds = run(m, "a", "a")
    assert not any(d.complete for d in ds) and len(calls) == 1
    assert run(m, "b")[0].complete


def test_no_resets_streak():
    m, _ = make([answer(empty=True), answer(), answer(empty=True)])
    assert not any(d.complete for d in run(m, "a", "b", "c"))


def test_malformed_json_is_unknown():
    m, _ = make([answer(empty=True), "not json", '{"case_empty": true}', answer(empty=True)])
    ds = run(m, "a", "b", "c", "d")
    assert [d.outcome for d in ds] == ["yes", "unknown", "unknown", "yes"]
    assert not any(d.complete for d in ds)


def test_vlm_exception_is_failure_and_resets():
    m, _ = make([answer(empty=True), RuntimeError("boom"), answer(empty=True)])
    ds = run(m, "a", "b", "c")
    assert [d.outcome for d in ds] == ["yes", "failure", "yes"]
    assert not ds[2].complete


def test_begin_resets_streak_and_seen_frames():
    m, _ = make([answer(empty=True)] * 3)
    run(m, "a")
    m.begin("empty", snapshot("init"))
    ds = run(m, "a", "b")
    assert [d.complete for d in ds] == [False, True]


def test_prompt_keeps_upright_check_wording():
    for s in (
        "You see the head camera of a humanoid robot looking down at a black case.",
        "bottle_upright_on_case is true only if a drink bottle stands upright on top of the black case (not lying down, not held, not on the floor).",
        "hand_at_bottle is true if any robot or human hand touches or holds a bottle.",
        "case_empty is true only if nothing stands or lies on top of the black case.",
        '{"case_empty": true|false, "bottle_upright_on_case": true|false, "hand_at_bottle": true|false}',
    ):
        assert s in PROMPT


def test_sends_only_the_current_frame():
    m, calls = make([answer(empty=True)])
    run(m, "a")
    assert [c["type"] for c in calls[0]] == ["image_url", "text"]


def test_empty_step_rejects_case_empty_with_upright_bottle():
    m, _ = make([answer(empty=True, upright=True)])
    assert run(m, "a")[0].outcome == "no"


@pytest.mark.parametrize("b64", ["!!!notbase64", "bm90IGEganBlZw=="])
def test_corrupt_frame_is_unknown_and_resets(b64):
    m, calls = make([answer(empty=True)] * 2)
    run(m, "a")
    d = m.check(replace(snapshot("bad"), jpeg_rgb_b64=b64))
    assert (d.outcome, d.reason) == ("unknown", "invalid_image") and m.streak == 0
    assert not run(m, "b")[0].complete and len(calls) == 2


@pytest.mark.parametrize("bad", ["true", 1])
def test_non_bool_values_are_unknown(bad):
    m, _ = make([json.dumps({"case_empty": bad, "bottle_upright_on_case": False, "hand_at_bottle": False})] * 2)
    ds = run(m, "a", "b")
    assert [d.outcome for d in ds] == ["unknown"] * 2 and not any(d.complete for d in ds)


def test_vlm_timeout_is_failure():
    m, _ = make([None], limits=HarnessLimits(monitor_deadline_s=0.05))
    m.vlm_call_fn = lambda system, content: time.sleep(0.3)
    d = run(m, "a")[0]
    assert d.outcome == "failure" and m.streak == 0


def test_stale_frame_is_unknown_without_vlm_call():
    m, calls = make([answer(empty=True)])
    d = m.check(snapshot("old", now=0.0))
    assert (d.outcome, d.reason) == ("unknown", "stale_frame") and not calls


def test_begin_rejects_unknown_step_and_check_needs_begin():
    m, _ = make([])
    with pytest.raises(ValueError):
        m.begin("full", snapshot("init"))
    with pytest.raises(RuntimeError):
        RearmMonitor(lambda s, c: "", lambda: 1.0).check(snapshot("a"))
