import ast
import json
from pathlib import Path
import time
from dataclasses import replace

import pytest

from test_g1_monitor import snapshot
from vlm_orchestrator.harness.g1.registry import HarnessLimits
from vlm_orchestrator.harness.g1.rearm_monitor import EMPTY_PROMPT, PROMPT, RearmMonitor


def answer(empty=False, upright=False, hand=False):
    return json.dumps(
        {"case_empty": empty, "bottle_upright_on_case": upright, "hand_at_bottle": hand}
    )


def make(replies, step="empty", limits=None):
    # Each queued entry is one frame's world state; PROMPT takes the next entry, EMPTY_PROMPT reads case_empty from it
    # (object_on_case = not case_empty; a non-bool passes through unchanged).
    queue, calls, current = list(replies), [], []

    def vlm(system, content):
        calls.append(content)
        if content[1]["text"] == EMPTY_PROMPT:
            v = json.loads(current[0])["case_empty"]
            return json.dumps({"object_on_case": (not v) if isinstance(v, bool) else v})
        current[:] = [queue.pop(0)]
        r = current[0]
        if isinstance(r, Exception):
            raise r
        try:
            d = json.loads(r)
        except ValueError:
            return r
        if "bottle_upright_on_case" in d:
            d.pop("case_empty", None)
        return json.dumps(d)

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
    assert not any(d.complete for d in ds) and len(calls) == 2
    assert run(m, "b")[0].complete


def test_no_resets_streak():
    m, _ = make([answer(empty=True), answer(), answer(empty=True)])
    assert not any(d.complete for d in run(m, "a", "b", "c"))


def test_malformed_json_is_unknown():
    m, _ = make([answer(empty=True), "not json", '{"case_empty": true}', answer(empty=True)])
    ds = run(m, "a", "b", "c", "d")
    assert [d.outcome for d in ds] == ["yes", "unknown", "unknown", "yes"]
    assert not any(d.complete for d in ds)
    # the evidence keeps the raw reply of a malformed answer (R21)
    assert ds[1].answers == {"upright_check": {"raw": "not json", "parsed": None}}
    assert ds[2].answers == {"upright_check": {"raw": '{"case_empty": true}', "parsed": None}}


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


def test_prompt_is_the_verbatim_upright_check_prompt():
    src = Path(__file__).parent.parent / "docs/artifacts/g1_vlm_upright_check_20261026/upright_check.py"
    node = next(n for n in ast.parse(src.read_text()).body
                if isinstance(n, ast.Assign) and n.targets[0].id == "PROMPT")
    assert PROMPT == ast.literal_eval(node.value)
    assert "case_empty" not in PROMPT
    assert '{"object_on_case": true|false}' in EMPTY_PROMPT and "handle and white labels" in EMPTY_PROMPT


def test_empty_step_upright_bottle_is_no_with_one_call():
    m, calls = make([answer(empty=True, upright=True)])
    assert run(m, "a")[0].outcome == "no" and len(calls) == 1


def test_empty_step_clear_but_not_empty_is_no():
    m, calls = make([answer(empty=False)])
    d = run(m, "a")[0]
    assert d.outcome == "no" and len(calls) == 2 and d.answers["object_on_case"]["parsed"] == {"object_on_case": True}


def test_empty_step_yes_records_both_answers():
    m, calls = make([answer(empty=True)] * 2)
    ds = run(m, "a", "b")
    assert [d.complete for d in ds] == [False, True] and len(calls) == 4
    assert set(ds[0].answers) == {"upright_check", "object_on_case"}
    assert ds[0].answers["upright_check"]["parsed"] == {"bottle_upright_on_case": False, "hand_at_bottle": False}


def test_empty_step_hand_alone_is_no():
    m, calls = make([answer(empty=True, hand=True)])
    assert run(m, "a")[0].outcome == "no" and len(calls) == 1


def test_sends_only_the_current_frame():
    m, calls = make([answer(empty=True)])
    run(m, "a")
    assert [c["type"] for c in calls[0]] == ["image_url", "text"]


@pytest.mark.parametrize("b64", ["!!!notbase64", "bm90IGEganBlZw=="])
def test_corrupt_frame_is_unknown_and_resets(b64):
    m, calls = make([answer(empty=True)] * 2)
    run(m, "a")
    d = m.check(replace(snapshot("bad"), jpeg_rgb_b64=b64))
    assert (d.outcome, d.reason) == ("unknown", "invalid_image") and m.streak == 0
    assert not run(m, "b")[0].complete and len(calls) == 4


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
