"""LoopRunner against a scripted native loop (FakeLoop mirrors harness_control/loop_control's phases)."""

import io
import itertools
import json
import queue
import threading
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from test_g1_monitor import snapshot

from vlm_orchestrator.harness.g1 import loop as loop_mod
from vlm_orchestrator.harness.g1.client import RPCError
from vlm_orchestrator.harness.g1.contract import (
    Lease,
    Response,
    Status,
    validate_params,
)
from vlm_orchestrator.harness.g1.loop import EvidenceLog, LoopRunner, OperatorConsole
from vlm_orchestrator.harness.g1.rearm_monitor import (
    EMPTY_PROMPT,
    TAKEN_PROMPT,
    RearmMonitor,
)
from vlm_orchestrator.harness.g1.registry import load_profile
from vlm_orchestrator.harness.g1.verdict import run_console

PROFILE = load_profile(Path(__file__).resolve().parents[1] / "configs/g1/handover-loop.example.yaml")
TRIAL_STAGES = ("hold_rest", "first_chunk", "blend_in", "episode", "await_take", "to_ready")


@pytest.fixture(autouse=True)
def fast_poll(monkeypatch):
    monkeypatch.setattr(loop_mod, "POLL_S", 0.005)


class FakeLoop:
    """Native HarnessControl + LoopController, advanced one motion step per main-thread get_status."""

    def __init__(self, session_dir, phase="IDLE", stage=None, frozen_stage=None, reason=None, trial=None):
        self.dir = Path(session_dir)
        self.phase, self.stage, self.frozen_stage, self.reason = phase, stage, frozen_stage, reason
        self.trial, self.end_reason = trial, None
        self.switched = phase not in ("IDLE", "COMPLETED")
        self.saved = (frozen_stage, None)  # the stage and motion a freeze interrupted
        self.owner, self.motion, self.recording, self.hold = None, None, False, True
        self.trace = []  # (t, method, params); console prompts as (t, "?<prompt>", None)
        self.lock = threading.RLock()
        self.hooks = []  # f(loop) after each tick; a True return drops the hook
        self.release_refusals, self.write_delay, self.write_files = 0, 0.0, True
        self.native_next = None  # loop_trial rejects a number below this (else the recorded count)
        self.frames = 0

    # motion
    def tick(self):
        if self.motion:
            left, then = self.motion
            self.motion = (left - 1, then) if left > 1 else None
            if left <= 1:
                then()
        self.hooks = [h for h in self.hooks if not h(self)]

    def move(self, phase, stage, polls, then):
        self.phase, self.stage, self.motion = phase, stage, (polls, then)

    def hold_at(self, stage):
        self.phase, self.stage, self.motion = "LOOP_HOLD", stage, None

    def _switched(self):
        self.switched = True
        self.hold_at("stance")

    def _trial_end(self):
        self.end_reason = self.end_reason or "handover complete"
        self._write()
        self.hold_at("await_take")  # R24: hold the offer pose until loop_to_ready

    def lowered(self):
        self.move("LOOP_MOTION", "handback", 2, self._handed_back)

    def _handed_back(self):
        self.phase, self.stage, self.motion, self.hold = "COMPLETED", None, None, False

    def _write(self):
        self.recording, n = False, self.trial
        if not self.write_files:
            return

        def write():
            base = self.dir / f"trial_{n:03d}"
            np.savez(f"{base}.npz", episode_end=self.end_reason, wall_start=0.0, wall_stop=1.0, model="m")
            np.savez(f"{base}_frames.npz", offsets=np.array([0, 1]))

        threading.Timer(self.write_delay, write).start() if self.write_delay else write()

    def freeze(self, reason):
        if not self.phase.startswith("LOOP_"):
            raise RPCError("NOT_READY", "Autonomous loop not taken over")
        if self.phase == "LOOP_FROZEN":
            return
        if self.recording:
            self.end_reason = self.end_reason or reason
            self._write()
        self.saved = (self.stage, self.motion)
        self.frozen_stage, self.stage, self.phase, self.reason, self.motion = (
            self.stage, "frozen", "LOOP_FROZEN", reason, None)

    def resume(self):
        stage, motion = self.saved
        self.frozen_stage = self.reason = None
        if stage == "stance" and not self.switched:  # R16a
            self.phase, self.stage, self.motion = "IDLE", None, None
        elif stage in TRIAL_STAGES or stage == "ready":
            self.move("LOOP_MOTION", "to_ready", 2, lambda: self.hold_at("ready"))
        elif motion:
            self.phase, self.stage, self.motion = "LOOP_MOTION", stage, motion
        else:
            self.hold_at(stage)

    def status(self):
        return Status(
            "boot", self.phase, None, None, 0, True, True, "POSE", None, self.hold, 0.0, 1, 0.0, "f",
            PROFILE.registry_sha256, PROFILE.policy_host, PROFILE.policy_port, PROFILE.checkpoint, self.owner,
            self.reason, loop_stage=self.stage, loop_trial=self.trial, trial_end_reason=self.end_reason,
            frozen_stage=self.frozen_stage,
        )

    def dispatch(self, method, params, session_id):
        if method == "claim_control":
            if self.owner:
                raise RPCError("BUSY", "owned")
            self.owner = session_id
            return Lease("boot", session_id, "lease", time.monotonic() + 2)
        if session_id != self.owner:
            raise RPCError("NOT_OWNER", "No matching owner")
        if method == "release_control":
            if self.phase.startswith("LOOP_"):  # R16b
                self.freeze("released")
            elif self.release_refusals:
                self.release_refusals -= 1
                raise RPCError("NOT_READY", "Release requires a confirmed planner hold")
            self.owner = None
        elif method == "loop_take_over":
            assert self.phase in ("IDLE", "COMPLETED")
            self.switched, self.trial, self.end_reason = False, None, None
            self.move("LOOP_MOTION", "stance", 2, self._switched)
        elif method == "loop_raise":
            assert (self.phase, self.stage) == ("LOOP_HOLD", "stance")
            self.move("LOOP_MOTION", "raise", 2, lambda: self.hold_at("ready"))
        elif method == "loop_trial":
            assert (self.phase, self.stage) == ("LOOP_HOLD", "ready")
            floor = self.native_next if self.native_next is not None else len(list(self.dir.glob("trial_*_frames.npz")))
            if params["trial"] < floor:
                raise RPCError("INVALID_REQUEST", f"trial {params['trial']} is already recorded in {self.dir}")
            self.trial, self.end_reason, self.recording = params["trial"], None, True
            self.move("LOOP_TRIAL", "episode", 3, self._trial_end)
        elif method == "loop_freeze":
            self.freeze("operator stop")
        elif method == "loop_resume":
            assert self.phase == "LOOP_FROZEN"
            self.resume()
        elif method == "loop_to_ready":
            assert (self.phase, self.stage) == ("LOOP_HOLD", "await_take")
            self.move("LOOP_MOTION", "to_ready", 2, lambda: self.hold_at("ready"))
        elif method == "loop_end":
            assert (self.phase, self.stage) == ("LOOP_HOLD", "ready")
            self.move("LOOP_MOTION", "lower", 2, self.lowered)
        return self.status()


class FakeClient:
    def __init__(self, loop, ticks):
        self.loop, self.ticks = loop, ticks

    def get_status(self):
        with self.loop.lock:
            if self.ticks:
                self.loop.tick()
            return self.loop.status()

    def observe(self):
        with self.loop.lock:
            self.loop.frames += 1
            return replace(snapshot(f"frame{self.loop.frames}", time.monotonic()), runtime_id="boot")

    def request(self, method, params, *, session_id=None, lease_id=None, request_id=None):
        validate_params(method, params)
        with self.loop.lock:
            self.loop.trace.append((time.monotonic(), method, params))
            return Response(request_id or "r", "boot", self.loop.dispatch(method, params, session_id))

    def close(self):
        pass


def factory(loop):
    made = []

    def make():  # the first client is the runner's main-thread client; later ones are heartbeats
        made.append(FakeClient(loop, ticks=not made))
        return made[-1]

    return make


class Console:
    def __init__(self, loop, stops=(), interrupt=None, until=None):
        self.loop, self.stops, self.interrupt, self.pressed_at = loop, list(stops), interrupt, None
        self.until = until or {}  # prompt prefix -> f(): the gate stays open, polled, until f() is true
        self.answered = {}  # prompt prefix -> monotonic time of the (fake) Enter
        self.out = io.StringIO()

    def wait_enter(self, prompt, poll=None, min_after=0.0):
        if poll:
            poll()
        key = prompt.split("?")[0] + "?"
        self.loop.trace.append((time.monotonic(), "?" + key, None))
        if self.interrupt and prompt.startswith(self.interrupt):
            raise KeyboardInterrupt
        while key in self.until and not self.until[key]():
            if poll:
                poll()
            time.sleep(loop_mod.POLL_S)
        self.answered[key] = time.monotonic()

    def stop_pressed(self):
        if self.stops and self.stops[0](self.loop):
            self.stops.pop(0)
            self.pressed_at = time.monotonic()
            return True
        return False

    def close(self):
        pass


class Rearm(RearmMonitor):
    """Real RearmMonitor; the fake VLM always sees what the current step asks for."""

    def __init__(self, block_s=0.0, on_hand=0):
        self.steps, self.block_s = [], block_s
        self.on_hand = on_hand  # the first `on_hand` "taken" calls see the bottle still on the robot's hand
        self.called, self.unblock = threading.Event(), threading.Event()
        super().__init__(self._vlm, time.monotonic, PROFILE.limits)

    def begin(self, step, initial):
        self.steps.append(step)
        super().begin(step, initial)

    def _vlm(self, system, content):
        self.called.set()
        if self.block_s:
            self.unblock.wait(self.block_s)
        if content[1]["text"] == EMPTY_PROMPT:
            return '{"object_on_case": false}'
        if content[1]["text"] == TAKEN_PROMPT:
            self.on_hand -= 1
            return json.dumps({"bottle_on_robot_hand": self.on_hand >= 0})
        return json.dumps({"bottle_upright_on_case": self.step == "bottle", "hand_at_bottle": False})


class Lines:
    """A stdin stand-in: blocks until a line is put; None is EOF."""

    def __init__(self):
        self.q = queue.Queue()

    def __iter__(self):
        while (line := self.q.get()) is not None:
            yield line


class AutoGates(OperatorConsole):
    """The real reader thread and stop key; gate prompts answer themselves."""

    def __init__(self, loop, stream, interrupt=None):
        super().__init__(stream)
        self.loop, self.interrupt = loop, interrupt

    def wait_enter(self, prompt, poll=None, min_after=0.0):
        self.loop.trace.append((time.monotonic(), "?" + prompt.split("?")[0] + "?", None))
        if self.interrupt and prompt.startswith(self.interrupt):
            raise KeyboardInterrupt


def build(tmp_path, loop=None, *, trials=1, console=None, monitor=None, confirm=True):
    loop = loop or FakeLoop(tmp_path)
    monitor = monitor or Rearm()
    runner = LoopRunner(PROFILE, factory(loop), monitor, tmp_path, trials, console or Console(loop),
                        confirm_paths=confirm)
    return loop, runner, monitor


def methods(loop):
    return [m for _, m, _ in loop.trace if m != "heartbeat"]


def trials_sent(loop):
    return [p["trial"] for _, m, p in loop.trace if m == "loop_trial"]


def evidence(tmp_path, trial):
    return [json.loads(s) for s in (tmp_path / f"trial_{trial:03d}_evidence.jsonl").read_text().splitlines()]


def in_episode(loop):
    return loop.stage == "episode"


def test_full_session_sequence(tmp_path):
    loop, runner, _ = build(tmp_path, trials=3)
    result = runner.run()
    assert result.outcome == "completed", result.reason
    assert methods(loop) == [
        "claim_control", "?deploy ready?", "?robot in position?", "loop_take_over", "?start loop?", "?raise?",
        "loop_raise", "loop_trial", "?bottle taken?", "loop_to_ready", "loop_trial", "?bottle taken?", "loop_to_ready", "loop_trial", "?bottle taken?", "loop_to_ready",
        "?lower?", "loop_end", "release_control",
    ]
    assert trials_sent(loop) == [0, 1, 2]
    assert all(p["session_dir"] == str(tmp_path.resolve()) for _, m, p in loop.trace if m == "loop_trial")
    assert loop.owner is None and loop.phase == "COMPLETED"


def test_first_trial_skips_empty_step_later_trials_require_it(tmp_path):
    _, runner, monitor = build(tmp_path, trials=3)
    assert runner.run().outcome == "completed"
    assert monitor.steps == ["bottle", "taken", "empty", "bottle", "taken", "empty", "bottle", "taken"]


def test_stop_freezes_then_clear_resumes_then_restart(tmp_path):
    loop, runner, monitor = build(tmp_path, trials=2)
    runner.console = Console(loop, stops=[in_episode])
    assert runner.run().outcome == "completed"
    m = methods(loop)
    assert m[m.index("loop_trial"):] == [
        "loop_trial", "loop_freeze", "?clear?", "loop_resume", "?restart loop?", "loop_trial", "?bottle taken?", "loop_to_ready", "?lower?",
        "loop_end", "release_control",
    ]
    assert trials_sent(loop) == [0, 1]
    assert monitor.steps == ["bottle", "bottle", "taken"]  # after a restart: the bottle step only
    assert evidence(tmp_path, 0)[-1]["kind"] == "trial_end" and evidence(tmp_path, 0)[-1]["reason"] == "operator stop"


def test_native_freeze_handled_as_stop(tmp_path):
    loop, runner, _ = build(tmp_path, trials=2)

    def guard(lp):
        if lp.stage == "episode":
            lp.freeze("robot state 0.25 s old")
            return True

    loop.hooks.append(guard)
    assert runner.run().outcome == "completed"
    m = methods(loop)
    assert "loop_freeze" not in m
    assert m[m.index("loop_trial"):m.index("?lower?")] == [
        "loop_trial", "?clear?", "loop_resume", "?restart loop?", "loop_trial", "?bottle taken?", "loop_to_ready"]
    kinds = {e["kind"]: e for e in evidence(tmp_path, 0)}
    assert "native freeze in episode: robot state 0.25 s old" in runner.console.out.getvalue().splitlines()
    assert kinds["native_freeze"]["reason"] == "robot state 0.25 s old"
    assert kinds["trial_end"]["reason"] == "robot state 0.25 s old"


def test_confirm_paths_gates_raise_and_lower(tmp_path):
    loop, runner, _ = build(tmp_path, trials=1)
    assert runner.run().outcome == "completed"
    m = methods(loop)
    assert m.index("?raise?") == m.index("loop_raise") - 1 and m.index("?lower?") == m.index("loop_end") - 1
    off = tmp_path / "off"
    off.mkdir()
    loop, runner, _ = build(off, trials=1, confirm=False)
    assert runner.run().outcome == "completed"
    m = methods(loop)
    assert "?raise?" not in m and "?lower?" not in m and "loop_raise" in m and "loop_end" in m
    assert m.index("?start loop?") < m.index("loop_raise")  # the start gate stays


def test_stop_key_and_verdict_console_are_separate(tmp_path):
    # The loop's console never parses text: any line, verdict-looking or empty, is one Enter press.
    probe = Lines()
    console = OperatorConsole(probe)
    probe.q.put("s 1 dropped it\n")
    deadline = time.monotonic() + 1
    while not console.stop_pressed():
        assert time.monotonic() < deadline
        time.sleep(0.005)
    probe.q.put(None)
    # So verdicts go to the verdict console's own stdin (another process); they never reach the loop.
    loop = FakeLoop(tmp_path)
    stdin = Lines()
    console = AutoGates(loop, stdin)

    def verdict_lines():
        for t in (0, 1):
            while not (tmp_path / f"trial_{t:03d}_frames.npz").exists():
                time.sleep(0.005)
            time.sleep(0.05)  # let the console re-read its pending list (it uses the one read before the line)
            yield "s"

    scorer = threading.Thread(target=run_console, args=(tmp_path, verdict_lines(), 0.01), daemon=True)
    scorer.start()
    _, runner, _ = build(tmp_path, loop, trials=2, console=console)
    assert runner.run().outcome == "completed"
    scorer.join(2)
    assert "loop_freeze" not in methods(loop) and trials_sent(loop) == [0, 1]
    assert [json.loads(s)["verdict"] for s in (tmp_path / "results.jsonl").read_text().splitlines()] == ["success"] * 2
    stdin.q.put(None)


def test_session_end_request_file_ends_after_current_trial(tmp_path):
    loop, runner, _ = build(tmp_path, trials=5)

    def request(lp):
        if lp.stage == "episode":
            (tmp_path / "end_session.request").touch()
            return True

    loop.hooks.append(request)
    assert runner.run().outcome == "completed"
    m = methods(loop)
    assert trials_sent(loop) == [0] and m[m.index("loop_trial") + 1:] == [
        "?bottle taken?", "loop_to_ready", "?lower?", "loop_end", "release_control"]  # the end still goes through the taken gate
    assert not (tmp_path / "end_session.request").exists()  # consumed: a later session is not ended by it


def test_fresh_runner_reclaims_frozen_loop_without_takeover(tmp_path):
    loop, first, _ = build(tmp_path, trials=3)

    def lease_expires(lp):
        if lp.stage == "episode":
            lp.freeze("lease expired")
            lp.owner = None
            return True

    loop.hooks.append(lease_expires)
    lost = first.run()
    assert lost.outcome == "interrupted" and "(phase LOOP_FROZEN, reason lease expired)" in lost.reason
    assert (loop.phase, loop.frozen_stage, loop.owner) == ("LOOP_FROZEN", "episode", None)
    mark = len(loop.trace)
    _, second, monitor = build(tmp_path, loop, trials=3)
    assert second.run().outcome == "completed"
    m = [x for _, x, _ in loop.trace[mark:] if x != "heartbeat"]
    assert "loop_take_over" not in m and "loop_raise" not in m
    assert m[:5] == ["claim_control", "?clear?", "loop_resume", "?restart loop?", "loop_trial"]
    assert "?deploy ready?" not in m
    assert trials_sent(loop) == [0, 1, 2] and monitor.steps[:4] == ["bottle", "taken", "empty", "bottle"]
    assert any(e["kind"] == "reclaim" for e in evidence(tmp_path, 1))


def test_reclaim_frozen_during_lowering_completes_without_restart(tmp_path):
    loop = FakeLoop(tmp_path, "LOOP_FROZEN", "frozen", frozen_stage="lower", reason="lease expired")
    loop.saved = ("lower", (2, loop.lowered))
    _, runner, _ = build(tmp_path, loop, trials=3)
    assert runner.run().outcome == "completed"
    assert methods(loop) == ["claim_control", "?clear?", "loop_resume", "release_control"]
    assert loop.phase == "COMPLETED" and loop.owner is None


def test_stop_dispatched_while_vlm_call_blocks(tmp_path, monkeypatch):
    monkeypatch.setattr(loop_mod, "POLL_S", 0.05)  # the real 20 Hz poll
    loop = FakeLoop(tmp_path)
    stdin = Lines()
    monitor = Rearm(block_s=5.0)
    pressed = []

    def press():
        monitor.called.wait(5)
        time.sleep(0.5)
        pressed.append(time.monotonic())
        stdin.q.put("\n")

    threading.Thread(target=press, daemon=True).start()
    _, runner, _ = build(tmp_path, loop, trials=1, monitor=monitor, confirm=False,
                         console=AutoGates(loop, stdin, interrupt="clear?"))
    try:
        result = runner.run()
    finally:
        monitor.unblock.set()
        stdin.q.put(None)
    assert result.outcome == "interrupted"
    freeze_at = [t for t, m, _ in loop.trace if m == "loop_freeze"]
    assert len(freeze_at) == 1 and 0 <= freeze_at[0] - pressed[0] < 0.2, freeze_at[0] - pressed[0]
    assert loop.phase == "LOOP_FROZEN" and loop.owner is None  # released frozen (R16b), reclaimable


def test_evidence_sidecar_records_trigger_and_stops(tmp_path):
    loop, runner, _ = build(tmp_path, trials=1)
    runner.console = Console(loop, stops=[in_episode])
    assert runner.run().outcome == "completed"
    events = evidence(tmp_path, 0)
    kinds = [e["kind"] for e in events]
    for kind in ("rearm_begin", "rearm_decision", "rearm_end", "trial_start", "trial_end", "stop_key", "loop_freeze"):
        assert kind in kinds, kind
    decisions = [e for e in events if e["kind"] == "rearm_decision"]
    assert len(decisions) == 2 and decisions[-1]["complete"]
    for d in decisions:
        assert (tmp_path / d["frame_path"]).is_file() and d["frame_path"].startswith("evidence/trial_000/")
        assert d["outcome"] == "yes" and d["frame_id"]
        assert json.loads(d["answers"]["upright_check"]["raw"]) == d["answers"]["upright_check"]["parsed"]
    stop = next(e for e in events if e["kind"] == "stop_key")
    freeze = next(e for e in events if e["kind"] == "loop_freeze")
    assert stop["pressed_at"] <= freeze["dispatched_at"]
    assert next(e for e in events if e["kind"] == "trial_end")["reason"] == "operator stop"
    with np.load(tmp_path / "trial_000.npz") as z:  # the lerobot trial files are the native writer's
        assert str(z["episode_end"]) == "operator stop"
    assert not (tmp_path / "results.jsonl").exists()
    assert sorted(p.name for p in tmp_path.iterdir()) == [  # post-trial events go to the next trial's sidecar
        "evidence", "trial_000.npz", "trial_000_evidence.jsonl", "trial_000_frames.npz", "trial_001_evidence.jsonl"]


def test_freeze_during_take_over_retakes_over(tmp_path):  # R16a
    loop, runner, _ = build(tmp_path, trials=1)
    runner.console = Console(loop, stops=[lambda lp: (lp.phase, lp.stage) == ("LOOP_MOTION", "stance")])
    assert runner.run().outcome == "completed"
    m = methods(loop)
    assert m[:10] == ["claim_control", "?deploy ready?", "?robot in position?", "loop_take_over", "loop_freeze",
                      "?clear?", "loop_resume", "loop_take_over", "?start loop?", "?raise?"]
    assert "?restart loop?" not in m and m.count("loop_take_over") == 2


def test_session_end_waits_for_trial_files(tmp_path, monkeypatch):  # R15
    loop, runner, _ = build(tmp_path, trials=1)
    loop.write_delay = 0.3
    assert runner.run().outcome == "completed"
    assert trials_sent(loop) == [0]  # an in-flight trial counts: no extra trial while its files are written
    release_at = next(t for t, m, _ in loop.trace if m == "release_control")
    assert (tmp_path / "trial_000_frames.npz").stat().st_mtime <= time.time()
    assert not any(e["kind"] == "trial_files_missing" for e in evidence(tmp_path, 1))
    assert release_at > next(t for t, m, _ in loop.trace if m == "loop_end")

    monkeypatch.setattr(loop_mod, "TRIAL_FILES_WAIT_S", 0.1)
    other = tmp_path / "missing"
    other.mkdir()
    loop, runner, _ = build(other, trials=1)
    loop.write_files = False
    assert runner.run().outcome == "completed"
    missing = [e for e in evidence(other, 1) if e["kind"] == "trial_files_missing"]
    assert missing and missing[0]["trials"] == [0]
    assert methods(loop)[-1] == "release_control" and loop.owner is None


def test_release_after_completed_retries_not_ready(tmp_path, monkeypatch):
    loop, runner, _ = build(tmp_path, trials=1)
    loop.release_refusals = 3
    assert runner.run().outcome == "completed"
    assert methods(loop).count("release_control") == 4 and loop.owner is None

    monkeypatch.setattr(loop_mod, "RELEASE_RETRY_S", 0.05)
    other = tmp_path / "stuck"
    other.mkdir()
    loop, runner, _ = build(other, trials=1)
    loop.release_refusals = 10**6
    result = runner.run()
    assert result.outcome == "fault" and "planner hold" in result.reason


def test_rejected_trial_number_is_fatal_and_releases_frozen(tmp_path):
    loop, runner, _ = build(tmp_path, trials=2)
    loop.native_next = 5
    result = runner.run()
    assert result.outcome == "fault" and "already recorded" in result.reason
    assert methods(loop)[-1] == "release_control"
    assert loop.phase == "LOOP_FROZEN" and loop.owner is None


def test_evidence_log_frame_path(tmp_path):
    log = EvidenceLog(tmp_path)
    s = snapshot("frame-x", 1.0)
    rel = log.frame(3, s)
    assert rel.startswith("evidence/trial_003/") and rel.endswith(".jpg") and (tmp_path / rel).read_bytes()
    log.event(3, "x", a=1)
    assert evidence(tmp_path, 3)[0]["kind"] == "x" and evidence(tmp_path, 3)[0]["a"] == 1


def test_enter_pressed_as_motion_ends_is_a_stop_not_the_gate_answer(tmp_path):
    # The press lands after the take-over's last motion poll, before "start loop?" is printed.
    loop, runner, _ = build(tmp_path, trials=1)
    runner.console = Console(loop, stops=[lambda lp: (lp.phase, lp.stage) == ("LOOP_HOLD", "stance")])
    assert runner.run().outcome == "completed"
    m = methods(loop)
    assert m[3:9] == ["loop_take_over", "loop_freeze", "?clear?", "loop_resume", "?raise?", "loop_raise"]
    assert "?start loop?" not in m  # resumed at the stance: back to the raise step


def test_loop_command_wires_runner(tmp_path, monkeypatch, capsys):
    from vlm_orchestrator.harness.g1 import cli
    from vlm_orchestrator.harness.g1.runner import MissionResult

    seen = {}

    class Runner:
        def __init__(self, profile, factory, monitor, session_dir, n_trials, console, confirm_paths=True):
            seen.update(profile=profile, monitor=monitor, session_dir=session_dir, n_trials=n_trials,
                        console=console, confirm_paths=confirm_paths)

        def run(self):
            return MissionResult("", "completed", "ok", tmp_path)

    monkeypatch.setattr(cli, "LoopRunner", Runner)
    monkeypatch.setattr(cli, "OperatorConsole", lambda: Console(None))
    argv = ["--profile", "configs/g1/handover-loop.example.yaml", "loop", "--session-dir", str(tmp_path),
            "--trials", "4", "--vlm-base-url", "http://127.0.0.1:8000/v1", "--vlm-model", "qwen3-vl-2b",
            "--no-confirm-paths"]
    assert cli.main(argv) == 0
    assert seen["n_trials"] == 4 and seen["confirm_paths"] is False and seen["session_dir"] == tmp_path
    assert isinstance(seen["monitor"], RearmMonitor) and seen["profile"].registry_sha256 == PROFILE.registry_sha256
    assert json.loads(capsys.readouterr().out)["outcome"] == "completed"


def test_clear_gate_ignores_presses_in_its_first_second(tmp_path):  # R21: a double-tap stop never resumes
    loop, stdin, presses = FakeLoop(tmp_path), Lines(), {}

    class Gates(OperatorConsole):  # the real wait_enter for "clear?"; other gates answer themselves
        def wait_enter(self, prompt, poll=None, min_after=0.0):
            loop.trace.append((time.monotonic(), "?" + prompt.split("?")[0] + "?", None))
            if not prompt.startswith("clear?"):
                return
            assert "presses in the first 1 s are ignored" in prompt
            shown = time.monotonic()

            def tap():
                for name, dt in (("early", 0.1), ("late", 1.2)):
                    time.sleep(max(0.0, shown + dt - time.monotonic()))
                    presses[name] = time.monotonic()
                    stdin.q.put("\n")

            threading.Thread(target=tap, daemon=True).start()
            super().wait_enter(prompt, poll, min_after)

    def stop(lp):
        if lp.stage == "episode":
            stdin.q.put("\n")
            return True

    loop.hooks.append(stop)
    _, runner, _ = build(tmp_path, loop, trials=1, console=Gates(stdin, out=io.StringIO()))
    try:
        assert runner.run().outcome == "completed"
    finally:
        stdin.q.put(None)
    resume_at = next(t for t, m, _ in loop.trace if m == "loop_resume")
    assert presses["early"] < presses["late"] < resume_at < presses["late"] + 0.2


def test_stop_evidence_uses_the_first_press_of_a_burst():
    stdin = Lines()
    console = OperatorConsole(stdin)
    t0 = time.monotonic()
    stdin.q.put("\n")
    time.sleep(0.1)
    stdin.q.put("\n")
    time.sleep(0.05)
    assert console.stop_pressed() and t0 <= console.pressed_at < t0 + 0.05
    assert not console.stop_pressed()  # the burst was drained
    stdin.q.put(None)


def test_no_restart_prompt_when_the_session_is_over(tmp_path):
    loop, runner, _ = build(tmp_path, trials=1)  # the stopped trial was the last one
    runner.console = Console(loop, stops=[in_episode])
    assert runner.run().outcome == "completed"
    m = methods(loop)
    assert m[m.index("loop_trial"):] == ["loop_trial", "loop_freeze", "?clear?", "loop_resume", "?lower?", "loop_end",
                                         "release_control"]
    other = tmp_path / "ended"
    other.mkdir()
    loop, runner, _ = build(other, trials=5)  # the end was requested during the stopped trial

    def end_then_stop(lp):  # the operator types e at the verdict console, then stops the trial
        if in_episode(lp):
            (other / "end_session.request").touch()
            return True

    runner.console = Console(loop, stops=[end_then_stop])
    assert runner.run().outcome == "completed"
    m = methods(loop)
    assert "?restart loop?" not in m and trials_sent(loop) == [0] and m[-3:] == ["?lower?", "loop_end", "release_control"]


def test_stale_end_request_removed_at_entry_and_after_completed(tmp_path):
    (tmp_path / "end_session.request").touch()  # left by an earlier session
    loop, runner, _ = build(tmp_path, trials=2)

    def typed_while_lowering(lp):
        if lp.stage == "lower":
            (tmp_path / "end_session.request").touch()
            return True

    loop.hooks.append(typed_while_lowering)
    assert runner.run().outcome == "completed"
    assert trials_sent(loop) == [0, 1]  # the stale request did not end this session
    assert any(e["kind"] == "stale_end_request_removed" for e in evidence(tmp_path, 0))
    assert not (tmp_path / "end_session.request").exists()


class DeadThenAlive(Rearm):
    """The VLM raises for its first `dead_s`, then answers like Rearm."""

    def __init__(self, dead_s):
        super().__init__()
        self.alive_at = time.monotonic() + dead_s

    def _vlm(self, system, content):
        if time.monotonic() < self.alive_at:
            raise RuntimeError("connection refused")
        return super()._vlm(system, content)


def test_console_shows_rearm_and_trial_progress(tmp_path, monkeypatch):
    loop, runner, _ = build(tmp_path, trials=1, monitor=DeadThenAlive(0.3))
    runner.console = Console(loop, stops=[in_episode])
    assert runner.run().outcome == "completed"
    out = runner.console.out.getvalue().splitlines()
    dead = "re-arm bottle: failure upright_check: RuntimeError: connection refused"
    assert out.count(dead) == 1  # repeated failures: once per SAY_REPEAT_S (2 s)
    assert out[out.index(dead) + 1:] == [
        "re-arm bottle: yes", "trial 0: start", "stop key: freezing", "frozen in episode (operator stop)",
        "trial 0: end (operator stop)"]

    monkeypatch.setattr(loop_mod, "SAY_REPEAT_S", 0.05)
    other = tmp_path / "repeat"
    other.mkdir()
    loop, runner, _ = build(other, trials=1, monitor=DeadThenAlive(0.3))
    assert runner.run().outcome == "completed"
    assert runner.console.out.getvalue().splitlines().count(dead) >= 2


def test_events_log_status_only_on_change(tmp_path):
    _, runner, _ = build(tmp_path, trials=2)
    result = runner.run()
    rows = [json.loads(s) for s in (result.evidence_dir / "events.jsonl").read_text().splitlines()]
    keys = [(r["phase"], r["loop_stage"], r["reason"], r["loop_trial"]) for r in rows if r["event"] == "status"]
    assert keys and all(a != b for a, b in itertools.pairwise(keys))
    assert ("LOOP_TRIAL", "episode", None, 1) in keys and keys[-1][0] == "COMPLETED"



def at(loop, method):
    return [t for t, m, _ in loop.trace if m == method]


def decisions_logged(session_dir, trial, n, kind="taken_decision"):
    def enough():
        path = session_dir / f"trial_{trial:03d}_evidence.jsonl"
        return path.exists() and sum(f'"kind": "{kind}"' in s for s in path.read_text().splitlines()) >= n
    return enough


def test_no_loop_to_ready_before_the_operator_enter_even_when_the_vlm_says_taken(tmp_path):  # R25
    loop, runner, _ = build(tmp_path, trials=1)  # the fake VLM sees the hand empty: "taken" yes, then complete
    runner.console = Console(loop, until={"bottle taken?": decisions_logged(tmp_path, 0, 4)})
    assert runner.run().outcome == "completed"
    events = evidence(tmp_path, 0)  # the taken check is logged with the trial that ended
    decisions = [e for e in events if e["kind"] == "taken_decision"]
    assert len(decisions) >= 4 and decisions[1]["complete"] and all(d["outcome"] == "yes" for d in decisions)
    for d in decisions:
        assert d["frame_path"].startswith("evidence/trial_000/") and (tmp_path / d["frame_path"]).is_file()
        assert d["answers"]["taken"]["parsed"] == {"bottle_on_robot_hand": False}
    enter = runner.console.answered["bottle taken?"]
    assert decisions[1]["at"] < enter < at(loop, "loop_to_ready")[0] < at(loop, "loop_end")[0]
    assert len(at(loop, "loop_to_ready")) == 1
    kinds = [e["kind"] for e in events]
    assert kinds.index("trial_end") < kinds.index("taken_begin") < kinds.index("taken_confirmed") < kinds.index("to_ready")
    assert "taken_end" not in kinds
    assert "taken: yes" in runner.console.out.getvalue().splitlines()


def test_taken_gate_waits_for_enter_while_the_vlm_sees_the_bottle(tmp_path):
    loop, runner, _ = build(tmp_path, trials=1, monitor=Rearm(on_hand=10**6))
    runner.console = Console(loop, until={"bottle taken?": decisions_logged(tmp_path, 0, 3)})
    assert runner.run().outcome == "completed"
    decisions = [e for e in evidence(tmp_path, 0) if e["kind"] == "taken_decision"]
    assert [d["outcome"] for d in decisions][:3] == ["no"] * 3
    assert at(loop, "loop_to_ready")[0] > runner.console.answered["bottle taken?"]  # the operator, not the VLM
    assert "taken: no" in runner.console.out.getvalue().splitlines()


def test_taken_gate_debounce_with_the_real_console(tmp_path):
    loop, stdin, presses = FakeLoop(tmp_path), Lines(), {}

    class Gates(OperatorConsole):  # the real wait_enter for "bottle taken?"; other gates answer themselves
        def wait_enter(self, prompt, poll=None, min_after=0.0):
            loop.trace.append((time.monotonic(), "?" + prompt.split("?")[0] + "?", None))
            if not prompt.startswith("bottle taken?"):
                return
            assert "presses in the first 1 s are ignored" in prompt
            shown = time.monotonic()

            def tap():
                for name, dt in (("early", 0.1), ("late", 1.2)):
                    time.sleep(max(0.0, shown + dt - time.monotonic()))
                    presses[name] = time.monotonic()
                    stdin.q.put("\n")

            threading.Thread(target=tap, daemon=True).start()
            super().wait_enter(prompt, poll, min_after)

    _, runner, _ = build(tmp_path, loop, trials=1, console=Gates(stdin, out=io.StringIO()))
    try:
        assert runner.run().outcome == "completed"
    finally:
        stdin.q.put(None)
    assert presses["early"] < presses["late"] < at(loop, "loop_to_ready")[0] < presses["late"] + 0.2
    assert "loop_freeze" not in methods(loop)


def test_enter_before_the_taken_prompt_is_a_stop(tmp_path):
    loop, runner, _ = build(tmp_path, trials=2)
    runner.console = Console(loop, stops=[lambda lp: lp.stage == "await_take"])
    assert runner.run().outcome == "completed"
    m = methods(loop)
    assert m[m.index("loop_trial"):] == [
        "loop_trial", "loop_freeze", "?clear?", "loop_resume", "?restart loop?", "loop_trial", "?bottle taken?",
        "loop_to_ready", "?lower?", "loop_end", "release_control"]  # the native resume from await_take is the move
    ends = [e for e in evidence(tmp_path, 0) if e["kind"] == "trial_end"]
    assert [e["reason"] for e in ends] == ["handover complete"]  # the freeze did not end the trial again


def test_native_freeze_during_the_taken_gate(tmp_path):
    loop, runner, _ = build(tmp_path, trials=1)

    ticks = []

    def guard(lp):  # a few polls into the gate
        if lp.stage == "await_take":
            ticks.append(1)
            if len(ticks) == 5:
                lp.freeze("camera frame 0.60 s old")
                return True

    runner.console = Console(loop, until={"bottle taken?": lambda: False})  # only the freeze ends the gate
    loop.hooks.append(guard)
    assert runner.run().outcome == "completed"
    m = methods(loop)
    assert m[m.index("loop_trial"):] == [
        "loop_trial", "?bottle taken?", "?clear?", "loop_resume", "?lower?", "loop_end", "release_control"]


def test_empty_step_skipped_after_the_vlm_saw_the_bottle_on_the_hand(tmp_path):
    loop, runner, monitor = build(tmp_path, trials=2, monitor=Rearm(on_hand=1))  # 1 on-hand answer, then empty
    runner.console = Console(loop, until={"bottle taken?": decisions_logged(tmp_path, 0, 2)})
    assert runner.run().outcome == "completed"
    assert monitor.steps == ["bottle", "taken", "bottle", "taken"]  # trial 1: bottle-only
    skip = [e for e in evidence(tmp_path, 1) if e["kind"] == "rearm_skip_empty"]
    assert skip and "bottle on the robot hand after trial 0" in skip[0]["reason"]
    assert any(s.startswith("re-arm: empty step skipped") for s in runner.console.out.getvalue().splitlines())
    assert next(e for e in evidence(tmp_path, 0) if e["kind"] == "taken_confirmed")["vlm_saw_on_hand"] is True


def test_empty_step_required_when_the_vlm_never_saw_the_bottle_on_the_hand(tmp_path):
    loop, runner, monitor = build(tmp_path, trials=2)
    runner.console = Console(loop, until={"bottle taken?": decisions_logged(tmp_path, 0, 2)})
    assert runner.run().outcome == "completed"
    assert monitor.steps == ["bottle", "taken", "empty", "bottle", "taken"]
    assert not any(e["kind"] == "rearm_skip_empty" for e in evidence(tmp_path, 1))


def test_session_end_after_the_last_trial_goes_through_the_taken_gate(tmp_path):
    loop, runner, _ = build(tmp_path, trials=1, confirm=False)
    assert runner.run().outcome == "completed"
    m = methods(loop)
    assert m[m.index("loop_trial"):] == ["loop_trial", "?bottle taken?", "loop_to_ready", "loop_end", "release_control"]


def test_reclaim_frozen_in_await_take(tmp_path):
    loop = FakeLoop(tmp_path, "LOOP_FROZEN", "frozen", frozen_stage="await_take", reason="lease expired", trial=0)
    loop.end_reason = "handover complete"
    loop._write()  # trial 0 was recorded before the lease lapsed
    loop.saved = ("await_take", None)
    _, runner, monitor = build(tmp_path, loop, trials=2)
    assert runner.run().outcome == "completed"
    assert methods(loop) == ["claim_control", "?clear?", "loop_resume", "?restart loop?", "loop_trial",
                             "?bottle taken?", "loop_to_ready", "?lower?", "loop_end", "release_control"]
    assert trials_sent(loop) == [1] and monitor.steps == ["bottle", "taken"]


def test_finished_session_refused_before_claim(tmp_path):
    for t in (0, 1):
        np.savez(tmp_path / f"trial_{t:03d}_frames.npz", offsets=np.array([0, 1]))
    loop, runner, _ = build(tmp_path, trials=2)
    result = runner.run()
    assert result.outcome == "fault"
    assert result.reason == f"session {tmp_path.resolve()} has no trials left (2 recorded); use a new --session-dir"
    assert methods(loop) == [] and loop.owner is None  # no claim, no take-over
    # a reclaim still proceeds, so a frozen lowering can finish
    loop = FakeLoop(tmp_path, "LOOP_FROZEN", "frozen", frozen_stage="lower", reason="lease expired")
    loop.saved = ("lower", (2, loop.lowered))
    _, runner, _ = build(tmp_path, loop, trials=2)
    assert runner.run().outcome == "completed"
    assert methods(loop) == ["claim_control", "?clear?", "loop_resume", "release_control"]
