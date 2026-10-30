"""Operator-side orchestrator of the autonomous handover loop (runs on PC2 beside the native harness).

Flow (one step method per state; a stop from any step goes to ``_recover``):

  claim -> entry branch by the native phase:
    IDLE/COMPLETED   start: "deploy ready?", "robot in position?" -> take_over
    LOOP_FROZEN      reclaim: recover (no take-over, no startup gates, no raise)
    LOOP_HOLD ready  reclaim: restart;  LOOP_HOLD stance: reclaim: raise
    anything else    error -> release (a release in a LOOP phase leaves it frozen, R16b)
  take_over  loop_take_over, wait stance, "start loop?" -> raise
  raise      ["raise?"], loop_raise, wait ready -> trials
  restart    "restart loop?" -> trials; straight to end when no trial remains or the end is requested
  trials     while trials remain and no end_session.request: re-arm (bottle only on the first trial after a
             start/restart/reclaim, after a "handover complete" trial (R30) or after a hold in which the VLM saw
             the bottle on the robot hand, else empty then bottle), loop_trial, wait await_take (R24: the arm holds its offer pose), log the end reason,
             "bottle taken?" (R25 the operator's Enter, R26 the native torque take detector, R32 the VLM seeing
             the hand empty in TAKEN_EMPTY_FRAMES frames in a row while the native harness reports no bottle on
             the wrist, or R29 the VLM alone in TAKEN_VLM_FRAMES frames, off by default), loop_to_ready,
             wait ready -> end
  end        ["lower?"], loop_end, wait COMPLETED -> finish
  finish     wait <=10 s for every started trial's frames file (R15), release (NOT_READY retried) -> done
  recover    loop_freeze unless already frozen, "clear?", loop_resume, then by the result:
             LOOP_HOLD stance -> raise, LOOP_HOLD ready (also a freeze in await_take: clear = bottle taken)
             -> restart, COMPLETED -> finish, IDLE (a take-over
             aborted by the freeze, R16a) -> take_over

The Enter key rule: a gate prompt is printed only while the robot holds still, and the Enter that answers it
means "proceed". Every other Enter, from the take-over on, is a stop: the main thread polls the console and the
status at 20 Hz and sends loop_freeze within one poll. Before each gate the runner checks the key once more, so
an Enter typed during the motion that led to the gate is still a stop; then the console flushes and prints the
prompt. At "clear?" the robot is already frozen, so extra presses are flushed, never a second stop, and presses
in its first second are ignored (R21: the second tap of a double-tap stop must not resume the robot); the same
holds for "bottle taken?" (R25), where Enter moves the arm from its offer pose to ready. Before the
take-over (the two startup gates) the robot is not the runner's to freeze: Enter only answers the gates. A native
freeze (LOOP_FROZEN) seen while waiting, re-arming or at a hold gate is handled exactly like the stop key.

Threads: the main thread owns the RPC client (status, observe, every loop_* call); one worker thread runs the
VLM check on a frame the main thread fetched and never calls RPC; the heartbeat keeps its own client.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import queue
import sys
import threading
import time
from dataclasses import asdict
from pathlib import Path

from .client import RPCError
from .contract import ObservationSnapshot, SkillCall
from .runner import HarnessRunner, MissionInterrupted, MissionResult
from .verdict import next_trial, remaining_trials

POLL_S = 0.05  # 20 Hz: stop key, status, re-arm
TRIAL_FILES_WAIT_S = 10.0  # R15
RELEASE_RETRY_S = 5.0  # COMPLETED: release is NOT_READY until the native planner hold is confirmed
SAY_REPEAT_S = 2.0  # a repeating re-arm failure/unknown is re-printed at most this often
# R29: consecutive "bottle off the hand" answers that end "bottle taken?"; 0 = log only. R30: 0 by default: at the
# offer pose the head camera sees only an edge of the bottle, and 4 of the 6 holds the camera ended in r33 had the
# bottle still on the palm.
TAKEN_VLM_FRAMES = 0
# R33: with R29 on, a taken check starts at most once per TAKEN_VLM_PERIOD_S, so its TAKEN_VLM_FRAMES answers
# cover at least that many seconds (user, 2026-10-30: three checks one second apart, all empty, or start again).
TAKEN_VLM_PERIOD_S = 1.0
# R32: the camera's "hand empty" in this many distinct frames in a row ends "bottle taken?" when the native harness
# also reports no bottle on the wrist (Status.take_hand_empty): a hand already empty when the hold starts (r36
# trials 7 and 8) gives the torque detector no unload to see. In r33 the camera's false "empty" holds had the
# bottle on the wrist (wrist yaw >= 0.88), which take_hand_empty rules out.
TAKEN_EMPTY_FRAMES = 3
# The camera's "empty" count expires when no new "yes" came for this long (decisions arrive every ~0.6-0.8 s): if
# frames stop arriving, old answers must not combine with a later take_hand_empty (review of PR #5).
TAKEN_EMPTY_FRESH_S = 2.0
CLEAR_DEBOUNCE_S = 1.0  # R21: "clear?" ignores presses this soon after it is shown
MOVING = ("LOOP_MOTION", "LOOP_TRIAL")


class OperatorConsole:
    """Enter only: any line, empty or not, is one press; text is never parsed (verdicts have their own console).

    A background thread reads ``stream``. EOF counts as a stop (no way left to stop) and fails every gate.
    """

    def __init__(self, stream=sys.stdin, out=sys.stdout):
        self._presses, self._eof, self.out, self.pressed_at = queue.Queue(), threading.Event(), out, None
        threading.Thread(target=self._read, args=(stream,), daemon=True).start()

    def _read(self, stream):
        for _ in stream:
            if self._eof.is_set():  # closed
                return
            self._presses.put(time.monotonic())
        self._eof.set()

    def _drain(self):
        """Empty the queue; returns the FIRST press of the burst (the stop's evidence time), or None."""
        first = None
        while True:
            try:
                at = self._presses.get_nowait()
            except queue.Empty:
                return first
            first = at if first is None else first

    def stop_pressed(self) -> bool:
        at = self._drain()
        if at is None and self._eof.is_set():
            at = time.monotonic()
        if at is not None:
            self.pressed_at = at
        return at is not None

    def wait_enter(self, prompt, poll=None, min_after=0.0) -> None:
        """Flush, print, wait for one press; ``poll()`` runs every POLL_S and may raise to abandon the gate.

        Presses received less than ``min_after`` s after the prompt is shown are discarded (debounce).
        """
        self._drain()
        print(prompt, file=self.out, flush=True)
        not_before = time.monotonic() + min_after
        while True:
            try:
                if self._presses.get(timeout=POLL_S) >= not_before:
                    return
            except queue.Empty:
                pass
            if self._eof.is_set():
                raise EOFError("Operator console closed")
            if poll:
                poll()

    def flush(self) -> None:
        """Drop pending presses (a gate that ended without one: they answered it, they are not stops)."""
        self._drain()

    def close(self) -> None:
        self._eof.set()


class EvidenceLog:
    """trial_NNN_evidence.jsonl and evidence/trial_NNN/*.jpg; never touches lerobot's trial files."""

    def __init__(self, session_dir):
        self.dir = Path(session_dir)

    def event(self, trial: int, kind: str, **fields) -> None:
        line = {"kind": kind, "at": time.monotonic(), "wall": time.time(), **fields}
        with open(self.dir / f"trial_{trial:03d}_evidence.jsonl", "a") as f:
            f.write(json.dumps(line, default=str) + "\n")

    def frame(self, trial: int, snapshot: ObservationSnapshot) -> str:
        rel = f"evidence/trial_{trial:03d}/{hashlib.sha256(snapshot.frame_id.encode()).hexdigest()}.jpg"
        (self.dir / rel).parent.mkdir(parents=True, exist_ok=True)
        (self.dir / rel).write_bytes(base64.b64decode(snapshot.jpeg_rgb_b64, validate=True))
        return rel


class _Taken(Exception):
    """What ended "bottle taken?": the native take detector (Status.take_detected, by "torque"), the VLM with
    Status.take_hand_empty (by "camera+wrist", R32) or the VLM alone (by "vlm", R29)."""

    def __init__(self, by):
        super().__init__(by)
        self.by = by


class _Stop(Exception):
    """The stop key (status None) or a native freeze (its LOOP_FROZEN status)."""

    def __init__(self, status=None):
        super().__init__("stop")
        self.status = status


class LoopRunner(HarnessRunner):
    def __init__(self, profile, client_factory, monitor, session_dir, n_trials, console, confirm_paths=True,
                 taken_vlm_frames=TAKEN_VLM_FRAMES):
        if type(n_trials) is not int or n_trials < 1:
            raise ValueError("n_trials must be a positive integer")
        self.session_dir = Path(session_dir).resolve()  # the native process writes here too
        self.session_dir.mkdir(parents=True, exist_ok=True)
        super().__init__(profile, client_factory, monitor, self.session_dir / "evidence")
        self.n_trials, self.console, self.confirm_paths = n_trials, console, confirm_paths
        if type(taken_vlm_frames) is not int or taken_vlm_frames < 0:
            raise ValueError("taken_vlm_frames must be a non-negative integer")
        self.taken_vlm_frames = taken_vlm_frames
        self.evidence = EvidenceLog(self.session_dir)

    def run(self) -> MissionResult:
        if self.profile.handover is None:
            raise ValueError("Handover loop profile required")
        # n: the trial in progress or the next one (R4/R4b); started: trials whose loop_trial this runner sent
        self.n, self.started, self.active, self._worker, self._stopped = 0, [], False, None, None
        self._said = (None, 0.0)  # the last re-arm decision line printed and when
        return self._run_session([SkillCall(self.profile.handover.skill_id, {})], self._session)

    def _validate_executor(self, status):
        """Runs before the claim. A fresh start needs trials left; a reclaim proceeds (a frozen lowering finishes)."""
        if status.phase in ("IDLE", "COMPLETED") and remaining_trials(self.session_dir, self.n_trials) <= 0:
            recorded = len(list(self.session_dir.glob("trial_*_frames.npz")))
            raise ValueError(f"session {self.session_dir} has no trials left ({recorded} recorded); "
                             "use a new --session-dir")

    # --- session --------------------------------------------------------------------------------------------

    def _session(self, _calls):
        step = self._entry()
        while step != "done":
            try:
                step = getattr(self, "_" + step)()
            except _Stop as stop:
                self._stopped, step = stop.status, "recover"
        return f"Loop session complete; {len(self.started)} trials run"

    def _entry(self):
        s = self._check()
        self.n = next_trial(self.session_dir)
        if s.phase in ("IDLE", "COMPLETED"):
            self.evidence.event(self.n, "session_start", phase=s.phase)
            if self._end_requested():  # left by an earlier session; this one has not started
                self.evidence.event(self.n, "stale_end_request_removed")
                self._log("stale_end_request_removed")
                (self.session_dir / "end_session.request").unlink(missing_ok=True)
            return "start"
        if s.loop_trial is not None:  # a reclaimed loop's last trial may still be writing its files
            self.n = max(self.n, s.loop_trial + 1)
            if not self._recorded(s.loop_trial):
                self.started.append(s.loop_trial)
        self.evidence.event(self.n, "reclaim", phase=s.phase, stage=s.loop_stage, frozen_stage=s.frozen_stage,
                            reason=s.reason, loop_trial=s.loop_trial, trial_end_reason=s.trial_end_reason)
        if s.phase == "LOOP_FROZEN":
            self._stopped = s
            return "recover"
        # Unreachable today: the native claim refuses LOOP_HOLD, and every ownerless loop is LOOP_FROZEN.
        if s.phase == "LOOP_HOLD" and s.loop_stage in ("ready", "stance"):
            return "restart" if s.loop_stage == "ready" else "raise"
        raise MissionInterrupted(f"Cannot take control of a loop in {s.phase} (stage {s.loop_stage})")

    def _start(self):
        self._gate("deploy ready? (deploy in PLANNER, native harness running) [Enter]")
        self._gate("robot in position? (standing, clear of the case) [Enter]")
        return "take_over"

    def _take_over(self):
        self._rpc("loop_take_over", {})
        self.evidence.event(self.n, "take_over")
        self._wait(("LOOP_HOLD", "stance"))
        self._gate("start loop? (stance held) [Enter]", self._hold_poll, stop=True)
        return "raise"

    def _raise(self):
        if self.confirm_paths:
            self._gate("raise? (arm to the ready pose) [Enter]", self._hold_poll, stop=True)
        self._rpc("loop_raise", {})
        self._wait(("LOOP_HOLD", "ready"))
        return "trials"

    def _restart(self):
        if self._remaining() <= 0 or self._end_requested():  # the session is over: no restart prompt
            return "end"
        self._gate("restart loop? (ready pose held) [Enter]", self._hold_poll, stop=True)
        self.evidence.event(self.n, "restart")
        return "trials"

    def _trials(self):
        skip_empty, why = True, None  # the first trial after a start, restart or reclaim re-arms bottle-only
        while self._remaining() > 0 and not self._end_requested():
            if why:
                self.evidence.event(self.n, "rearm_skip_empty", reason=why)
                self._say(f"re-arm: empty step skipped ({why})")
            if not all(self._rearm(step) for step in (("bottle",) if skip_empty else ("empty", "bottle"))):
                break  # end_session.request during the re-arm
            self._rpc("loop_trial", {"trial": self.n, "session_dir": str(self.session_dir)})
            self.started.append(self.n)
            self.active = True
            self.evidence.event(self.n, "trial_start")
            self._say(f"trial {self.n}: start")
            ended = self._wait(("LOOP_HOLD", "await_take"))
            self._trial_ended(ended)
            # R24/R25: the arm holds its offer pose until the operator confirms the take; the session end goes
            # through here too (loop_end is ready-only)
            trial = self.n - 1
            seen_on_hand = self._taken_gate(trial)
            self._rpc("loop_to_ready", {})
            self._wait(("LOOP_HOLD", "ready"))
            self.evidence.event(trial, "to_ready")
            self._say("to ready")
            # R30: after a handover the bottle left with the person, so the case needs no empty check; after any
            # other end the bottle may still be on the case (or lying there), so empty then bottle
            handed = ended.trial_end_reason == "handover complete"
            skip_empty = handed or seen_on_hand
            why = (f"trial {trial} ended with a handover" if handed else
                   f"the VLM saw the bottle on the robot hand after trial {trial}" if seen_on_hand else None)
        return "end"

    def _end(self):
        request = self.session_dir / "end_session.request"
        self.evidence.event(self.n, "session_end", cause="end_session.request" if request.exists() else "trials done")
        # the request stays until _finish: a stop at "lower?" resumes to ready, and _restart must still see it
        if self.confirm_paths:
            self._gate("lower? (arm down, hand back to the planner) [Enter]", self._hold_poll, stop=True)
        if not request.exists() and self._remaining() > 0:  # a redo verdict while "lower?" waited: one more trial
            self.evidence.event(self.n, "session_end_withdrawn", reason="redo verdict added a trial")
            self._say("a redo verdict added a trial: not lowering")
            return "restart"
        self._rpc("loop_end", {})
        self._wait(("COMPLETED", None))
        return "finish"

    def _finish(self):
        deadline = time.monotonic() + TRIAL_FILES_WAIT_S
        while (missing := [t for t in self.started if not self._recorded(t)]) and time.monotonic() < deadline:
            self._check()
            time.sleep(POLL_S)
        if missing:
            self.evidence.event(self.n, "trial_files_missing", trials=missing)
        (self.session_dir / "end_session.request").unlink(missing_ok=True)  # one typed during the lowering
        self._release()
        return "done"

    def _recover(self):
        status = self._stopped
        if status is None:
            self.evidence.event(self.n, "stop_key", pressed_at=self.console.pressed_at)
            self._say("stop key: freezing")
            sent = time.monotonic()
            try:
                status = self._rpc("loop_freeze", {})
            except RPCError as exc:  # the key raced the loop leaving its LOOP phases
                phase = self._check().phase if exc.code == "NOT_READY" else None
                if phase not in ("COMPLETED", "IDLE"):
                    raise
                self.evidence.event(self.n, "stop_outside_loop", phase=phase)
                return "finish" if phase == "COMPLETED" else "start"
            self.evidence.event(self.n, "loop_freeze", dispatched_at=sent, acked_at=time.monotonic(),
                                reason=status.reason, frozen_stage=status.frozen_stage)
            self._say(f"frozen in {status.frozen_stage} ({status.reason})")
        else:
            self.evidence.event(self.n, "native_freeze", reason=status.reason, frozen_stage=status.frozen_stage)
            self._say(f"native freeze in {status.frozen_stage}: {status.reason}")
        self._trial_ended(status)
        # R21: debounced; a double-tap stop must never resume the robot
        self._gate(f"clear? (frozen in {status.frozen_stage}: {status.reason}) "
                   f"[Enter resumes; presses in the first {CLEAR_DEBOUNCE_S:g} s are ignored]",
                   min_after=CLEAR_DEBOUNCE_S)
        self._rpc("loop_resume", {})
        s = self._wait(("LOOP_HOLD", "stance"), ("LOOP_HOLD", "ready"), ("COMPLETED", None), ("IDLE", None))
        self.evidence.event(self.n, "resume", phase=s.phase, stage=s.loop_stage)
        return {"stance": "raise", "ready": "restart"}.get(s.loop_stage) or (
            "finish" if s.phase == "COMPLETED" else "take_over")

    # --- helpers --------------------------------------------------------------------------------------------

    def _say(self, text):
        """One short line on the operator console (the re-arm and trial progress)."""
        print(text, file=self.console.out, flush=True)

    def _say_decision(self, d):
        key, now = (d.step, d.outcome, d.reason), time.monotonic()
        if key != self._said[0] or (d.outcome in ("failure", "unknown") and now - self._said[1] >= SAY_REPEAT_S):
            self._said = (key, now)
            label = "taken" if d.step == "taken" else f"re-arm {d.step}"
            self._say((f"{label}: {d.outcome}" + ("" if d.reason == d.step else f" {d.reason}"))[:120])

    def _hold_poll(self):
        status = self._check()
        if status.phase == "LOOP_FROZEN":
            raise _Stop(status)
        return status

    def _poll(self):
        if self.console.stop_pressed():
            raise _Stop()
        return self._hold_poll()

    def _gate(self, prompt, poll=None, stop=False, min_after=0.0):
        (poll or self._check)()
        if stop and self.console.stop_pressed():  # an Enter typed before the prompt is a stop, not an answer
            raise _Stop()
        self.console.wait_enter(prompt, poll or self._check, min_after=min_after)
        self.evidence.event(self.n, "gate", prompt=prompt)

    def _wait(self, *targets):
        while True:
            s = self._poll()
            if (s.phase, s.loop_stage) in targets:
                return s
            if s.phase not in MOVING:
                raise MissionInterrupted(f"Unexpected loop state {s.phase} (stage {s.loop_stage})")
            time.sleep(POLL_S)

    def _trial_ended(self, status):
        if self.active:
            self.evidence.event(self.n, "trial_end", reason=status.trial_end_reason)
            self._say(f"trial {self.n}: end ({status.trial_end_reason})")
            self.active, self.n = False, self.n + 1

    def _recorded(self, trial):
        return (self.session_dir / f"trial_{trial:03d}_frames.npz").exists()

    def _remaining(self):  # a trial whose files are still being written counts as recorded
        return remaining_trials(self.session_dir, self.n_trials) - sum(not self._recorded(t) for t in self.started)

    def _end_requested(self):
        return (self.session_dir / "end_session.request").exists()

    def _taken_gate(self, trial):
        """ "bottle taken?": ends on the operator's Enter (R25, debounced like "clear?"), the native torque take
        detector (R26, Status.take_detected) or, with taken_vlm_frames > 0, that many VLM "taken" yeses in a row
        (R29: each from a distinct frame; any other decision restarts the count; R33: checks start at least
        TAKEN_VLM_PERIOD_S apart), whichever comes first.

        Returns True when any decision saw the bottle on the robot hand (bottle_on_robot_hand true).
        """
        self._vlm_begin("taken", trial)
        seen_on_hand, polls, empty, last_yes, last_check = [], [0], [0], [None], [None]

        def poll():
            s = self._hold_poll()
            if (s.phase, s.loop_stage) != ("LOOP_HOLD", "await_take"):
                raise MissionInterrupted(f"Taken gate needs the await_take hold, not {s.phase} (stage {s.loop_stage})")
            polls[0] += 1
            # the first call is _gate's pre-prompt check: a take seen there ends the gate on the next poll, after
            # the pre-prompt stop check and the prompt
            if polls[0] > 1 and s.take_detected is True:
                raise _Taken("torque")
            d = self._vlm_collect("taken", trial)
            if d is not None:
                # a malformed reply is stored with parsed None: it counts as neither on the hand nor empty
                if (((d.answers or {}).get("taken") or {}).get("parsed") or {}).get("bottle_on_robot_hand") is True:
                    seen_on_hand.append(d.frame_id)
                empty[0] = empty[0] + 1 if d.outcome == "yes" else 0
                last_yes[0] = time.monotonic() if d.outcome == "yes" else None
                if self.taken_vlm_frames and empty[0] >= self.taken_vlm_frames:
                    raise _Taken("vlm")
            if empty[0] and time.monotonic() - last_yes[0] > TAKEN_EMPTY_FRESH_S:  # no fresh camera evidence
                empty[0], last_yes[0] = 0, None
            if polls[0] > 1 and empty[0] >= TAKEN_EMPTY_FRAMES and s.take_hand_empty is True:  # R32
                raise _Taken("camera+wrist")
            now = time.monotonic()
            if self.taken_vlm_frames and last_check[0] is not None and now - last_check[0] < TAKEN_VLM_PERIOD_S:
                return  # R33
            if self._worker is None:
                self._vlm_dispatch(trial)
                if self._worker is not None:
                    last_check[0] = now

        try:
            self._gate("bottle taken? (offer pose held; moves on when the arm feels the bottle lifted off, the camera "
                       "sees the hand empty, or Enter) "
                       f"[presses in the first {CLEAR_DEBOUNCE_S:g} s are ignored]", poll, stop=True,
                       min_after=CLEAR_DEBOUNCE_S)
            by = "operator"
        except _Taken as taken:
            self.console.flush()  # an Enter typed for this gate must not become a stop during the move to ready
            by = taken.by
            self._say({"torque": "taken: detected by arm torque",
                       "camera+wrist": "taken: the camera sees the hand empty and the wrist carries no bottle",
                       }.get(by, f"taken: the camera saw the hand empty {self.taken_vlm_frames} times in a row"))
        self.evidence.event(trial, "taken_confirmed", by=by, vlm_saw_on_hand=bool(seen_on_hand))
        return bool(seen_on_hand)

    def _rearm(self, step):
        """Two distinct-frame yeses at the ready hold; False when the session end is requested meanwhile."""
        self._vlm_begin(step, self.n)
        while True:
            s = self._poll()
            if (s.phase, s.loop_stage) != ("LOOP_HOLD", "ready"):
                raise MissionInterrupted(f"Re-arm needs the ready hold, not {s.phase} (stage {s.loop_stage})")
            if self._end_requested():
                return False
            d = self._vlm_collect(step, self.n)
            if d is not None and d.complete:
                self.evidence.event(self.n, "rearm_end", step=step, frame_id=d.frame_id)
                return True
            self._vlm_dispatch(self.n)
            time.sleep(POLL_S)

    @staticmethod
    def _kind(step):
        return "taken" if step == "taken" else "rearm"

    def _vlm_begin(self, step, trial):
        while self._worker is not None and self._worker[0].is_alive():  # a check left by a stop or a gate
            self._poll()
            time.sleep(POLL_S)
        self._worker = None  # its result, if any, belongs to the previous step: dropped
        self.monitor.begin(step, None)  # RearmMonitor ignores the initial frame
        self.evidence.event(trial, f"{self._kind(step)}_begin", step=step)

    def _vlm_collect(self, step, trial):
        """The finished check's decision (logged and printed), or None while it runs or none is in flight."""
        if self._worker is None or self._worker[0].is_alive():
            return None
        _, out, path = self._worker
        self._worker = None
        decision, error = out.get_nowait()
        if error:
            raise MissionInterrupted(f"VLM monitor error ({self._kind(step)}): {error}")
        self.evidence.event(trial, f"{self._kind(step)}_decision", **asdict(decision), frame_path=path)
        self._say_decision(decision)
        return decision

    def _vlm_dispatch(self, trial):
        """Observe (main thread) and start one check when none is in flight; newer frames are not fetched meanwhile."""
        if self._worker is not None:
            return
        try:
            frame = self._client.observe()
        except RPCError as exc:
            if exc.code != "UNAVAILABLE":
                raise
            return
        try:
            path = self.evidence.frame(trial, frame)
        except (binascii.Error, ValueError):  # a corrupt frame: skip it, as the monitor would (invalid_image)
            self.evidence.event(trial, "frame_invalid", frame_id=frame.frame_id)
            return
        out = queue.Queue(maxsize=1)

        def check():
            try:
                out.put((self.monitor.check(frame), None))
            except Exception as exc:
                out.put((None, exc))

        self._worker = (threading.Thread(target=check, daemon=True), out, path)
        self._worker[0].start()

    def _release(self):
        deadline = time.monotonic() + RELEASE_RETRY_S
        while True:
            try:
                return self._rpc("release_control", {})
            except RPCError as exc:
                if exc.code != "NOT_READY" or time.monotonic() >= deadline:
                    raise
            time.sleep(POLL_S)

    def _cancel_and_release(self):
        """Error path: release as is. In a LOOP phase that freezes the loop for a later reclaim (R16b)."""
        if self._lease is None:
            return
        try:
            if self._client.get_status().owner_session_id == self._session_id:
                self._release()
        except Exception as exc:
            self._log("cleanup_unconfirmed", reason=str(exc))
