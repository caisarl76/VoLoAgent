"""Parity of the native autonomous-loop pieces with lerobot a6e267c6's own code, on the s16 and s19 G1 sessions.

Usage (the lerobot venv, as g1_handover_20261021_s10_loop_sim/parity_check.py):
    ~/work/lerobot/.venv/bin/python parity_check.py OUT.json

lerobot's files are taken from git (`git show a6e267c6:...`) into a temp dir; ~/work/lerobot is not touched.
Native modules come from the g1-loop-native worktree. Every trial is compared under its own recorded `args`.

1. token_stream: native PoseCommander blend in + follow(TokenStream), with the controller's R1b handoff, vs
   SonicBackend.begin / set_chunk / episode_tick (ChunkResampler + send() slew). The raw chunks were not logged,
   so the chunks are 40-row windows of the logged output (plain and randomly perturbed) delivered at the logged
   arrival ticks with the logged t0, plus a synthetic arrival-spacing sweep. Native runs on a monotonic time base
   T0 + t_ep (T0 = 0 and a realistic uptime). The blend in is also compared with the logged blend-in ticks.
2. handover_detector: native vs lerobot HandoverDetector on each trial's logged right-hand mean (state[:, 21:28])
   from the first episode tick to the trial end, now = the logged wall time.
3. motion: native LoopController take_over -> raise_ -> end vs SonicBackend take_over -> raise_to_ready ->
   finish() (fake publisher recording send(); ticks() not sleeping), and the native commander segments vs every
   logged non-policy segment (planner token, wait first chunk, to ready, blend out, hold rest, handback blend).
4. episode_guards: lerobot's episode_end strings (its run_episode assignments and arm_watchdog, evaluated from
   its source with ast) vs native EpisodeGuards reasons and categories on forced conditions, both guards
   replayed on the logged episodes, and the logged episode_end strings mapped to native reasons.
"""
import argparse
import ast
import datetime
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace

import numpy as np

LEROBOT, COMMIT = Path.home() / "work/lerobot", "a6e267c6"
NATIVE = Path("/home/jihun/work/VoLoAgent/.worktrees/g1-loop-native")
SESSIONS = Path("/mnt/data/jihun/g1_eval_sessions")
SESSION_NAMES = ("s16_wbt_groot_handover261015_he_blend", "s19_wbt_groot_handover261015_he_ready_grid")
LR_FILES = ("sonic_policy_streamer.py", "stream_backends.py", "sonic_token_stream.py", "eval_session.py",
            "sonic_targets.py", "wbc_common.py", "ready_pose_261020_ep76.npz")
SETTING_KEYS = ("chunk_blend_s", "chunk_anchor_s", "max_token_step", "end_on_handover", "ready_pose", "policy_fps",
                "replan_s", "hold_rest_s", "handoff_blend_s", "start", "end", "dex3_right_order", "duration_s")
TOKEN_TOL = 1e-6
T0_UPTIME = 250000.123456  # a realistic time.monotonic() (about 2.9 days of uptime)
ROWS = 40

LR = Path(tempfile.mkdtemp(prefix=f"lerobot_{COMMIT}_"))
for name in LR_FILES:
    (LR / name).write_bytes(subprocess.run(
        ["git", "-C", str(LEROBOT), "show", f"{COMMIT}:examples/g1_dex3_training/{name}"],
        check=True, capture_output=True).stdout)
sys.path[:0] = [str(NATIVE), str(LR)]
import eval_session as lr_eval  # noqa: E402
import sonic_policy_streamer as streamer  # noqa: E402
import stream_backends as backends  # noqa: E402

from gear_sonic.utils.inference import loop_control  # noqa: E402
from gear_sonic.utils.inference.lerobot_sonic_policy import TokenStream  # noqa: E402
from gear_sonic.utils.inference.loop_episode import EpisodeGuards, HandoverDetector  # noqa: E402
from gear_sonic.utils.inference.loop_motion import PoseCommander, load_ready_pose  # noqa: E402

backends.ticks = range  # ticks(n) without its 20 ms sleeps; the sent values do not depend on the clock
GATE = Path(tempfile.mkdtemp(prefix="gate_"))
STREAMER_TREE = ast.parse((LR / "sonic_policy_streamer.py").read_text())


# ---------------------------------------------------------------------------------------------- shared helpers
class FakePub:
    """The backend's ZMQ PUB: keeps every send()."""

    def __init__(self):
        self.sent = []

    def send(self, msg):
        self.sent.append(msg)


def decode(msg):
    """stream_backends' pose (v4) and command (v1) messages."""
    for topic in ("pose", "command"):
        if msg.startswith(topic.encode()):
            payload = msg[len(topic) + backends.HEADER_SIZE:]
            break
    if topic == "command":
        return ("command", bool(payload[0]), bool(payload[2]))  # start, planner
    token = np.frombuffer(payload[:256], "<f4")
    hands = np.r_[np.frombuffer(payload[264:292], "<f4"), np.frombuffer(payload[292:320], "<f4")]
    return ("pose", token, hands)


def lerobot_backend(args, planner=None):
    a = SimpleNamespace(
        max_token_step=args["max_token_step"], action_space="sonic78", chunk_anchor_s=args["chunk_anchor_s"],
        ready_pose=str(LR / "ready_pose_261020_ep76.npz"), gate_dir=GATE, handoff_blend_s=args["handoff_blend_s"],
        start=args["start"], end=args["end"])
    sent, pub = [], FakePub()
    state = SimpleNamespace(latest=lambda: {"token_state": planner})
    backend = backends.SonicBackend(
        a, pub, state, lambda token, hands, phase, *_: sent.append((token.copy(), np.array(hands), phase)),
        args["policy_fps"], None)
    return backend, sent, pub


def settings(args):
    return {k: args.get(k) for k in SETTING_KEYS}


def load_session(name):
    raw = SESSIONS / name / "raw"
    trials = []
    for path in sorted(raw.glob("trial_[0-9][0-9][0-9].npz")):
        z = np.load(path, allow_pickle=False)
        log = {k: z[k] for k in z.files}
        log["phase"] = np.array([str(p) for p in log["phase"]])
        trials.append(dict(index=int(path.stem[-3:]), log=log, args=json.loads(str(z["args"])),
                           episode_end=str(z["episode_end"])))
    return trials


def runs(phase):
    """[(label, first index, length)] of the consecutive phase runs."""
    out, start = [], 0
    for i in range(1, len(phase) + 1):
        if i == len(phase) or phase[i] != phase[start]:
            out.append((str(phase[start]), start, i - start))
            start = i
    return out


# ---------------------------------------------------------------------------------------------- 1. token stream
def replay_stream(args, rest, first, t_ep, chunks, t0_offset):
    """chunks: {episode tick: (t0, rows)} for the replans (tick 0 is `first`, t0 0). Returns both outputs."""
    fps, blend, step, anchor = args["policy_fps"], args["chunk_blend_s"], args["max_token_step"], args["chunk_anchor_s"]
    backend, sent, _ = lerobot_backend(args)
    backend.rest_token, backend.rest_hands = rest[0].copy(), rest[1].copy()
    backend.last_sent, backend.last_hands = rest[0].copy(), rest[1].copy()  # the robot holds the rest pose
    backend.begin(first, None)
    for j, t in enumerate(t_ep):
        if j in chunks:
            backend.set_chunk(chunks[j][1], None, chunks[j][0], blend, t)
        backend.episode_tick(t, j, 0.0)
    # native: LoopController's blend in, _start_episode and _set_chunk (R1b) around PoseCommander.follow
    commander = PoseCommander(step)
    commander.start(*rest)
    commander.blend_to(first[0, :64], first[0, 64:], loop_control.BLEND_TICKS, "blend in")
    native = [commander.tick(t0_offset - 1.0) for _ in range(loop_control.BLEND_TICKS)]
    stream = TokenStream(fps, blend, step, anchor)
    for j, t in enumerate(t_ep):
        now = t0_offset + t
        if j == 0 or j in chunks:
            rows, t0 = (first, now) if j == 0 else (chunks[j][1], t0_offset + chunks[j][0])
            stream.last_token = commander.last_token.copy()
            stream.set_chunk(rows, t0=t0, now=now)
            if j == 0:
                commander.follow(stream, "episode")
        native.append(commander.tick(now))
    lr_tok = np.array([s[0] for s in sent])
    lr_hands = np.array([s[1] for s in sent], np.float32)
    nt_tok = np.array([n[0] for n in native])
    nt_hands = np.array([n[1] for n in native])
    n = loop_control.BLEND_TICKS
    tok_diff = np.abs(lr_tok - nt_tok)
    return dict(
        ticks=len(sent), chunks=len(chunks) + 1,
        blend_in_identical=bool(np.array_equal(lr_tok[:n], nt_tok[:n]) and np.array_equal(lr_hands[:n], nt_hands[:n])),
        labels_identical=[s[2] for s in sent] == [x[2] for x in native],
        max_abs_token_diff=float(tok_diff.max()),
        hands_equal=bool(np.array_equal(lr_hands, nt_hands)),
        max_abs_hand_diff=float(np.abs(lr_hands - nt_hands).max()),
        lerobot_slewed_ticks=int(backend.slewed),
    ), (lr_tok, lr_hands)


def stream_case_pass(r, gated):
    """The gate is T0 = 0, lerobot's own time base, where both sides see the same float times. The uptime base is
    reported only (None): a shifted base rounds an exact cross-fade end (w == 1.0) onto a different tick."""
    if not gated:
        return None
    return (r["blend_in_identical"] and r["labels_identical"] and r["max_abs_token_diff"] <= TOKEN_TOL
            and r["hands_equal"])


def token_stream_trial(trial, rng):
    log, args = trial["log"], trial["args"]
    ph = log["phase"]
    wait, blend_in, ep = (np.flatnonzero(ph == p) for p in ("wait first chunk", "blend in", "episode"))
    rest = (log["token"][wait[-1]].astype(np.float32), log["hands"][wait[-1]].astype(np.float32))
    t_ep = [float(t) for t in log["t_ep"][ep]]
    chunk_t0 = log["chunk_t0"][ep]
    rows_all = np.hstack([log["token"][ep], log["hands"][ep]]).astype(np.float32)
    arrivals = np.flatnonzero(np.diff(chunk_t0) != 0) + 1

    def window(t0):
        start = int(np.searchsorted(np.asarray(t_ep), t0 - 1e-9))
        rows = rows_all[start:start + ROWS]
        return np.vstack([rows, np.repeat(rows[-1:], ROWS - len(rows), 0)])

    out = {}
    for variant in ("window", "perturbed"):
        first = window(0.0)
        chunks = {int(i): (float(chunk_t0[i]), window(chunk_t0[i])) for i in arrivals}
        if variant == "perturbed":  # chunks that disagree: non-trivial anchor offsets, cross-fades and slew
            for i, (t0, rows) in chunks.items():
                rows = rows.copy()
                rows[:, :64] += rng.normal(0, 0.15, 64).astype(np.float32) + rng.normal(0, 0.01, (ROWS, 64)).astype(np.float32)
                rows[:, 64:] += rng.normal(0, 0.05, 14).astype(np.float32)
                chunks[i] = (t0, rows)
        for name, offset in (("t0_0", 0.0), ("t0_uptime", T0_UPTIME)):
            r, (lr_tok, lr_hands) = replay_stream(args, rest, first, t_ep, chunks, offset)
            r["pass"] = stream_case_pass(r, gated=offset == 0.0)
            if variant == "window" and offset == 0.0:  # the blend in is fully determined by the log
                n = len(blend_in)
                r["blend_in_matches_log"] = bool(
                    np.array_equal(lr_tok[:n], log["token"][blend_in].astype(np.float32))
                    and np.array_equal(lr_hands[:n], log["hands"][blend_in].astype(np.float32)))
                r["pass"] = r["pass"] and r["blend_in_matches_log"]
            out[f"{variant}/{name}"] = r
    return out


def token_stream_sweep(args, rng):
    """Synthetic: a 30 s episode, arrival spacings 5..30 ticks (every value), then 15 and 16 ticks repeated
    (an arrival on the tick at which the previous 0.3 s cross-fade completes), then random ones."""
    n = int(args["duration_s"] / backends.TICK)
    rest = (rng.normal(0, 0.3, 64).astype(np.float32), rng.uniform(0, 0.4, 14).astype(np.float32))
    walk = np.cumsum(rng.normal(0, 0.02, (n + ROWS, 78)), 0).astype(np.float32)
    t_ep = [i * backends.TICK for i in range(n)]
    spacing = np.r_[np.arange(5, 31), np.tile([15, 16], 20), rng.integers(5, 31, 40)]
    chunks, tick = {}, 0
    for s in spacing:
        tick += int(s)
        if tick >= n:
            break
        t0 = max(t_ep[tick] - float(rng.uniform(0.15, 0.5)), 0.0)
        rows = walk[tick:tick + ROWS] + rng.normal(0, 0.1, 78).astype(np.float32)
        chunks[tick] = (t0, rows)
    first = walk[:ROWS].copy()
    out = {}
    for name, offset in (("t0_0", 0.0), ("t0_uptime", T0_UPTIME)):
        r, _ = replay_stream(args, rest, first, t_ep, chunks, offset)
        r["pass"] = stream_case_pass(r, gated=offset == 0.0)
        out[name] = r
    return out


ORIGINAL_SET_CHUNK = TokenStream.set_chunk


def candidate_fix(original):
    """The diagnosed difference: lerobot's anchored set_chunk first evaluates ``ref = self.token_at(now)``,
    which ends a cross-fade that has completed by ``now`` (``_prev = None``), so the outgoing chunk itself becomes
    the hands' fade source. Native takes ref from last_token and never evaluates the output first."""

    def set_chunk(self, rows, t0, now):
        if self.anchor_s > 0 and self._chunk is not None:
            self._row_at(now)
        return original(self, rows, t0, now)

    return set_chunk


# ---------------------------------------------------------------------------------------------- 2. detector
def detector_trial(trial):
    log, args = trial["log"], trial["args"]
    ep = np.flatnonzero(log["phase"] == "episode")
    series = log["state"][ep[0]:, 21:28].astype(np.float32).mean(axis=1)
    wall = log["wall"][ep[0]:]
    open_s = args.get("end_on_handover") or 1.0  # s16 ran without the detector: its default is used
    native, lerobot = HandoverDetector(open_s=open_s), lr_eval.HandoverDetector(open_s=open_s)
    nat = [native.update(float(x), float(w)) for x, w in zip(series, wall)]
    lr = [lerobot.update(float(x), float(w)) for x, w in zip(series, wall)]
    fire = int(np.argmax(nat)) if any(nat) else None
    n_ep = len(ep)
    enabled = bool(args.get("end_on_handover"))
    r = dict(identical=nat == lr, fire_tick=fire, episode_ticks=n_ep, detector_enabled=enabled,
             episode_end=trial["episode_end"])
    if enabled:  # lerobot breaks before logging the firing tick: the first post-episode tick is its state
        r["fires_where_logged"] = trial["episode_end"] == "handover complete" and fire == n_ep
        if fire is not None and fire != n_ep:  # the open stretch's length on the logged (record-time) stamps
            closed = series > native.closed_above
            since = max(i for i in range(1, fire + 1) if closed[i - 1] and not closed[i])
            r["open_s_on_logged_stamps"] = {f"tick_{k}": float(wall[k] - wall[since]) for k in (fire, n_ep) if k < len(wall)}
    else:
        r["would_fire_s_before_operator_stop"] = None if fire is None else round(float(wall[n_ep - 1] - wall[fire]), 3)
    r["pass"] = r["identical"]
    return r


# ---------------------------------------------------------------------------------------------- 3. motion
class Hooks:
    def __init__(self, events):
        self.events = events

    def switch_to_pose(self):
        self.events.append(("command", True, False))

    def switch_to_planner(self):
        self.events.append(("command", True, True))

    def request_policy_chunk(self):
        raise AssertionError("no policy in the motion check")


def motion_full(args, planner):
    """take over -> raise -> (session end) lower -> hand back, both pipelines end to end."""
    (GATE / "settled.0").touch()
    backend, sent, pub = lerobot_backend(args, planner)
    backend.take_over(0, 1)
    backend.raise_to_ready("raise")
    backend.finish()
    lr = [decode(m) for m in pub.sent]

    events = []
    ctl = loop_control.LoopController(
        PoseCommander(args["max_token_step"]), load_ready_pose(), Hooks(events),
        make_stream=lambda: None, right_order=np.arange(7), trial_args={}, model="parity")
    now = [T0_UPTIME]
    labels = []

    def run_until(done):
        while not done():
            token, hands, label = ctl.tick(now[0], None, 0.0, 0.0)
            now[0] += 0.02
            events.append(("pose", token, hands))
            labels.append(label)

    ctl.take_over({"token_state": planner}, now[0])
    run_until(lambda: ctl.phase == "LOOP_HOLD")
    ctl.raise_(now[0])
    run_until(lambda: ctl.phase == "LOOP_HOLD" and ctl.stage == "ready")
    ctl.end(now[0])
    run_until(lambda: ctl.phase == "COMPLETED")

    poses = lambda seq: [e for e in seq if e[0] == "pose"]  # noqa: E731
    commands = lambda seq: [(i - sum(1 for e in seq[:i] if e[0] == "command"), e)  # noqa: E731
                            for i, e in enumerate(seq) if e[0] == "command"]
    lp, npo = poses(lr), poses(events)
    same_len = len(lp) == len(npo)
    tok_eq = same_len and all(np.array_equal(a[1], b[1]) for a, b in zip(lp, npo))
    hand_eq = same_len and all(np.array_equal(a[2], b[2]) for a, b in zip(lp, npo))
    rec_eq = same_len and all(np.array_equal(s[0], b[1]) and np.array_equal(s[1], b[2]) for s, b in zip(sent, npo))
    # commands: (number of poses sent before it, command); native hooks run inside the tick that ends the segment
    lr_cmd = [(n, e) for n, e in commands(lr)]
    nt_cmd = [(n, e) for n, e in commands(events)]
    segments = [(lab, n) for lab, _, n in runs(np.array(labels))]
    r = dict(ticks_lerobot=len(lp), ticks_native=len(npo), tokens_identical=bool(tok_eq), hands_identical=bool(hand_eq),
             recorded_equals_published=bool(rec_eq), labels_identical=[s[2] for s in sent] == labels,
             segments=segments, lerobot_slewed_ticks=int(backend.slewed),
             lerobot_commands=[(n, list(e[1:])) for n, e in lr_cmd], native_commands=[(n, list(e[1:])) for n, e in nt_cmd])
    # lerobot sends a command after pose n; the native hook fires during tick n (before the main loop publishes it)
    r["command_offsets"] = [ln - nn for (ln, _), (nn, _) in zip(lr_cmd, nt_cmd)]
    r["commands_same"] = [e for _, e in lr_cmd] == [e for _, e in nt_cmd]
    r["pass"] = tok_eq and hand_eq and rec_eq and r["labels_identical"] and r["commands_same"]
    return r


def commander_segment(start, segments, step):
    """Native PoseCommander from (token, hands) through LoopController's segment calls."""
    c = PoseCommander(step)
    c.start(*start)
    n = 0
    for kind, label, a, b, ticks in segments:
        (c.blend_to(a, b, ticks, label) if kind == "blend" else
         c.play(np.tile(a, (ticks, 1)), np.tile(b, (ticks, 1)), label))
        n += ticks
    out = [c.tick(0.0) for _ in range(n)]
    return np.array([o[0] for o in out]), np.array([o[1] for o in out]), [o[2] for o in out]


def motion_logged(trial, ready):
    """Every logged segment that the native commander produces without a policy, compared tick by tick."""
    log, args = trial["log"], trial["args"]
    step = args["max_token_step"]
    tok, hands, ph = log["token"].astype(np.float32), log["hands"].astype(np.float32), log["phase"]
    zero = np.zeros(14, np.float32)
    checks = []
    rr = runs(ph)
    planner = None
    for k, (label, i, n) in enumerate(rr):
        prev = (tok[i - 1], hands[i - 1]) if i else None
        if label == "planner token":
            planner = tok[i]
            seg = [("play", label, tok[i], zero, n)]
            start = (tok[i], zero)
        elif label == "wait first chunk":  # lerobot wait_tick: the rest pose (ready pose or planner stance)
            rest = (ready.token, ready.hands) if args.get("ready_pose") else (planner, zero)
            if rest[0] is None:
                continue
            seg, start = [("play", label, rest[0], rest[1], n)], prev or rest
        elif label == "to ready":
            seg, start = [("blend", label, ready.token, ready.hands, n)], prev
        elif label == "blend out":  # finish(): back to the stance, hold it, hand back (one native chain)
            tail = rr[k:k + 3]
            if [t[0] for t in tail] != ["blend out", "hold rest", "handback blend"] or planner is None:
                continue
            seg = [("blend", "blend out", planner, zero, tail[0][2]), ("play", "hold rest", planner, zero, tail[1][2]),
                   ("blend", "handback blend", planner, zero, tail[2][2])]
            start, n = prev, sum(t[2] for t in tail)
        else:
            continue
        nt, nh, nl = commander_segment(start, seg, step)
        checks.append(dict(segment=label, start=i, ticks=n, identical=bool(
            np.array_equal(nt, tok[i:i + n]) and np.array_equal(nh, hands[i:i + n]) and nl == list(ph[i:i + n]))))
    return checks


# ---------------------------------------------------------------------------------------------- 4. guards
def lerobot_strings(args, state_age, camera_age, arm_reason):
    """lerobot's run_episode `episode_end = ...` assignments, in source order, evaluated from its source."""
    run_episode = next(n for n in ast.walk(STREAMER_TREE) if isinstance(n, ast.FunctionDef) and n.name == "run_episode")
    ns = dict(a=SimpleNamespace(**args), state=SimpleNamespace(age=lambda: state_age),
              images=SimpleNamespace(age=lambda: camera_age), reason=arm_reason)
    out = []
    for node in ast.walk(run_episode):
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "episode_end" for t in node.targets):
            out.append((node.lineno, eval(compile(ast.Expression(node.value), "lerobot", "eval"), ns)))  # noqa: S307
    return [s for _, s in sorted(out)]


def lerobot_arm_watchdog(args, history):
    fn = next(n for n in ast.walk(STREAMER_TREE) if isinstance(n, ast.FunctionDef) and n.name == "arm_watchdog")
    ns = dict(np=np, history=history, a=SimpleNamespace(**args), ARM_LO=streamer.ARM_LO, ARM_HI=streamer.ARM_HI,
              ARM_SPEED_TICKS=streamer.ARM_SPEED_TICKS, max_arm_speed=backends.max_arm_speed)
    exec(ast.unparse(fn), ns)  # noqa: S102
    return ns["arm_watchdog"]


def msg_of(state, token_state=None):
    body = np.zeros(29, np.float32)
    body[15:29] = state[:14]
    msg = dict(body_q=body, left_hand_q=state[14:21], right_hand_q=state[21:28])
    if token_state is not None:
        msg["token_state"] = token_state
    return msg


def guard_kwargs(args):
    return dict(cap_s=args["duration_s"], max_arm_speed=args["max_arm_speed"],
                arm_range_margin=args["arm_range_margin"], max_chunk_age_s=args["max_chunk_age_s"],
                max_state_age_s=args["max_state_age_s"], max_camera_age_s=args["max_camera_age_s"],
                token_echo_timeout_s=args["token_echo_timeout_s"])


def arm_stream(args, states, walls, blend_in=0):
    """Both arm watchdogs on one stream; the first `blend_in` entries are blend-in ticks (lerobot history only).
    lerobot checks before the tick's send() appends its state; native appends first (Note for T12)."""
    history = []
    lw = lerobot_arm_watchdog(args, history)
    guards = EpisodeGuards(**{**guard_kwargs(args), "token_echo_timeout_s": 0.0})
    lr_fire = nt_fire = None
    for j, (s, w) in enumerate(zip(states, walls)):
        if j >= blend_in:
            if j == blend_in:
                guards.start(w)
            if lr_fire is None and (reason := lw()):
                lr_fire = (j - blend_in, f"watchdog: {reason}")
            v = guards.check(w, msg_of(s), 0.0, 0.0, w, [])
            if nt_fire is None and v is not None:
                nt_fire = (j - blend_in, v)
        history.append((0.0, s, w))
    return lr_fire, nt_fire


def guards_forced(args, rng):
    """Each guard forced on its own; native (category, reason) vs lerobot's string from its source."""
    base = rng.uniform(-0.3, 0.3, 28).astype(np.float32)
    base[:14] = (streamer.ARM_LO + streamer.ARM_HI) / 2

    def native(now, state_age=0.0, camera_age=0.0, last_chunk=None, echo=None, states=None):
        g = EpisodeGuards(**guard_kwargs(args))
        g.start(100.0)
        recent = [np.zeros(64, np.float32)]
        v = None
        for k, s in enumerate(states or [base]):
            t = 100.0 + (now - 100.0) * (k + 1) / len(states or [base])
            v = g.check(t, msg_of(s, np.zeros(64, np.float32) if echo is None else echo), state_age, camera_age,
                        100.0 if last_chunk is None else last_chunk, recent)
            if v:
                break
        return v

    fast = [base.copy() for _ in range(6)]
    for k in range(6):
        fast[k][3] += 0.2 * k  # 10 rad/s on arm joint 3
    out_range = base.copy()
    out_range[5] = streamer.ARM_HI[5] + 0.4
    walls = [100.0 + 0.02 * k for k in range(6)]
    lr_fast, nt_fast = arm_stream(args, fast, walls)
    lr_range, nt_range = arm_stream(args, [out_range] * 6, walls)
    cases = {
        "chunk": native(101.02, last_chunk=100.0),
        "state": native(100.02, state_age=0.25),
        "camera": native(100.02, camera_age=0.6),
        "echo": native(100.6, echo=np.ones(64, np.float32), states=[base] * 30),
        "cap": native(100.0 + args["duration_s"]),
        "arm_speed": nt_fast[1] if nt_fast else None,
        "arm_range": nt_range[1] if nt_range else None,
    }
    lr_speed = lerobot_strings(args, 0.25, 0.6, lr_fast[1].removeprefix("watchdog: ") if lr_fast else "")
    lr_range_s = lerobot_strings(args, 0.25, 0.6, lr_range[1].removeprefix("watchdog: ") if lr_range else "")
    # source order: completed, operator stop, no valid chunk, robot state, camera, echo, watchdog, handover
    lr = dict(cap=lr_speed[0], chunk=lr_speed[2], state=lr_speed[3], camera=lr_speed[4], echo=lr_speed[5],
              arm_speed=lr_speed[6], arm_range=lr_range_s[6])
    spec = dict(cap="end", chunk="end", state="freeze", camera="freeze", echo="freeze", arm_speed="end", arm_range="end")
    rows = {k: dict(lerobot=lr[k], native=list(cases[k]) if cases[k] else None,
                    string_equal=bool(cases[k]) and cases[k][1] == lr[k],
                    category_as_spec=bool(cases[k]) and cases[k][0] == spec[k]) for k in spec}
    timing = dict(arm_speed_fire_tick=dict(lerobot=lr_fast[0] if lr_fast else None, native=nt_fast[0] if nt_fast else None),
                  arm_range_fire_tick=dict(lerobot=lr_range[0] if lr_range else None, native=nt_range[0] if nt_range else None))
    # a fast last stretch of the blend in, then a still arm: lerobot's history holds the blend-in ticks (t_ep 0)
    blend = [base.copy() for _ in range(8)]
    for k in range(5):
        blend[k][3] += 0.2 * k
    for k in range(5, 8):
        blend[k][3] = blend[4][3]
    walls8 = [100.0 + 0.02 * k for k in range(8)]
    lr_b, nt_b = arm_stream(args, blend, walls8, blend_in=5)
    timing["fast_blend_in_end"] = dict(lerobot=list(lr_b) if lr_b else None, native=list(nt_b) if nt_b else None)
    others = dict(zip(("operator_stop", "handover"), (lr_speed[1], lr_speed[7])))
    native_src = (NATIVE / "gear_sonic/utils/inference/harness_control.py").read_text()
    ctl_src = (NATIVE / "gear_sonic/utils/inference/loop_control.py").read_text()
    rows["operator_stop"] = dict(lerobot=others["operator_stop"], native=["freeze", "operator stop"],
                                 string_equal=f'self.loop.freeze("{others["operator_stop"]}", now)' in native_src,
                                 category_as_spec=True)  # spec: a stop freezes the robot
    rows["handover"] = dict(lerobot=others["handover"], native=["end", "handover complete"],
                            string_equal=f'verdict = ("end", "{others["handover"]}")' in ctl_src, category_as_spec=True)
    return rows, timing


def native_priority(args):
    """Native first-hit order with every condition true at once, removing the winner each round. 25 quiet ticks
    (only the echo mismatch, when active, runs its 0.5 s timer) build the history; the last tick has them all."""
    order, active = [], {"cap", "chunk", "state", "camera", "echo", "arm"}
    mid = np.zeros(28, np.float32)
    mid[:14] = (streamer.ARM_LO + streamer.ARM_HI) / 2
    far = mid.copy()
    far[:14] = streamer.ARM_LO - 1.0
    while active:
        g = EpisodeGuards(**guard_kwargs(args))
        g.start(0.0)
        echo = np.ones(64, np.float32) if "echo" in active else np.zeros(64, np.float32)
        recent = [np.zeros(64, np.float32)]
        for k in range(25):  # up to 0.48 s: the echo timer has not run out yet
            assert g.check(0.02 * k, msg_of(mid, echo), 0.0, 0.0, 0.02 * k, recent) is None
        t = 0.6 + (args["duration_s"] if "cap" in active else 0.0)
        v = g.check(t, msg_of(far if "arm" in active else mid, echo), 0.3 if "state" in active else 0.0,
                    0.7 if "camera" in active else 0.0, t - 5.0 if "chunk" in active else t, recent)
        name = next(k for k, s in (("cap", "completed"), ("chunk", "no valid"), ("state", "robot state"),
                                   ("camera", "camera"), ("echo", "deploy token_state"), ("arm", "watchdog"))
                    if v and v[1].startswith(s))
        order.append(name)
        active.discard(name)
    return order


def guards_logged(trial):
    """Both guards on the logged episode: robot state and wall times as logged, ages 0 (not logged), no echo
    (token_state not logged), chunk arrivals at the logged ticks. lerobot's arm history includes the blend in."""
    log, args = trial["log"], trial["args"]
    ph = log["phase"]
    bi, ep = np.flatnonzero(ph == "blend in"), np.flatnonzero(ph == "episode")
    states, wall, chunk_t0, t_ep = log["state"].astype(np.float32), log["wall"], log["chunk_t0"], log["t_ep"]
    history = [(0.0, states[i], float(wall[i])) for i in bi]
    lw = lerobot_arm_watchdog(args, history)
    guards = EpisodeGuards(**{**guard_kwargs(args), "token_echo_timeout_s": 0.0})
    guards.start(float(wall[ep[0]]))
    lr_end = nt_end = None
    last_chunk_wall, last_chunk_t, top_speed = float(wall[ep[0]]), 0.0, 0.0
    for j, i in enumerate(ep):
        if j and chunk_t0[i] != chunk_t0[ep[j - 1]]:
            last_chunk_wall, last_chunk_t = float(wall[i]), float(t_ep[i])
        if lr_end is None:
            if float(t_ep[i]) - last_chunk_t > args["max_chunk_age_s"]:
                lr_end = (j, "no valid chunk")
            elif reason := lw():
                lr_end = (j, f"watchdog: {reason}")
        v = guards.check(float(wall[i]), msg_of(states[i]), 0.0, 0.0, last_chunk_wall, [])
        if nt_end is None and v is not None:
            nt_end = (j, list(v))
        history.append((float(t_ep[i]), states[i], float(wall[i])))
        if j >= 3:
            sp = backends.max_arm_speed(float(wall[ep[j - 3]]), states[ep[j - 3]], float(wall[i]), states[i])
            top_speed = max(top_speed, sp[1] if sp else 0.0)
    blend_speed = max((backends.max_arm_speed(float(wall[bi[k - 3]]), states[bi[k - 3]], float(wall[bi[k]]), states[bi[k]])
                       or (0, 0.0))[1] for k in range(3, len(bi)))
    return dict(episode_ticks=len(ep), lerobot_end=lr_end, native_end=nt_end, episode_end=trial["episode_end"],
                max_arm_speed_episode=round(top_speed, 3), max_arm_speed_blend_in=round(float(blend_speed), 3),
                pass_=lr_end is None and nt_end is None)


# ---------------------------------------------------------------------------------------------- main
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("out", type=Path)
    out_path = parser.parse_args().out
    rng = np.random.default_rng(12)
    ready = load_ready_pose()
    lr_ready = np.load(LR / "ready_pose_261020_ep76.npz")
    asset_same = all(np.array_equal(getattr(ready, k), lr_ready[k].astype(np.float32))
                     for k in ("token", "hands", "state", "raise_tokens", "raise_hands", "lower_tokens", "lower_hands"))
    native_head = subprocess.run(["git", "-C", str(NATIVE), "rev-parse", "--short", "HEAD"],
                                 capture_output=True, text=True).stdout.strip()
    result = dict(generated=datetime.datetime.now().isoformat(timespec="seconds"), lerobot_commit=COMMIT,
                  native_head=native_head, numpy=np.__version__, ready_pose_asset_identical=asset_same,
                  token_tolerance=TOKEN_TOL, t0_uptime=T0_UPTIME, sessions={}, checks={}, observations={})
    checks = {k: dict(sessions={}) for k in ("token_stream", "handover_detector", "motion", "episode_guards")}
    for name in SESSION_NAMES:
        trials = load_session(name)
        args0 = trials[0]["args"]
        per_trial = [settings(t["args"]) for t in trials]
        result["sessions"][name] = dict(trials=len(trials), settings=per_trial[0],
                                        settings_same_all_trials=all(s == per_trial[0] for s in per_trial),
                                        anchoring=bool(args0["chunk_anchor_s"]), episode_ends=[t["episode_end"] for t in trials])

        ts = {f"trial_{t['index']:03d}": token_stream_trial(t, rng) for t in trials}
        ts["synthetic_spacing_sweep"] = token_stream_sweep(args0, rng)
        cases = [c for v in ts.values() for c in v.values()]
        checks["token_stream"]["sessions"][name] = dict(
            cases=len(cases), gated_cases=sum(c["pass"] is not None for c in cases),
            pass_=all(c["pass"] for c in cases if c["pass"] is not None),
            max_abs_token_diff_t0_0=max(v[k]["max_abs_token_diff"] for v in ts.values() for k in v if k.endswith("t0_0")),
            max_abs_token_diff_t0_uptime=max(v[k]["max_abs_token_diff"] for v in ts.values() for k in v if k.endswith("t0_uptime")),
            max_abs_hand_diff_t0_0=max(v[k]["max_abs_hand_diff"] for v in ts.values() for k in v if k.endswith("t0_0")),
            max_abs_hand_diff_t0_uptime=max(v[k]["max_abs_hand_diff"] for v in ts.values() for k in v if k.endswith("t0_uptime")),
            lerobot_slewed_ticks_perturbed=sum(v[k]["lerobot_slewed_ticks"] for v in ts.values() for k in v if "perturbed" in k),
            failing=[f"{t}/{k}" for t, v in ts.items() for k, c in v.items() if c["pass"] is False],
            hands_differ_t0_uptime=[f"{t}/{k}" for t, v in ts.items() for k, c in v.items()
                                    if k.endswith("t0_uptime") and not c["hands_equal"]],
            detail=ts)
        TokenStream.set_chunk = candidate_fix(ORIGINAL_SET_CHUNK)  # diagnosis only: not part of any pass
        fix_rng = np.random.default_rng(12)
        fixed = {f"trial_{t['index']:03d}": token_stream_trial(t, fix_rng) for t in trials}
        fixed["synthetic_spacing_sweep"] = token_stream_sweep(args0, fix_rng)
        TokenStream.set_chunk = ORIGINAL_SET_CHUNK
        result.setdefault("diagnosis", {}).setdefault("token_stream_with_candidate_fix", {})[name] = dict(
            failing=[f"{t}/{k}" for t, v in fixed.items() for k, c in v.items() if c["pass"] is False],
            hands_differ_t0_uptime=[f"{t}/{k}: {c['max_abs_hand_diff']:.3g}" for t, v in fixed.items()
                                    for k, c in v.items() if k.endswith("t0_uptime") and not c["hands_equal"]],
            max_abs_hand_diff_t0_0=max(v[k]["max_abs_hand_diff"] for v in fixed.values() for k in v if k.endswith("t0_0")))

        det = {f"trial_{t['index']:03d}": detector_trial(t) for t in trials}
        checks["handover_detector"]["sessions"][name] = dict(
            pass_=all(d["pass"] for d in det.values()), identical=sum(d["identical"] for d in det.values()),
            trials=len(det), fires_where_logged=sum(bool(d.get("fires_where_logged")) for d in det.values()),
            detail=det)

        planner = trials[0]["log"]["token"][0].astype(np.float32)  # trial 0's logged take-over token
        full = motion_full(args0, planner)
        logged = {f"trial_{t['index']:03d}": motion_logged(t, ready) for t in trials}
        n_seg = sum(len(v) for v in logged.values())
        checks["motion"]["sessions"][name] = dict(
            pass_=full["pass"] and all(c["identical"] for v in logged.values() for c in v),
            full_sequence=full, logged_segments=n_seg,
            logged_segments_identical=sum(c["identical"] for v in logged.values() for c in v),
            logged_segment_kinds=sorted({c["segment"] for v in logged.values() for c in v}), logged_detail=logged)

        forced, timing = guards_forced(args0, rng)
        replay = {f"trial_{t['index']:03d}": guards_logged(t) for t in trials}
        mapped = {e: next((k for k, r in forced.items() if r["lerobot"] == e), None)
                  for e in sorted({t["episode_end"] for t in trials})}
        lr_order = ["chunk", "state", "camera", "echo", "arm"]  # run_episode's checks in source order
        nat_order = native_priority(args0)
        checks["episode_guards"]["sessions"][name] = dict(
            pass_=(all(r["string_equal"] and r["category_as_spec"] for r in forced.values())
                   and all(r["pass_"] for r in replay.values()) and all(mapped.values())
                   and [k for k in nat_order if k != "cap"] == lr_order),
            forced=forced, timing=timing, logged_episode_ends_mapped=mapped,
            priority=dict(lerobot_source_order=lr_order, native=nat_order), logged_replay=replay)

        ep = np.flatnonzero(trials[1]["log"]["phase"] == "episode")
        t_ep = trials[1]["log"]["t_ep"][ep]
        result["observations"].setdefault("record_frame_index", {})[name] = dict(
            logged_equals_int_t_ep_x_policy_fps=bool(np.array_equal(
                trials[1]["log"]["frame"][ep], [min(int(t * args0["policy_fps"]), 10**9 - 1) for t in t_ep])),
            native_formula=f"int(t_ep * LIVE_FPS={loop_control.LIVE_FPS})",
            native_matches_logged=bool(np.array_equal([int(t * loop_control.LIVE_FPS) for t in t_ep],
                                                      trials[1]["log"]["frame"][ep])),
            policy_fps=args0["policy_fps"])

    for k, c in checks.items():
        c["pass"] = all(s["pass_"] for s in c["sessions"].values())
    result["checks"] = checks
    result["all_pass"] = all(c["pass"] for c in checks.values()) and asset_same
    out_path.write_text(json.dumps(result, indent=1, default=str) + "\n")
    summary = {k: dict(pass_=c["pass"], **{s: v["pass_"] for s, v in c["sessions"].items()}) for k, c in checks.items()}
    print(json.dumps(dict(all_pass=result["all_pass"], asset_same=asset_same, checks=summary), indent=1))


if __name__ == "__main__":
    main()
