"""Operator verdicts for the autonomous handover loop, as lerobot eval_session files.

Stdin and the session directory only: never talks to the robot or the RPC.
results.jsonl holds success/fail lines (lerobot's converter keeps every line);
redo verdicts go to redo.jsonl in the same line format.
"""

import json
import queue
import re
import threading
from pathlib import Path

import numpy as np

# verbatim lerobot eval_session.DEFAULT_FAILURE_REASONS (a6e267c6)
FAILURE_REASONS = (
    "did not reach the object",
    "grasp failed / missed",
    "dropped the object",
    "wrong object / target",
    "collision (table, knocked over)",
    "stalled / froze",
    "safety stop (watchdog)",
    "other",
)
SAFETY_STOP = "safety stop (watchdog)"
PROMPT = "s / f / r [+ reason index] [note], e = end session, c = cancel verdicts typed ahead"


def _trials(session_dir, pattern):
    return sorted(int(re.search(r"\d+", p.name).group()) for p in Path(session_dir).glob(pattern))


def _scored(session_dir, name):
    path = Path(session_dir) / name
    if not path.exists():
        return []
    return [json.loads(s)["trial"] for s in path.read_text().splitlines() if s.strip()]


def _recorded(session_dir):
    return _trials(session_dir, "trial_*_frames.npz")


def pending_trials(session_dir) -> list[int]:
    done = set(_scored(session_dir, "results.jsonl")) | set(_scored(session_dir, "redo.jsonl"))
    return [t for t in _recorded(session_dir) if t not in done]


def next_trial(session_dir) -> int:
    scored = _scored(session_dir, "results.jsonl") + _scored(session_dir, "redo.jsonl")
    return max(max(_recorded(session_dir), default=-1) + 1, max(scored, default=-1) + 1)


def remaining_trials(session_dir, n_trials) -> int:
    return n_trials - (len(_recorded(session_dir)) - len(_scored(session_dir, "redo.jsonl")))


def append_verdict(session_dir, trial, verdict, reason, note) -> dict:
    session_dir = Path(session_dir)
    base = session_dir / f"trial_{trial:03d}"
    with np.load(f"{base}.npz") as z:
        episode_end = str(z["episode_end"])
        wall_start, wall_stop = float(z["wall_start"]), float(z["wall_stop"])
        model = (str(z["model"]) if "model" in z.files else "") or None
    with np.load(f"{base}_frames.npz") as f:
        frames = len(f["offsets"]) - 1
    if trial in _scored(session_dir, "results.jsonl") + _scored(session_dir, "redo.jsonl"):
        raise ValueError(f"trial {trial} already has a verdict")
    if verdict != "fail":
        reason = note = ""
    elif verdict == "fail" and not reason:
        if not episode_end.startswith("watchdog"):
            raise ValueError("fail needs a reason")
        reason = SAFETY_STOP
    line = {
        "trial": trial,
        "verdict": verdict,
        "reason": reason or "",
        "note": note or "",
        "episode_end": episode_end,
        "wall_start": wall_start,
        "wall_stop": wall_stop,
        "frames": frames,
        "model": model,
    }
    name = "redo.jsonl" if verdict == "redo" else "results.jsonl"
    with open(session_dir / name, "a") as f:
        f.write(json.dumps(line) + "\n")
    return line


def request_session_end(session_dir) -> None:
    (Path(session_dir) / "end_session.request").touch()


def _parse(text):
    """'f 3 note text' -> (verdict, reason or None, note)."""
    tokens = text.split()
    verdict = {"s": "success", "f": "fail", "r": "redo"}[tokens[0].lower()]
    rest = tokens[1:]
    reason = None
    if rest and rest[0].isdigit():
        n = int(rest.pop(0))
        if not 1 <= n <= len(FAILURE_REASONS):
            raise ValueError(f"reason index must be 1..{len(FAILURE_REASONS)}")
        reason = FAILURE_REASONS[n - 1]
    return verdict, reason, " ".join(rest)


def run_console(session_dir, lines, poll_s=0.5) -> None:
    """Score pending trials from `lines` (stdin); waits for new trials until `e` or EOF.

    A verdict applies only to the trial whose episode_end was printed last. Verdict lines typed while no trial is
    shown are held, in order, and each is applied once the next pending trial has been shown.
    """
    q = queue.Queue()

    def read():
        for s in lines:
            q.put(s)
        q.put(None)

    threading.Thread(target=read, daemon=True).start()
    shown = bad = None
    held = []
    while True:
        try:
            pending = pending_trials(session_dir)
        except ValueError as exc:  # corrupt jsonl: never skip it (would allow double scoring)
            pending = []
            if str(exc) != bad:
                bad = str(exc)
                print(f"? cannot read results files: {exc}", flush=True)
        if not pending:
            shown = None
        elif pending[0] != shown:
            shown = pending[0]
            with np.load(Path(session_dir) / f"trial_{shown:03d}.npz") as z:
                print(f"trial {shown}: {z['episode_end']}  [{PROMPT}]", flush=True)
        if held and shown is not None:
            text = held.pop(0)
        else:
            try:
                text = q.get(timeout=poll_s)
            except queue.Empty:
                continue
            if text is None:
                if held:
                    print(f"{len(held)} held verdict(s) not applied: no trial shown", flush=True)
                return
            words = text.split()
            if not words:
                continue
            if words[0].lower() == "c":  # a stray verdict typed while no trial was shown must not score the next one
                print(f"dropped {len(held)} held verdict(s)", flush=True)
                held.clear()
                continue
            if words[0].lower() == "e":
                if held:
                    print(f"{len(held)} held verdict(s) not applied: score them in a later verdict run", flush=True)
                request_session_end(session_dir)
                return
            if words[0].lower() not in ("s", "f", "r"):
                print(f"? {PROMPT}", flush=True)
                continue
            if shown is None:  # typed ahead: waits for the next trial to be shown
                held.append(text)
                print("no pending trial; held for the next one", flush=True)
                continue
        try:
            verdict, reason, note = _parse(text)
            line = append_verdict(session_dir, shown, verdict, reason, note)
            print(f"trial {shown}: {line['verdict']} {line['reason']}".rstrip(), flush=True)
        except ValueError as exc:
            print(f"? {exc}", flush=True)
