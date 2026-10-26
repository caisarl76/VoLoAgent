"""Measure a run of run_autonomous_loop.py: writes RUN/analysis.json and prints it.

Usage: analyze_run.py RUN_DIR   (VoLoAgent venv; lerobot's summarize runs in the lerobot venv)

Sources: the probe's events.jsonl, the native-side message log (native-messages.jsonl), the timestamped deploy
lines (sonic-lines.jsonl), the simulator ground truth (ground-truth.jsonl, 20 Hz) and the session directory.
All times are CLOCK_MONOTONIC seconds, shared by every process on the workstation.
"""
import base64
import json
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np

ASSET = Path("/home/jihun/work/VoLoAgent/.worktrees/g1-loop-native/gear_sonic/data/handover/ready_pose_261020_ep76.npz")
LEROBOT_PY = Path.home() / "work/lerobot/.venv/bin/python"
GAP_LIMIT_S = 0.025
NOT_ROBOT = {"task_bottle", "world", ""}


def jsonl(path):
    return [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()] if Path(path).exists() else []


def pose_payload(b64):
    raw = base64.b64decode(b64)
    token = np.frombuffer(raw[:256], "<f4")
    frame = int(np.frombuffer(raw[256:264], "<i8")[0])
    hands = np.frombuffer(raw[264:320], "<f4")
    return token, frame, hands


def stream_checks(ticks, sends, t0, t1):
    """One pose message per tick, no gaps over 25 ms in POSE, frame indices consecutive, within [t0, t1]."""
    ticks = [r for r in ticks if t0 <= r["t"] <= t1]
    sends = [r for r in sends if t0 <= r["t"] <= t1 + 0.05]
    out = dict(ticks=len(ticks))
    if not ticks:
        return out
    per_tick, i = [], 0
    pose = [r for r in sends if r["topic"] == "pose"]
    times = [r["t"] for r in pose]
    for a, b in zip(ticks, ticks[1:] + [dict(t=ticks[-1]["t"] + 0.019)]):
        j = i
        while j < len(times) and times[j] < b["t"]:
            j += 1
        per_tick.append(sum(1 for t in times[i:j] if t >= a["t"]))
        i = j
    counts = np.bincount(per_tick)
    out["pose_messages"] = len([t for t in times if ticks[0]["t"] <= t <= ticks[-1]["t"] + 0.019])
    out["pose_messages_per_tick_histogram"] = {str(k): int(v) for k, v in enumerate(counts) if v}
    frames = [pose_payload(r["b64"])[1] for r in pose if ticks[0]["t"] <= r["t"]]
    out["frame_index_steps"] = {str(k): int(v) for k, v in zip(*np.unique(np.diff(frames), return_counts=True))}
    other = [dict(t=r["t"], topic=r["topic"], start_stop_planner=r.get("start_stop_planner"),
                  label=next((x["label"] for x in reversed(ticks) if x["t"] <= r["t"]), None))
             for r in sends if r["topic"] != "pose" and ticks[0]["t"] <= r["t"] <= ticks[-1]["t"]]
    out["non_pose_messages_inside_loop"] = other
    pose_on = next((r["t"] for r in sends if r.get("start_stop_planner") == [1, 0, 0]), None)
    pose_off = next((r["t"] for r in sends if r.get("start_stop_planner") == [1, 0, 1] and pose_on
                     and r["t"] > pose_on), None)
    out["pose_mode_window"] = [pose_on, pose_off]
    if pose_on:
        window = [t for t in times if pose_on <= t <= (pose_off or t1)]
        gaps = np.diff(window)
        out["pose_mode_send_gaps_s"] = dict(n=len(gaps), max=float(gaps.max()), p99=float(np.percentile(gaps, 99)),
                                            median=float(np.median(gaps)),
                                            over_25ms=int((gaps > GAP_LIMIT_S).sum()))
        late = np.nonzero(gaps > GAP_LIMIT_S)[0]
        out["pose_mode_gaps_over_25ms"] = [
            dict(at=window[i + 1], gap_s=float(gaps[i]),
                 label=next((x["label"] for x in reversed(ticks) if x["t"] <= window[i + 1]), None),
                 stage=next((x["stage"] for x in reversed(ticks) if x["t"] <= window[i + 1]), None)) for i in late]
        tick_now = np.array([r["now"] for r in ticks if pose_on <= r["t"] <= (pose_off or t1)])
        tg = np.diff(tick_now)
        out["pose_mode_tick_gaps_s"] = dict(max=float(tg.max()), over_25ms=int((tg > GAP_LIMIT_S).sum()))
    return out


def main():
    run = Path(sys.argv[1]).resolve()
    events = jsonl(run / "events.jsonl")
    ev = {}
    for e in events:
        ev.setdefault(e["event"], []).append(e)
    native = jsonl(run / "native-messages.jsonl")
    ticks = [r for r in native if r["k"] == "tick"]
    sends = [r for r in native if r["k"] == "send"]
    session = run / "session"
    t_start, t_end = ev["session_start"][0]["at"], ev["session_end"][0]["at"]
    dead_at = ev["dead_stream_take_over"][0]["at"] if "dead_stream_take_over" in ev else None
    result = json.loads((run / "result.json").read_text())
    a = {}

    # --- one pose message per tick, gaps --------------------------------------------------------------------
    a["session_stream"] = stream_checks(ticks, sends, t_start, t_end)
    if dead_at and result.get("dead_stream"):
        a["dead_stream_take_over_stream"] = stream_checks(ticks, sends, dead_at, result["dead_stream"]["killed_at"])

    # --- trials, verdicts, lerobot summarize ------------------------------------------------------------------
    trials = []
    for path in sorted(session.glob("trial_[0-9][0-9][0-9].npz")):
        with np.load(path) as z:
            n = int(path.stem[6:9])
            with np.load(session / f"trial_{n:03d}_frames.npz") as f:
                frames = len(f["offsets"]) - 1
            trials.append(dict(trial=n, episode_end=str(z["episode_end"]), ticks=int(len(z["t"])) if "t" in z.files
                               else None, frames=frames, wall_s=float(z["wall_stop"] - z["wall_start"])))
    a["trials"] = trials
    a["results_jsonl"] = jsonl(session / "results.jsonl")
    a["redo_jsonl"] = jsonl(session / "redo.jsonl")
    with tempfile.TemporaryDirectory() as tmp:
        source = subprocess.check_output(["git", "-C", str(Path.home() / "work/lerobot"), "show",
                                          "a6e267c6:examples/g1_dex3_training/eval_session.py"])
        (Path(tmp) / "eval_session.py").write_bytes(source)
        summary = subprocess.run([str(LEROBOT_PY), "-I", "-c", "import sys; sys.path.insert(0, sys.argv[1]); "
                                  "import eval_session; from pathlib import Path; "
                                  "print(eval_session.summarize(Path(sys.argv[2])))", tmp, str(session)],
                                 capture_output=True, text=True)
    a["lerobot_summarize"] = dict(returncode=summary.returncode, stdout=summary.stdout, stderr=summary.stderr[-2000:])

    # --- runner evidence: trial starts/ends, re-arm decisions -------------------------------------------------
    evidence = {}
    for path in sorted(session.glob("trial_*_evidence.jsonl")):
        evidence[int(path.name[6:9])] = jsonl(path)
    a["evidence_kinds"] = {str(n): [e["kind"] for e in es if e["kind"] != "rearm_decision"] for n, es in evidence.items()}
    a["trial_ends"] = [dict(trial=n, reason=e["reason"]) for n, es in evidence.items() for e in es if e["kind"] == "trial_end"]
    rearm = {}
    for n, es in evidence.items():
        rearm[str(n)] = [dict(at=e["at"], step=e["step"], outcome=e["outcome"], reason=e["reason"],
                              complete=e["complete"]) for e in es if e["kind"] == "rearm_decision"]
    a["rearm_decision_counts"] = {n: {f"{d['step']}:{d['outcome']}": sum(1 for x in ds if (x["step"], x["outcome"]) ==
                                      (d["step"], d["outcome"])) for d in ds} for n, ds in rearm.items()}

    # --- leftover bottle ---------------------------------------------------------------------------------------
    left = (result.get("person") or {}).get("leftover")
    if left:
        nxt = left["trial"] + 1
        es = evidence.get(nxt, [])
        start = next((e["at"] for e in es if e["kind"] == "trial_start"), None)
        empty_end = next((e["at"] for e in es if e["kind"] == "rearm_end" and e["step"] == "empty"), None)
        placed = next((p["at"] for p in result["person"]["placements"] if p["trial"] == nxt), None)
        during = [d for d in rearm.get(str(nxt), []) if d["at"] < left.get("removed_at", 0)]
        a["leftover"] = dict(trial=left["trial"], seen_at=left["seen_at"], removed_at=left.get("removed_at"),
                             standing_s=(left.get("removed_at") or 0) - left["seen_at"],
                             decisions_while_standing={o: sum(1 for d in during if d["outcome"] == o)
                                                       for o in {d["outcome"] for d in during}},
                             yes_while_standing=sum(1 for d in during if d["outcome"] == "yes"),
                             empty_confirmed_after_removal_s=None if empty_end is None else empty_end - left["removed_at"],
                             bottle_placed_after_removal_s=None if placed is None else placed - left["removed_at"],
                             next_trial_started_after_placement_s=None if start is None or placed is None else start - placed,
                             no_trial_before_clear_and_new_bottle=bool(start and placed and start > placed > left["removed_at"]))

    # --- stop mid-raise: freeze latency and resume continuity --------------------------------------------------
    press = next((e for e in ev.get("console_press", []) if e["why"] == "stop mid-raise"), None)
    if press:
        frozen = [r for r in ticks if r["t"] > press["at"] and r["label"] == "frozen"]
        first_frozen = frozen[0] if frozen else None
        stop_ev = next((e for es in evidence.values() for e in es if e["kind"] == "stop_key"), {})
        freeze_ev = next((e for es in evidence.values() for e in es if e["kind"] == "loop_freeze"), {})
        before = [r for r in ticks if r["t"] < (first_frozen or press)["t" if first_frozen else "at"]]
        a["stop"] = dict(pressed_at=press["at"], runner_received_at=stop_ev.get("pressed_at"),
                         freeze_dispatched_at=freeze_ev.get("dispatched_at"), freeze_acked_at=freeze_ev.get("acked_at"),
                         frozen_stage=freeze_ev.get("frozen_stage"), label_at_press=before[-1]["label"] if before else None,
                         first_frozen_tick_at=first_frozen and first_frozen["t"],
                         press_to_frozen_tick_s=first_frozen and first_frozen["t"] - press["at"],
                         press_to_runner_s=stop_ev.get("pressed_at") and stop_ev["pressed_at"] - press["at"],
                         press_to_ack_s=freeze_ev.get("acked_at") and freeze_ev["acked_at"] - press["at"])
        # raise ticks and their tokens: the pose message sent after each tick carries that tick's output
        pose = [r for r in sends if r["topic"] == "pose"]
        times = np.array([r["t"] for r in pose])
        # the session's raise path: consecutive ticks from the first "to raise start" through its "to ready" blend
        start = next(i for i, r in enumerate(ticks) if t_start <= r["t"] and r["label"] == "to raise start")
        seq, seen_ready = [], False
        for r in ticks[start:]:
            if r["label"] == "to ready":
                seen_ready = True
            elif seen_ready:
                break
            seq.append(r)
        labels = [r["label"] for r in seq]
        tokens = []
        for r in seq:
            j = int(np.searchsorted(times, r["t"]))
            tokens.append(pose_payload(pose[j]["b64"])[0] if j < len(pose) else None)
        tokens = np.array(tokens)
        asset = np.load(ASSET)
        idx = [i for i, label in enumerate(labels) if label == "raise"]
        frz = [i for i, label in enumerate(labels) if label == "frozen"]
        steps = np.abs(np.diff(tokens[idx], axis=0)).max(axis=1)
        resume_i = next((i for i in idx if frz and i > frz[-1]), None)
        a["resume"] = dict(
            label_runs=[[lab, n] for lab, n in _runs(labels)],
            raise_ticks=len(idx), raise_rows_expected=int(len(asset["raise_tokens"])),
            frozen_ticks=len(frz),
            frozen_token_spread=float(np.abs(tokens[frz] - tokens[frz[0]]).max()) if frz else None,
            frozen_equals_last_raise_token=bool(frz and np.array_equal(tokens[frz[0]], tokens[frz[0] - 1])),
            max_raise_token_step=float(steps.max()),
            resume_token_step=None if resume_i is None else float(np.abs(tokens[resume_i] - tokens[resume_i - 1]).max()),
            raise_tokens_vs_asset_max_abs=float(np.abs(tokens[idx] - asset["raise_tokens"][:len(idx)]).max())
            if len(idx) <= len(asset["raise_tokens"]) else None)

    # --- deploy: Planner timeout around take-overs; dead pose stream ------------------------------------------
    lines = jsonl(run / "sonic-lines.jsonl")
    pose_on_times = [r["t"] for r in sends if r.get("start_stop_planner") == [1, 0, 0]]
    takeover_ticks = [r["t"] for i, r in enumerate(ticks) if r["label"] == "planner token"
                      and (i == 0 or ticks[i - 1]["label"] != "planner token")]
    a["planner_timeout_lines"] = [dict(at=s["at"], line=s["line"],
                                       after_take_over_s=min((s["at"] - t for t in takeover_ticks if t <= s["at"]),
                                                             default=None),
                                       after_pose_switch_s=min((s["at"] - t for t in pose_on_times if t <= s["at"]),
                                                               default=None))
                                  for s in lines if "Planner timeout" in s["line"]]
    a["take_over_starts"] = takeover_ticks
    planner_sends = [r["t"] for r in sends if r["topic"] == "planner"]
    a["take_over_planner_silence"] = [
        dict(take_over_at=t, last_planner_message_before_s=None if not before else t - before[-1],
             planner_silence_at_pose_switch_s=None if not before or not after else after[0] - before[-1])
        for t in takeover_ticks
        for before in [[p for p in planner_sends if p <= t]]
        for after in [[p for p in pose_on_times if p >= t]]]
    a["pose_switches"] = pose_on_times
    truth = jsonl(run / "ground-truth.jsonl")
    if result.get("dead_stream"):
        k = result["dead_stream"]["killed_at"]
        last_pose = max((r["t"] for r in sends if r["topic"] == "pose" and r["t"] <= k + 0.05), default=None)
        window = [r for r in truth if k - 0.5 <= r["received_monotonic"] <= k + 3.5]
        ref = min(window, key=lambda r: abs(r["received_monotonic"] - k))
        body0 = np.array(ref["body_q"])
        a["dead_stream"] = dict(
            killed_at=k, last_logged_pose_send_before_kill_s=None if last_pose is None else last_pose - k,
            deploy_lines=[dict(after_kill_s=s["at"] - k, line=s["line"]) for s in lines if k <= s["at"] <= k + 3.5],
            robot=[dict(after_kill_s=r["received_monotonic"] - k, pelvis_z=r["qpos"][2],
                        max_joint_change_rad=float(np.abs(np.array(r["body_q"]) - body0).max()),
                        feet_on_floor=len(r["foot_floor_contacts"]), box_contacts=r["box_contacts"])
                   for r in window[::5]],
            pelvis_z_min=min(r["qpos"][2] for r in window if r["received_monotonic"] >= k),
            max_joint_change_rad_3s=max(float(np.abs(np.array(r["body_q"]) - body0).max()) for r in window
                                        if r["received_monotonic"] >= k),
            deploy_alive=result["dead_stream"]["deploy_alive"])

    # --- box contacts and falls ------------------------------------------------------------------------------
    def stage_at(t):
        r = next((x for x in reversed(ticks) if x["t"] <= t), None)
        return None if r is None or t - r["t"] > 0.1 else f"{r['stage']}/{r['label']}"

    contacts = [dict(at=r["received_monotonic"], bodies=sorted(set(r["box_contacts"]) - NOT_ROBOT),
                     stage=stage_at(r["received_monotonic"]))
                for r in truth if set(r["box_contacts"]) - NOT_ROBOT]
    in_session = [c for c in contacts if t_start <= c["at"] <= t_end]
    a["box_contacts"] = dict(samples_total=len(truth), contact_samples_session=len(in_session),
                             contact_samples_all=len(contacts),
                             bodies_session=sorted({b for c in in_session for b in c["bodies"]}),
                             stages_session=sorted({c["stage"] or "-" for c in in_session}),
                             first=in_session[:10])
    sess = [r for r in truth if t_start <= r["received_monotonic"] <= t_end]
    a["pelvis_z_session"] = dict(min=min(r["qpos"][2] for r in sess), max=max(r["qpos"][2] for r in sess))
    sim = json.loads((run / "simulator-result.json").read_text()) if (run / "simulator-result.json").exists() else {}
    a["simulator"] = dict(automatic_resets=sim.get("automatic_resets"), startup_resets=sim.get("startup_resets"),
                          fixture_events=sim.get("fixture_events"))
    (run / "analysis.json").write_text(json.dumps(a, indent=2, default=str) + "\n")
    print(json.dumps(a, indent=2, default=str))


def _runs(labels):
    out = []
    for label in labels:
        if out and out[-1][0] == label:
            out[-1][1] += 1
        else:
            out.append([label, 1])
    return out


if __name__ == "__main__":
    main()
