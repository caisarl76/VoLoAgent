"""Summarize loop-probe run folders into trials.json.

Usage (simulator Python, from the acceptance folder for scene.xml's mesh paths):
    summarize_trials.py OUTPUT.json RUN_DIR [RUN_DIR ...]
Per run: autopilot outcome, fixture events, handover cycles completed, each
production ready return's decision and outcome, and box contacts. Only the
probe's fixture pre-pose (tagged fixture_start_pre_pose in native-trace.jsonl;
no lead guard, looser settling) is excluded; failed and interrupted production
returns are kept. Box contacts come from replaying
the logged qpos in the run's scene: `clearance` reads 0.0 falsely for G1 meshes.
"""
import collections
import json
from pathlib import Path
import sys

import mujoco

DESK = ["desk_top", *(f"desk_leg_{i}" for i in range(4))]


def loop_samples(run):
    rows = [json.loads(line) for line in (run / "ground-truth.jsonl").open()]
    restarts = [i for i in range(1, len(rows)) if rows[i]["sim_time"] < rows[i - 1]["sim_time"] - 1]
    return rows[restarts[-1]:] if restarts else rows  # After grounded startup.


def box_contacts(run, samples):
    model = mujoco.MjModel.from_xml_path(str(run / "scene.xml"))
    data = mujoco.MjData(model)
    desk = {model.geom(name).id for name in DESK}
    bottle = model.jnt_bodyid[model.joint("task_bottle_free").id]
    counts = collections.Counter()
    for sample in samples:
        if sample["box_away"]:
            continue
        data.qpos[:] = sample["qpos"]
        mujoco.mj_forward(model, data)
        bodies = set()
        for contact in data.contact[:data.ncon]:
            pair = (contact.geom1, contact.geom2)
            inside = [g in desk for g in pair]
            if any(inside) and not all(inside):
                body = int(model.geom_bodyid[pair[inside.index(False)]])
                if body != bottle:  # The standing bottle has no contacts at runtime.
                    bodies.add(model.body(body).name)
        counts.update(bodies)
    return dict(counts)


def summarize(run):
    result = json.loads((run / "result.json").read_text())
    events = json.loads((run / "simulator-result.json").read_text())["fixture_events"]
    decisions = [line.split("ready return: ", 1)[1].strip() for line in (run / "native.log").open()
                 if "[harness] ready return:" in line]
    tags = [e["event"] for e in map(json.loads, (run / "native-trace.jsonl").open())
            if e["event"] in ("fixture_start_pre_pose", "ready_return_begin")]
    if len(tags) != len(decisions):
        raise ValueError(f"{run}: {len(decisions)} return decisions but {len(tags)} tagged returns")
    decisions = [d for d, tag in zip(decisions, tags) if tag == "ready_return_begin"]
    mission = [json.loads(line) for path in (run / "missions").glob("*/events.jsonl") for line in path.open()]
    returns = []
    for i, e in enumerate(mission):  # The autopilot's own reset_ready requests: production returns.
        if e["event"] == "request" and e["method"] == "reset_ready":
            end = next((s for s in mission[i + 1:] if s["event"] == "status"
                        and s["phase"] in ("COMPLETED", "INTERRUPTED", "FAULT")), None)
            returns.append("unfinished" if end is None else "settled" if end["phase"] == "COMPLETED"
                           else f'{end["phase"].lower()}: {end["reason"]}')
    if len(returns) != len(decisions):
        raise ValueError(f"{run}: {len(returns)} reset_ready requests but {len(decisions)} production returns")
    phases = [(e["handover_phase"], e["cycle_id"]) for e in mission if e["event"] == "handover_phase"]
    return dict(
        run=run.name, box_front_m=result["box_front_m"], box_height_m=result["box_height_m"],
        cycles_requested=result["cycles"], autopilot_outcome=result["autopilot_outcome"],
        autopilot_reason=result["autopilot_reason"],
        cycles_completed=sum(e["event"] == "handover_cycle_complete" for e in mission),
        phases=phases,
        production_ready_returns=[dict(decision=d, outcome=o) for d, o in zip(decisions, returns)],
        production_ready_returns_settled=f"{returns.count('settled')}/{len(returns)}",
        fixture_events=[{k: (round(v, 3) if isinstance(v, float) else v) for k, v in e.items()} for e in events],
        box_contact_samples_by_body=box_contacts(run, loop_samples(run)),
        native_base_commit=result["native_base_commit"], volo_commit=result["volo_commit"],
        binary_sha256=result.get("binary_sha256"),
    )


if __name__ == "__main__":
    trials = [summarize(Path(path)) for path in sys.argv[2:]]
    Path(sys.argv[1]).write_text(json.dumps(trials, indent=2) + "\n")
    for trial in trials:
        print(trial["run"], trial["autopilot_outcome"], trial["cycles_completed"],
              trial["production_ready_returns_settled"], trial["box_contact_samples_by_body"])
