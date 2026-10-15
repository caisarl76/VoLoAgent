"""Summarize the measured-desk (0.15 m edge) trials.

Usage: summarize.py [TRIAL_DIR...]  (default: every folder under trials/); writes trials.json.
Reuses the yaw-feedforward summary (return outcome, delivered tau) and adds desk and
bottle evidence from saved physics. The desk is in place for the whole standing
run, and from its restore for offer entries. Physics is sampled at 20 Hz, so a
touch between samples can be missed.
"""
import importlib.util
import json
from pathlib import Path
import sys

import mujoco
import numpy as np

ROOT = Path(__file__).parent
DESK = ["desk_top", *(f"desk_leg_{i}" for i in range(4))]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


base = load("ff_summarize", ROOT.parent / "g1_handover_20261015_yaw_feedforward/summarize.py")


def contact_evidence(run):
    result = json.loads((run / "result.json").read_text())
    restored = ((result.get("pre_pose") or {}).get("desk") or {})
    gt = base.torque.lines(run / "ground-truth.jsonl")
    sim = np.array([g["sim_time"] for g in gt])
    start = int(np.where(np.diff(sim) < 0)[0][-1] + 1) if np.any(np.diff(sim) < 0) else 0
    since = restored.get("sim_time", 0.0) if result.get("entry") else 0.0
    window = [g for g in gt[start:] if g["sim_time"] >= since]
    model = mujoco.MjModel.from_xml_path(str(run / "scene.xml"))
    data = mujoco.MjData(model)
    desk = {model.geom(n).id for n in DESK}
    bottle = {model.geom("task_bottle_geom").id}
    root = model.body("pelvis").id
    robot = {b for b in range(model.nbody) if b == root or model.body_rootid[b] == root}
    counts = dict(desk=0, bottle=0)
    first = dict(desk=None, bottle=None)
    bodies = dict(desk=set(), bottle=set())
    for g in window:
        data.qpos[:] = g["qpos"]
        mujoco.mj_forward(model, data)
        for c in data.contact[:data.ncon]:
            pair = {c.geom1, c.geom2}
            mine = [model.body(model.geom_bodyid[x]).name for x in pair if model.geom_bodyid[x] in robot]
            for key, geoms in (("desk", desk), ("bottle", bottle)):
                if pair & geoms and mine:
                    counts[key] += 1
                    bodies[key].update(mine)
                    first[key] = first[key] if first[key] is not None else g["sim_time"] - since
    return dict(contact_samples=len(window), robot_desk_contact_samples=counts["desk"],
                robot_bottle_contact_samples=counts["bottle"], first_desk_contact_s=first["desk"],
                desk_contact_bodies=sorted(bodies["desk"]), bottle_contact_bodies=sorted(bodies["bottle"]),
                feedforward="ready_gravity_feedforward: true" in (run / "simulation-profile.yaml").read_text())


def summarize(run):
    result = json.loads((run / "result.json").read_text())
    if result.get("entry") and not ((result.get("pre_pose") or {}).get("desk") or {}).get("restored"):
        return dict(entry=result["entry"], terminal_phase=None, pre_pose=result.get("pre_pose"), error=result.get("error"))
    report = base.summarize(run)
    report.update(contact_evidence(run))
    return report


if __name__ == "__main__":
    runs = [Path(p) for p in sys.argv[1:]] or sorted(p for p in (ROOT / "trials").iterdir() if p.is_dir())
    report = {run.name: summarize(run) for run in runs}
    (ROOT / "trials.json").write_text(json.dumps(report, indent=2, default=float) + "\n")
    for name, r in report.items():
        print(name, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()
                     if k in ("terminal_phase", "terminal_reason", "decision_direct", "decision_raise", "feedforward",
                              "stage1_entered_s", "return_s", "robot_desk_contact_samples", "first_desk_contact_s",
                              "desk_contact_bodies", "robot_bottle_contact_samples", "bottle_contact_bodies",
                              "raise_tail_worst_motor", "raise_tail_max_error_rad", "error")})
