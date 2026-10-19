"""Summarize post-handover-entry trials: pre-pose, return outcome, preserved arm, desk clearance.

Usage: summarize.py [TRIAL_DIR...]  (default: every folder under trials/); writes trials.json.
The return is the second reset in the paired trace (final target changes); desk
evidence uses saved physics after the desk was restored.
"""
import gzip
import json
from pathlib import Path
import sys

import mujoco
import numpy as np

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT.parent / "g1_handover_20261013_lift"))
from geometry import closest_bounds_gap  # noqa: E402

IDS = [12, 13, 14, 15, 22, 16, 23, 17, 24, 18, 25, 19, 26, 20, 27, 21, 28]  # planner order -> motor
DESK = ["desk_top", *(f"desk_leg_{i}" for i in range(4))]


def lines(path):
    if not path.exists():
        return [json.loads(s) for s in gzip.open(str(path) + ".gz", "rt")]
    return [json.loads(s) for s in path.read_text().splitlines()]


def joints31(r):
    return np.r_[np.asarray(r["body_q_measured_motor"])[IDS], r["left_hand_q_measured"], r["right_hand_q_measured"]]


def summarize(run):
    result = json.loads((run / "result.json").read_text())
    pre = result.get("pre_pose", {})
    rows = lines(run / "paired-reference-feedback.jsonl")
    first = rows[0]["final_target"]
    ret = [r for r in rows if r["final_target"] != first]
    report = dict(entry=result.get("entry"), return_mode=result.get("return_mode"),
                  native_run_vla_sha256=result["native_source_sha256"]["gear_sonic/scripts/run_vla_inference.py"],
                  pre_pose_phase=pre.get("phase"), pre_pose_s=pre.get("duration_s"),
                  desk_restored=(pre.get("desk") or {}).get("restored"),
                  desk_gap_at_restore_m=((pre.get("desk") or {}).get("desk_gap") or {}).get("gap_m"),
                  terminal_phase=(result.get("terminal_status") or {}).get("phase"),
                  terminal_reason=(result.get("terminal_status") or {}).get("reason"), error=result.get("error"),
                  owned_processes_stopped=result["owned_processes_stopped"])
    if not ret:
        return report
    t0 = ret[0]["at"]
    entry = joints31(ret[0])
    error = np.array([np.abs(joints31(r) - r["active_target"]) for r in ret])
    left = [i for i in range(17) if IDS[i] in range(12, 22)]  # waist + left arm, preserved
    report.update(
        return_s=ret[-1]["at"] - t0,
        stage1_entered_s=next((r["at"] - t0 for r in ret if r["stage"] == 1), None),
        max_raise_dwell_s=max(r["lift_dwell_s"] or 0 for r in ret),
        right_arm_move_rad=np.abs(np.subtract(ret[0]["final_target"][:17], entry[:17])).max().item(),
        preserved_waist_left_arm_max_error_rad=float(error[:, left].max()),
        preserved_worst_motor=IDS[left[int(error[:, left].max(0).argmax())]],
        final_tail_max_error_rad=float(error[-40:].max()),
    )
    gt = lines(run / "ground-truth.jsonl")
    sim = np.array([g["sim_time"] for g in gt])
    start = int(np.where(np.diff(sim) < 0)[0][-1] + 1) if np.any(np.diff(sim) < 0) else 0
    restored_at = (pre.get("desk") or {}).get("sim_time", np.inf)
    after = [g for g in gt[start:] if g["sim_time"] >= restored_at]
    model = mujoco.MjModel.from_xml_path(str(run / "scene.xml"))
    data = mujoco.MjData(model)
    desk = {model.geom(n).id for n in DESK}
    root = model.body("pelvis").id
    robot = {b for b in range(model.nbody) if b == root or model.body_rootid[b] == root}
    gaps, contacts = [], 0
    for g in after:
        data.qpos[:] = g["qpos"]
        mujoco.mj_forward(model, data)
        gaps.append(closest_bounds_gap(model, data, ["desk_top"])["gap_m"])
        contacts += sum(1 for c in data.contact[:data.ncon] if ({c.geom1, c.geom2} & desk) and (
            model.geom_bodyid[c.geom1] in robot or model.geom_bodyid[c.geom2] in robot))
    report.update(desk_samples=len(after), min_conservative_desk_gap_m=min(gaps) if gaps else None,
                  min_signed_desk_clearance_m=min((g["clearance"]["distance_m"] for g in after
                                                   if g["clearance"]["obstacle"].startswith("desk")), default=None),
                  robot_desk_contacts=contacts,
                  foot_floor_contact_samples=sum(bool(g["foot_floor_contacts"]) for g in after))
    return report


if __name__ == "__main__":
    runs = [Path(p) for p in sys.argv[1:]] or sorted((ROOT / "trials").iterdir())
    report = {run.name: summarize(run) for run in runs}
    (ROOT / "trials.json").write_text(json.dumps(report, indent=2, default=float) + "\n")
    counts = {}
    for r in report.values():
        key = f"{r['entry']} {r['return_mode']} native {r['native_run_vla_sha256'][:8]}"
        counts.setdefault(key, [0, 0])
        counts[key][0] += r["terminal_phase"] == "COMPLETED"
        counts[key][1] += 1
    print("completed/trials:", counts)
    for name, r in report.items():
        print(name, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()
                     if k not in ("entry", "return_mode", "owned_processes_stopped")})
