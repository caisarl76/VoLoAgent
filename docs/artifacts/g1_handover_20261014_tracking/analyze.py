"""Analyze paired native references, saved physics and decoder targets."""
import bisect
import hashlib
import json
import os
from pathlib import Path
import sys

import mujoco
import numpy as np

ROOT = Path(__file__).parent
NATIVE = ROOT.parents[3] / "g1-harness-native"
sys.path[:0] = [str(NATIVE), str(ROOT.parent / "g1_handover_20261013_lift")]
os.environ.setdefault("MPLCONFIGDIR", "/tmp/volo-tracking-mpl")
from analyze_native import analyze as analyze_physics
from gear_sonic.utils.teleop.xr_upperbody_bridge import G1_UPPER_BODY_JOINT_INDICES as IDS

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RUNS = ("paired-baseline", "paired-repeat", "corrected-first", "candidate-15-first", "decoder-trace")


def inspect(run):
    result = json.loads((run / "result.json").read_text())
    assert hashlib.sha256((run / "trace-worker-as-run.py").read_bytes()).hexdigest() == result["trace_worker_sha256"]
    rows = [json.loads(s) for s in (run / "paired-reference-feedback.jsonl").read_text().splitlines()]
    physics = [json.loads(s) for s in (run / "ground-truth.jsonl").read_text().splitlines()]
    times = [r["received_monotonic"] for r in physics]
    measured = np.array([np.asarray(r["body_q_measured_motor"])[IDS] for r in rows])
    reference = np.array([r["upper_body_reference"] for r in rows])
    target = np.array([r["active_target"][:17] for r in rows])
    matched = [physics[max(0, bisect.bisect_right(times, r["at"]) - 1)] for r in rows]
    physical = np.array([np.asarray(r["body_q"])[IDS] for r in matched])
    checked_joints = 17
    rate_excess = None
    if "left_hand_reference" in rows[0]:
        measured_all = np.array([np.r_[np.asarray(r["body_q_measured_motor"])[IDS],
                                       r["left_hand_q_measured"], r["right_hand_q_measured"]] for r in rows])
        reference_all = np.array([np.r_[r["upper_body_reference"], r["left_hand_reference"],
                                        r["right_hand_reference"]] for r in rows])
        checked_joints = 31
        dt = np.clip(np.array([r["loop_dt_s"] for r in rows[1:]]), 0, .05)
        rate_excess = float(np.max(np.abs(np.diff(reference_all, axis=0)) - .5 * dt[:, None]))
    else:
        measured_all, reference_all = measured, reference
    lead = np.abs(reference_all - measured_all)
    tail = np.abs(measured[-40:] - target[-40:])
    worst = int(np.unravel_index(tail.argmax(), tail.shape)[1])
    physics_report = analyze_physics(run)
    physics_report["last_pre_terminal_0p5_s_max_right_arm_error_rad"] = physics_report.pop(
        "last_pre_completion_0p5_s_max_right_arm_error_rad"
    )
    report = dict(
        terminal_phase=result["terminal_status"]["phase"], terminal_reason=result["terminal_status"]["reason"],
        samples=len(rows), stages=sorted(set(r["stage"] for r in rows)),
        maximum_paired_reference_lead_rad=float(lead.max()), checked_joint_count=checked_joints,
        maximum_rate_bound_excess_rad=rate_excess,
        final_tail_max_native_vs_saved_physics_gap_rad=float(np.abs(measured[-40:] - physical[-40:]).max()),
        final_tail_worst_motor=IDS[worst], final_tail_max_active_target_error_rad=float(tail.max()),
        max_recorded_lift_dwell_s=max(r["lift_dwell_s"] or 0 for r in rows),
        source_sha256=result["native_source_sha256"], owned_processes_stopped=result["owned_processes_stopped"],
        hardware_acceptance=False, physics=physics_report,
    )
    if "controller_last_action" in rows[0]:
        assert all(len(r["controller_last_action"]) == 29 for r in rows)
        action = np.array([r["controller_last_action"] for r in rows[-40:]])
        motor = IDS[worst]
        model = mujoco.MjModel.from_xml_path(str(run / "scene.xml"))
        data = mujoco.MjData(model)
        names = [model.joint(i).name for i in range(model.njnt)
                 if any(k in model.joint(i).name for k in ("hip", "knee", "ankle", "waist", "shoulder", "elbow", "wrist"))]
        assert len(names) == 29
        joint = model.joint(names[motor])
        floor_id = model.geom("floor").id
        robot_root = model.body("pelvis").id
        robot_bodies = {robot_root}
        for body in range(robot_root + 1, model.nbody):
            if int(model.body_parentid[body]) in robot_bodies:
                robot_bodies.add(body)
        gravity, non_floor_contacts, robot_non_floor_contacts = [], 0, 0
        for r in matched[-40:]:
            data.qpos[:] = r["qpos"]
            data.qvel[:] = 0  # Isolate the model's gravity load from velocity bias.
            mujoco.mj_forward(model, data)
            gravity.append(float(data.qfrc_bias[int(joint.dofadr[0])]))
            for contact in data.contact:
                geoms = (int(contact.geom1), int(contact.geom2))
                if floor_id not in geoms:
                    non_floor_contacts += 1
                    robot_non_floor_contacts += int(any(
                        int(model.geom_bodyid[geom]) in robot_bodies for geom in geoms
                    ))
        kp = .003609725 * (10 * 2 * np.pi) ** 2  # STIFFNESS_5020 in native policy_parameters.hpp.
        assert names[motor] == "right_shoulder_yaw_joint"
        delta = action[:, motor] - measured[-40:, worst]
        report["decoder"] = dict(
            joint=names[motor], mean_goal_rad=float(np.mean(target[-40:, worst])),
            mean_reference_rad=float(np.mean(reference[-40:, worst])),
            mean_motor_target_rad=float(np.mean(action[:, motor])),
            mean_measured_rad=float(np.mean(measured[-40:, worst])),
            mean_motor_target_vs_reference_gap_rad=float(np.mean(np.abs(action[:, motor] - reference[-40:, worst]))),
            mean_motor_target_vs_measured_gap_rad=float(np.mean(np.abs(delta))),
            model_gravity_torque_mean_nm=float(np.mean(gravity)),
            estimated_proportional_torque_mean_nm=float(np.mean(kp * delta)),
            shoulder_kp_nm_per_rad=float(kp), sampled_non_floor_contacts=non_floor_contacts,
            sampled_robot_non_floor_contacts=robot_non_floor_contacts,
            scope="Last 40 native samples; last_action is already motor-order absolute angle after native scale/default conversion. Torque is an estimate, excluding damping/friction; not measured motor torque.",
        )
    return report, rows


if __name__ == "__main__":
    report, raw = {}, {}
    for name in RUNS:
        report[name], raw[name] = inspect(ROOT / name)
    for name in ("corrected-first", "candidate-15-first", "decoder-trace"):
        assert report[name]["maximum_paired_reference_lead_rad"] <= .15000001
    assert report["paired-repeat"]["maximum_paired_reference_lead_rad"] > .15
    for name in ("candidate-15-first", "decoder-trace"):
        assert report[name]["maximum_rate_bound_excess_rad"] <= 1e-8
    (ROOT / "analysis.json").write_text(json.dumps(report, indent=2) + "\n")
    r = raw["decoder-trace"]
    start = r[0]["at"]
    x = [s["at"] - start for s in r]
    j = IDS.index(24)
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    for label, values in (
        ("Raised target", [s["active_target"][j] for s in r]),
        ("Bounded reference", [s["upper_body_reference"][j] for s in r]),
        ("Decoder motor target", [s["controller_last_action"][24] for s in r]),
        ("Measured joint", [s["body_q_measured_motor"][24] for s in r]),
    ):
        axes[0].plot(x, values, label=label)
    axes[0].set_ylabel("Right shoulder yaw (rad)")
    axes[0].legend()
    for name in ("paired-repeat", "decoder-trace"):
        s = raw[name]
        lead = [max(abs(np.asarray(t["upper_body_reference"]) - np.asarray(t["body_q_measured_motor"])[IDS])) for t in s]
        axes[1].plot([t["at"] - s[0]["at"] for t in s], lead, label=name)
    axes[1].axhline(.15, color="black", linestyle="--", label="0.15 rad bound")
    axes[1].set_ylabel("Maximum upper-body reference lead (rad)")
    axes[1].set_xlabel("Time since first traced return command (s)")
    axes[1].legend()
    for axis in axes:
        axis.grid(alpha=.25)
    fig.suptitle("Reference guard corrected; simulated arm still does not settle")
    fig.tight_layout()
    fig.savefig(ROOT / "tracking-comparison.png", dpi=150)
    plt.close(fig)
    print(json.dumps({name: {k: v for k, v in r.items() if k not in ("physics", "source_sha256")}
                      for name, r in report.items()}, indent=2))
