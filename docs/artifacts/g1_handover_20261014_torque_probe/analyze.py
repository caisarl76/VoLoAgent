"""Compare baseline and arm-gravity-feedforward ready-return trials.

Usage: analyze.py [TRIAL_DIR...]  (default: every folder under trials/) (writes analysis.json and torque-comparison.png here)
"""
import gzip
import json
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).parent
os.environ.setdefault("MPLCONFIGDIR", "/tmp/volo-tracking-mpl")
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

IDS = [12, 13, 14, 15, 22, 16, 23, 17, 24, 18, 25, 19, 26, 20, 27, 21, 28]  # planner order
WATCH = {24: "right_shoulder_yaw", 18: "left_elbow", 23: "right_shoulder_roll"}


def lines(path):
    """Read a JSONL stream, or its saved .gz copy."""
    if not path.exists():
        return [json.loads(s) for s in gzip.open(str(path) + ".gz", "rt")]
    return [json.loads(s) for s in path.read_text().splitlines()]


def joints31(r):
    return np.r_[np.asarray(r["body_q_measured_motor"])[IDS], r["left_hand_q_measured"], r["right_hand_q_measured"]]


def inspect(run):
    result = json.loads((run / "result.json").read_text())
    rows, pd, gt = lines(run / "paired-reference-feedback.jsonl"), lines(run / "pd-trace.jsonl"), lines(run / "ground-truth.jsonl")
    t0 = rows[0]["at"]
    stage1 = next((r["at"] - t0 for r in rows if r["stage"] == 1), None)
    raised = [r for r in rows if r["stage"] == 0]
    tail = rows[-40:]
    err = np.array([np.abs(joints31(r) - np.asarray(r["active_target"])) for r in tail]).max(0)
    raise_err = np.array([np.abs(joints31(r)[:17] - np.asarray(r["active_target"][:17])) for r in raised[-40:]]).max(0)
    lead = max(np.abs(np.r_[r["upper_body_reference"], r["left_hand_reference"], r["right_hand_reference"]] - joints31(r)).max()
               for r in rows)
    # Map the last 2 s of the reset (monotonic) onto sim time through the physics samples.
    mono = np.array([g["received_monotonic"] for g in gt])
    sim = np.array([g["sim_time"] for g in gt])
    # Grounded startup resets sim time; only use samples after the last reset.
    start = int(np.where(np.diff(sim) < 0)[0][-1] + 1) if np.any(np.diff(sim) < 0) else 0
    lo, hi = np.interp([rows[-1]["at"] - 2.0, rows[-1]["at"]], mono[start:], sim[start:])
    after = [p for p in pd if p["kp"][24] > 0]
    window = [p for p in after if lo <= p["sim_time"] <= hi]
    balance = {}
    for motor, name in WATCH.items():
        a = lambda k: np.array([p[k][motor] for p in window])  # noqa: E731
        balance[name] = dict(
            motor=motor, command_rad=float(a("cmd_q").mean()), measured_rad=float(a("q").mean()),
            command_minus_measured_rad=float((a("cmd_q") - a("q")).mean()),
            pd_torque_nm=float(a("pd_torque").mean()), diagnostic_feedforward_nm=float(a("diagnostic_feedforward").mean()),
            applied_actuator_force_nm=float(a("actuator_force").mean()),
            static_gravity_hold_nm=float(a("static_gravity").mean()),
            friction_constraint_nm=float(a("qfrc_constraint").mean()), velocity_rad_s=float(a("dq").mean()),
            kp=float(a("kp").mean()), kd=float(a("kd").mean()),
        )
    clipped = [p for p in window if np.any(np.abs(np.subtract(p["pd_torque"], p["clipped_torque"])
                                                  + np.asarray(p["diagnostic_feedforward"])) > 1e-9)]
    preserved = [i for i in range(17) if i not in (4, 6, 8, 10, 12, 14, 16)]  # waist + left arm
    return dict(
        terminal_phase=result["terminal_status"]["phase"], terminal_reason=result["terminal_status"]["reason"],
        gravity_feedforward_motors=result["diagnostic_gravity_feedforward_motors"],
        physics_modified=result["physics_modified"], owned_processes_stopped=result["owned_processes_stopped"],
        stage1_entered_s=stage1, max_raise_dwell_s=max(r["lift_dwell_s"] or 0 for r in rows),
        max_reference_lead_rad=float(lead),
        raise_tail_max_error_rad=float(raise_err.max()), raise_tail_worst_motor=IDS[int(raise_err.argmax())],
        raise_tail_preserved_max_error_rad=float(raise_err[preserved].max()),
        final_tail_max_error_rad=float(err.max()), final_tail_worst_index31=int(err.argmax()),
        nonzero_tau_ff_from_controller=any(any(p["tau_ff"]) for p in after),
        clipping_samples_in_window=len(clipped), window_samples=len(window), balance=balance,
    ), rows


if __name__ == "__main__":
    runs = [Path(p) for p in sys.argv[1:]] or sorted((ROOT / "trials").iterdir())
    report, raw = {}, {}
    for run in runs:
        report[run.name], raw[run.name] = inspect(run)
    (ROOT / "analysis.json").write_text(json.dumps(report, indent=2) + "\n")
    fig, ax = plt.subplots(figsize=(10, 5))
    for name, rows in raw.items():
        t0 = rows[0]["at"]

        ax.plot([r["at"] - t0 for r in rows], [r["body_q_measured_motor"][24] for r in rows],
                color={"": "tab:blue", "24": "tab:green"}.get(",".join(map(str, report[name]["gravity_feedforward_motors"])), "tab:orange"), alpha=.8,
                label=f"{name} ({report[name]['terminal_phase'].lower()})")
    ax.axhline(-0.5984442830085754, color="black", linestyle="--", label="raised waypoint")
    ax.axhspan(-0.6484, -0.5484, color="gray", alpha=.15, label="±0.05 rad tolerance")
    ax.set_xlabel("Time since reset_ready (s)")
    ax.set_ylabel("Measured right shoulder yaw (rad)")
    ax.set_title("Unchanged (blue), sim-only gravity FF on all arm motors (orange) or right shoulder yaw only (green)")
    ax.grid(alpha=.25)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(ROOT / "torque-comparison.png", dpi=150)
    # Each condition's pass count; one success does not establish repeatability.
    counts = {}
    for r in report.values():
        key = ",".join(map(str, r["gravity_feedforward_motors"])) or "none"
        counts.setdefault(key, [0, 0])
        counts[key][0] += r["terminal_phase"] == "COMPLETED"
        counts[key][1] += 1
    print("completed/trials by feedforward motors:", counts)
    for name, r in report.items():
        b = r["balance"]["right_shoulder_yaw"]
        print(f"{name}: {r['terminal_phase']}/{r['terminal_reason']} stage1={r['stage1_entered_s']} "
              f"dwell={r['max_raise_dwell_s']:.2f} lead={r['max_reference_lead_rad']:.3f} "
              f"final_err={r['final_tail_max_error_rad']:.3f}@{r['final_tail_worst_index31']} "
              f"yaw: gap={b['command_minus_measured_rad']:+.3f} applied={b['applied_actuator_force_nm']:+.2f} "
              f"grav={b['static_gravity_hold_nm']:+.2f} dq={b['velocity_rad_s']:+.3f}")
