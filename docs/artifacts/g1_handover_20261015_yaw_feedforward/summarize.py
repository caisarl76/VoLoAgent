"""Summarize the yaw-feedforward trials: return outcome plus delivered tau vs MuJoCo gravity.

Usage: summarize.py [TRIAL_DIR...]  (default: every folder under trials/); writes trials.json.
Standing runs (pd-trace) reuse the torque probe's analysis; offer-entry runs reuse
the post-handover-entry summary (desk gap, contacts, preserved joints).
"""
import importlib.util
import json
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).parent
ART = ROOT.parent
os.environ.setdefault("MPLCONFIGDIR", str(Path(os.environ.get("TMPDIR", "/tmp")) / "volo-ff-mpl"))


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


torque = load("torque_analyze", ART / "g1_handover_20261014_torque_probe/analyze.py")
entry = load("entry_summarize", ART / "g1_handover_20261014_post_handover_entry/summarize.py")


def delivered_tau(run):
    """Controller tau on motor 24 against the simulator's static gravity at the same sample."""
    pd = [p for p in torque.lines(run / "pd-trace.jsonl") if p["kp"][24] > 0]
    tau = np.array([p["tau_ff"] for p in pd])
    gravity = np.array([p["static_gravity"][24] for p in pd])
    on = tau[:, 24] != 0
    return dict(
        samples=len(pd), samples_with_tau=int(on.sum()),
        other_motor_tau_max_nm=float(np.abs(np.delete(tau, 24, axis=1)).max()),
        tau_range_nm=[float(tau[on, 24].min()), float(tau[on, 24].max())] if on.any() else None,
        tau_minus_gravity_max_abs_nm=float(np.abs(tau[on, 24] - gravity[on]).max()) if on.any() else None,
        tau_minus_gravity_mean_nm=float((tau[on, 24] - gravity[on]).mean()) if on.any() else None,
    )


def summarize(run):
    result = json.loads((run / "result.json").read_text())
    native = (run / "native.log").read_text()
    report = dict(decision_direct=native.count("ready return: direct"), decision_raise=native.count("raise first"))
    if result.get("entry"):
        report.update(entry.summarize(run))
    else:
        report.update(entry="grounded_standing",
                      terminal_phase=(result.get("terminal_status") or {}).get("phase"),
                      terminal_reason=(result.get("terminal_status") or {}).get("reason"))
        if (result.get("terminal_status") or {}).get("phase"):
            report.update(torque.inspect(run)[0])
    if (run / "pd-trace.jsonl").exists() or (run / "pd-trace.jsonl.gz").exists():
        report["delivered_tau_motor24"] = delivered_tau(run)
    return report


if __name__ == "__main__":
    runs = [Path(p) for p in sys.argv[1:]] or sorted(p for p in (ROOT / "trials").iterdir() if p.is_dir())
    report = {run.name: summarize(run) for run in runs}
    (ROOT / "trials.json").write_text(json.dumps(report, indent=2, default=float) + "\n")
    counts = {}
    for r in report.values():
        counts.setdefault(r["entry"], [0, 0])
        counts[r["entry"]][0] += r["terminal_phase"] == "COMPLETED"
        counts[r["entry"]][1] += 1
    print("completed/trials:", counts)
    for name, r in report.items():
        print(name, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()
                     if k in ("terminal_phase", "terminal_reason", "decision_direct", "decision_raise", "stage1_entered_s",
                              "return_s", "min_conservative_desk_gap_m", "robot_desk_contacts",
                              "preserved_waist_left_arm_max_error_rad", "raise_tail_worst_motor",
                              "raise_tail_max_error_rad", "delivered_tau_motor24")})
