"""Per-trial outcome and raise-stage blockers for the waypoint candidate trials.

Usage: summarize.py [TRIAL_DIR...]  (default: every folder under trials/); writes trials.json.
"""
import gzip
import json
from pathlib import Path
import sys

import numpy as np
import yaml

ROOT = Path(__file__).parent
IDS = [12, 13, 14, 15, 22, 16, 23, 17, 24, 18, 25, 19, 26, 20, 27, 21, 28]  # planner order -> motor


def lines(path):
    if not path.exists():
        return [json.loads(s) for s in gzip.open(str(path) + ".gz", "rt")]
    return [json.loads(s) for s in path.read_text().splitlines()]


def summarize(run):
    result = json.loads((run / "result.json").read_text())
    rows = lines(run / "paired-reference-feedback.jsonl")
    raised = [r for r in rows if r["stage"] == 0][-40:]
    error = np.array([np.abs(np.asarray(r["body_q_measured_motor"])[IDS] - r["active_target"][:17])
                      for r in raised]).max(0)
    stage1 = next((r["at"] - rows[0]["at"] for r in rows if r["stage"] == 1), None)
    return dict(
        raised_waypoint_rad=list(yaml.safe_load((run / "simulation-profile.yaml").read_text())
                                ["handover"]["raised_right_arm_joints"].values()),
        terminal_phase=result["terminal_status"]["phase"],
        terminal_reason=result["terminal_status"]["reason"], duration_s=rows[-1]["at"] - rows[0]["at"],
        stage1_entered_s=stage1, max_raise_dwell_s=max(r["lift_dwell_s"] or 0 for r in rows),
        raise_tail_worst={str(IDS[i]): float(error[i]) for i in np.argsort(-error)[:3]},
        physics_modified=result.get("physics_modified"), owned_processes_stopped=result["owned_processes_stopped"],
    )


if __name__ == "__main__":
    runs = [Path(p) for p in sys.argv[1:]] or sorted((ROOT / "trials").iterdir())
    report = {run.name: summarize(run) for run in runs}
    (ROOT / "trials.json").write_text(json.dumps(report, indent=2) + "\n")
    for name, r in report.items():
        worst = ", ".join(f"m{k} {v:.3f}" for k, v in r["raise_tail_worst"].items())
        print(f"{name}: {r['terminal_phase']}/{r['terminal_reason']} t={r['duration_s']:.1f}s "
              f"stage1={r['stage1_entered_s']} dwell={r['max_raise_dwell_s']:.2f} raise-tail worst: {worst}")
