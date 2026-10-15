"""Summarize the online pelvis-estimate trials.

Usage: summarize.py [TRIAL_DIR...]  (default: every folder under trials/); writes trials.json.
Reuses the measured-desk summary (outcome, desk and bottle contacts). It adds the
logged ready-return decision and compares the logged pelvis estimate with the
simulator's pelvis height. True height is the pelvis z at the desk restore,
before the return starts; the floor is z = 0.
"""
import gzip
import importlib.util
import json
from pathlib import Path
import re
import sys

import numpy as np

ROOT = Path(__file__).parent


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


desk = load("desk_summarize", ROOT.parent / "g1_handover_20261015_desk_edge_015/summarize.py")
DECISION = re.compile(r"\[harness\] ready return: (direct|raise first) \((.*)\)$")


def summarize(run):
    report = desk.summarize(run)
    result = json.loads((run / "result.json").read_text())
    decisions = [m.groups() for m in map(DECISION.search, (run / "native.log").read_text().splitlines()) if m]
    report.update(desk_height_m=result.get("desk_height_m", 0.85), decision=decisions[-1] if decisions else None)
    estimate = re.search(r"pelvis ([0-9.]+) m", decisions[-1][1]) if decisions else None
    restored = ((result.get("pre_pose") or {}).get("desk") or {}).get("sim_time")
    if estimate and restored is not None:
        gt = [json.loads(s) for s in gzip.open(run / "ground-truth.jsonl.gz", "rt")]
        sim = np.array([g["sim_time"] for g in gt])
        start = int(np.where(np.diff(sim) < 0)[0][-1] + 1) if np.any(np.diff(sim) < 0) else 0
        sample = min(gt[start:], key=lambda g: abs(g["sim_time"] - restored))
        report.update(pelvis_estimate_m=float(estimate.group(1)), pelvis_true_m=float(sample["qpos"][2]))
    return report


if __name__ == "__main__":
    runs = [Path(p) for p in sys.argv[1:]] or sorted(p for p in (ROOT / "trials").iterdir() if p.is_dir())
    report = {run.name: summarize(run) for run in runs}
    (ROOT / "trials.json").write_text(json.dumps(report, indent=2, default=float) + "\n")
    for name, r in report.items():
        print(name, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()
                     if k in ("desk_height_m", "decision", "terminal_phase", "terminal_reason", "return_s",
                              "min_conservative_desk_gap_m", "robot_desk_contact_samples",
                              "robot_bottle_contact_samples", "pelvis_estimate_m", "pelvis_true_m", "error")})
