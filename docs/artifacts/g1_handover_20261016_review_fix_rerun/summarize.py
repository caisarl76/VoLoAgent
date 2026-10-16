"""Summarize the post-review rerun trials with the pelvis-estimate summary; writes trials.json here.

Usage: summarize.py [TRIAL_DIR...]  (default: every folder under trials/)
"""
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).parent
spec = importlib.util.spec_from_file_location(
    "pelvis_summarize", ROOT.parent / "g1_handover_20261015_pelvis_estimate/summarize.py")
pelvis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pelvis)

if __name__ == "__main__":
    runs = [Path(p) for p in sys.argv[1:]] or sorted(p for p in (ROOT / "trials").iterdir() if p.is_dir())
    report = {run.name: pelvis.summarize(run) for run in runs}
    (ROOT / "trials.json").write_text(json.dumps(report, indent=2, default=float) + "\n")
    for name, r in report.items():
        print(name, r["terminal_phase"], r["decision"], round(r["min_conservative_desk_gap_m"] * 1000, 1), "mm",
              r["robot_desk_contact_samples"], r["robot_bottle_contact_samples"])
