"""Check saved production native-return physics, without sending commands."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT.parent / "g1_handover_20261013_lift"))
from analyze_native import analyze


if __name__ == "__main__":
    report = analyze(ROOT / "native-return-final")
    report["last_pre_terminal_0p5_s_max_right_arm_error_rad"] = report.pop(
        "last_pre_completion_0p5_s_max_right_arm_error_rad"
    )
    report["acceptance_passed"] = bool(
        report["final_ready_settling_passed"]
        and report["grounded_standing_screen"]["ready"]
        and report["last_pre_terminal_0p5_s_max_right_arm_error_rad"] <= .05
        and report["minimum_conservative_sampled_desk_gap"]["gap_m"] > 0
        and report["sampled_hand_or_forearm_desk_contact_states"] == 0
        and report["owned_processes_stopped"]
    )
    report.update(production_motion_implementation=True, artifact_motion_wrapper=False,
                  entry_pose="Default standing, not an actual post-handover pose",
                  full_vla_cycle_tested=False, continuous_swept_clearance_certified=False)
    (ROOT / "analysis.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "samples"}, indent=2))
    raise SystemExit(0 if report["acceptance_passed"] else 1)
