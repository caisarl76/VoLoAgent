"""Summarize saved native-loop repeats, distinguishing guard state from wire output."""

import argparse
import hashlib
import json
import math
from pathlib import Path


def degrees(value):
    return math.degrees(math.atan2(math.sin(value), math.cos(value)))


def analyze(folder):
    trace_path = folder / "native-trace.jsonl"
    trace = [json.loads(s) for s in trace_path.read_text().splitlines()]
    result = json.loads((folder / "result.json").read_text())
    turns = [x for x in trace if x["event"] == "turn_sample"]
    active = [x for x in turns if x["phase"] == "TURNING"]
    origin = turns[0]["goal_yaw_rad"] - math.radians(result["turn_angle_deg"])
    final = turns[-1]
    # The interrupted row retains the old scalar reference when the update
    # raises. It is not the separate measured-heading hold sent on the wire.
    rate = max(abs(degrees(x["reference_yaw_rad"] - x["previous_reference_yaw_rad"])) / x["dt_s"]
               for x in active)
    lead = max(abs(degrees(x["reference_yaw_rad"] - x["measured_yaw_rad"])) for x in active)
    assert rate <= 10.00001 and lead <= 5.00001
    assert result["turn_success"] is False
    assert result["turn_interruption_hold"] == "passed_fresh_ack_and_no_translation_or_resume"
    assert result["latent_wire_count"] == 0
    assert result["post_interrupt_stationary_packets"] >= 20
    assert result["hold_status"]["hold_confirmed"] is True
    assert result["hold_status"]["telemetry_index"] > result["turn_status"]["telemetry_index"]
    assert result["owned_process_exit_codes"] == {"mujoco": -15, "sonic": -15, "native": 0}
    return {"case": folder.name, "turn_success": False,
            "requested_turn_deg": result["turn_angle_deg"],
            "post_walk_hold_s": result["post_walk_hold_s"],
            "turn_elapsed_s": result["turn_elapsed_s"],
            "entry_yaw_deg": degrees(origin),
            "final_measured_progress_deg": degrees(final["measured_yaw_rad"] - origin),
            "final_measured_goal_error_deg": abs(degrees(final["goal_yaw_rad"] - final["measured_yaw_rad"])),
            "active_reference_max_rate_deg_s": rate, "active_reference_max_lead_deg": lead,
            "terminal_retained_reference_lead_deg": degrees(final["lead_rad"]),
            "terminal_required_retraction_rate_deg_s": (None if final["required_retraction_rate_rps"] is None
                                                        else math.degrees(final["required_retraction_rate_rps"])),
            "terminal_reason": result["turn_status"]["reason"],
            "hold_ack_elapsed_s": result["hold_ack_elapsed_s"],
            "post_interrupt_stationary_packets": result["post_interrupt_stationary_packets"],
            "latent_wire_count": 0, "owned_process_cleanup_passed": True,
            "sources": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                        for p in [trace_path, folder / "events.jsonl", folder / "result.json"]}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    result = {"scope": "Two simulator repeats; one trial per configuration, not repeatability acceptance.",
              "cases": [analyze(args.root / name) for name in ("native-repeat-baseline", "native-repeat-pause")],
              "limits_changed": False, "production_changed": False,
              "inference": "A one-second stationary request pause did not solve turning. Entry physical state/timing remain confounded.",
              "guard_scope": "The new baseline closes one actual-native-loop lead/rate-guard interruption-and-hold check, not every possible guard failure."}
    (args.root / "turn-repeat-summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
