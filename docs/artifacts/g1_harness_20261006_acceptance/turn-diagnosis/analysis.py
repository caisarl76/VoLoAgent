"""Offline, stdlib-only audit of four archived native-loop turn traces.

No controller imports, process launches, network calls or archive writes.
Times are host monotonic log times, not controller capture timestamps.
"""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics


CASES = ("negative-zero", "positive-zero", "negative-wrap", "positive-wrap")


def wrap(value):
    return math.atan2(math.sin(value), math.cos(value))


def deg(value):
    return math.degrees(wrap(value))


def yaw(quat):
    w, x, y, z = quat
    norm2 = sum(v * v for v in quat)
    return math.atan2(2 * (w * z + x * y) / norm2, 1 - 2 * (y * y + z * z) / norm2)


def angle(vector):
    return math.atan2(vector[1], vector[0])


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def clip(value, bound):
    return min(bound, max(-bound, value))


def analyze(folder, output):
    result = json.loads((folder / "result.json").read_text())
    trace = rows(folder / "native-trace.jsonl")
    events = rows(folder / "events.jsonl")
    turn = [r for r in trace if r["event"] == "turn_sample"]
    goal = turn[0]["goal_yaw_rad"]
    origin = wrap(goal - math.radians(result["turn_angle_deg"]))
    t0, end = turn[0]["at"], turn[-1]["at"]
    sign = math.copysign(1, result["turn_angle_deg"])
    tail = [r for r in turn if r["at"] >= end - 2]
    last_index = -1
    spans, span = [], None
    unique = []
    for row in turn:
        if row["telemetry_index"] <= last_index:
            continue
        last_index = row["telemetry_index"]
        unique.append(row)
        if abs(deg(goal - row["measured_yaw_rad"])) <= 3:
            if span is None:
                span = {"start_s": row["at"] - t0, "end_s": row["at"] - t0, "indices": 0}
            span["end_s"] = row["at"] - t0
            span["indices"] += 1
        elif span is not None:
            spans.append(span)
            span = None
    if span is not None:
        spans.append(span)
    for span in spans:
        span["duration_s"] = span["end_s"] - span["start_s"]

    # Independently reconstruct the scalar rate/lead/trim recurrence from saved
    # dt and measured heading. Logging happens after tick, so dwell transition
    # clock is approximate; stop recurrence at the first recorded frozen ref
    # after a >=0.5 s in-tolerance interval.
    trim, nominal, done = 0.0, False, False
    replay_errors = []
    first_finish = next((s["start_s"] + .5 for s in spans if s["duration_s"] >= .5), None)
    trims = []
    for row in turn:
        prev = row["previous_reference_yaw_rad"]
        measured = row["measured_yaw_rad"]
        if row["phase"] != "INTERRUPTED" and not done:
            nominal |= abs(wrap(goal - prev)) < 1e-6
            if nominal and abs(wrap(goal - measured)) > math.radians(1.5):
                trim = clip(trim + row["dt_s"] * wrap(goal - measured), math.radians(5))
            bounded = wrap(measured + clip(wrap(goal + trim - measured), math.radians(5)))
            predicted = wrap(prev + clip(wrap(bounded - prev), math.radians(10) * row["dt_s"]))
            replay_errors.append(abs(deg(predicted - row["reference_yaw_rad"])))
            if first_finish is not None and row["at"] - t0 >= first_finish:
                done = True
        trims.append(trim)

    conversions = [r for r in trace if r["event"] == "planner_conversion" and t0 <= r["at"] <= end]
    offsets = [wrap(angle(r["world"]["facing"]) - angle(r["reference"]["facing"])) for r in conversions]
    wire = [r for r in events if r["event"] == "action_wire" and r["topic"] == "planner"]
    matches = []
    for conversion in conversions:
        candidates = [r for r in wire if 0 <= r["at"] - conversion["at"] <= .15]
        assert candidates
        matches.append(min(abs(deg(angle(r["fields"]["facing"]) - angle(conversion["reference"]["facing"]))) for r in candidates))
    physics = [r for r in rows(folder / "ground-truth.jsonl") if t0 <= r["at"] <= end]
    physics_matches = [(r, min(turn, key=lambda t: abs(t["at"] - r["at"]))) for r in physics]
    physics_errors = [abs(deg(yaw(p["floating_base_pose"][3:]) - t["measured_yaw_rad"])) for p, t in physics_matches]
    nominal_rows = [r for r in turn if abs(wrap(goal - r["reference_yaw_rad"])) < 1e-6]
    final = turn[-1]
    rate = max(abs(deg(r["reference_yaw_rad"] - r["previous_reference_yaw_rad"])) / r["dt_s"] for r in turn)
    lead = max(abs(deg(r["reference_yaw_rad"] - r["measured_yaw_rad"])) for r in turn)
    longest = max((s["duration_s"] for s in spans), default=0)
    assert rate <= 10.00001 and lead <= 5.00001
    assert max(replay_errors) < 1e-6, (folder.name, max(replay_errors))
    assert all(b["telemetry_index"] >= a["telemetry_index"] for a, b in zip(turn, turn[1:]))
    assert max(abs(deg(v - offsets[0])) for v in offsets) < 1e-10
    assert max(matches) < 1e-4
    assert result["turn_success"] == (longest >= .5)
    assert result["latent_wire_count"] == 0
    if not result["turn_success"]:
        assert result["turn_status"]["reason"] == "planner_error:Turn deadline exceeded"

    summary = {
        "case": folder.name,
        "success": result["turn_success"],
        "entry_yaw_deg": deg(origin),
        "requested_delta_deg": result["turn_angle_deg"],
        "recorded_turn_elapsed_s": result["turn_elapsed_s"],
        "sample_count": len(turn),
        "advancing_telemetry_count": len(unique),
        "duplicate_telemetry_count": len(turn) - len(unique),
        "max_saved_tick_gap_s": max(b["at"] - a["at"] for a, b in zip(turn, turn[1:])),
        "reference_raw_wrap_crossings": sum(abs(b["reference_yaw_rad"] - a["reference_yaw_rad"]) > math.pi for a, b in zip(turn, turn[1:])),
        "max_circular_reference_step_deg": max(abs(deg(b["reference_yaw_rad"] - a["reference_yaw_rad"])) for a, b in zip(turn, turn[1:])),
        "max_reference_rate_deg_s": rate,
        "max_reference_lead_deg": lead,
        "fraction_samples_at_4_99deg_lead": sum(abs(deg(r["lead_rad"])) >= 4.99 for r in turn) / len(turn),
        "final_measured_delta_deg": deg(final["measured_yaw_rad"] - origin),
        "final_reference_delta_deg": deg(final["reference_yaw_rad"] - origin),
        "final_motion_target_delta_deg": deg(yaw(final["controller_target_quaternion"]) - origin),
        "final_measured_error_deg": abs(deg(goal - final["measured_yaw_rad"])),
        "min_measured_error_deg": min(abs(deg(goal - r["measured_yaw_rad"])) for r in turn),
        "in_tolerance_intervals": spans,
        "max_in_tolerance_span_s": longest,
        "first_nominal_reference_s": nominal_rows[0]["at"] - t0 if nominal_rows else None,
        "reconstructed_final_trim_deg": deg(trims[-1]),
        "reference_recurrence_max_error_deg": max(replay_errors),
        "tail_2s_directional_reference_minus_target_deg": statistics.mean(sign * deg(r["reference_yaw_rad"] - yaw(r["controller_target_quaternion"])) for r in tail),
        "tail_2s_directional_target_minus_measured_deg": statistics.mean(sign * deg(yaw(r["controller_target_quaternion"]) - r["measured_yaw_rad"]) for r in tail),
        "tail_2s_measured_net_rate_deg_s": deg(tail[-1]["measured_yaw_rad"] - tail[0]["measured_yaw_rad"]) / (tail[-1]["at"] - tail[0]["at"]),
        "conversion_count": len(conversions),
        "conversion_frame_offset_deg": deg(offsets[0]),
        "conversion_offset_max_drift_deg": max(abs(deg(v - offsets[0])) for v in offsets),
        "conversion_to_nearby_received_wire_max_min_error_deg": max(matches),
        "physics_sample_count": len(physics),
        "nearest_host_time_physics_measured_mean_abs_error_deg": statistics.mean(physics_errors),
        "nearest_host_time_physics_measured_max_abs_error_deg": max(physics_errors),
        "hold_ack_elapsed_s": result.get("hold_ack_elapsed_s"),
        "post_interrupt_stationary_packets": result.get("post_interrupt_stationary_packets"),
        "controller_binary_sha256": result["controller_binary_sha256"],
        "source_sha256": {str(p.relative_to(folder)): hashlib.sha256(p.read_bytes()).hexdigest() for p in [folder / "native-trace.jsonl", folder / "events.jsonl", folder / "ground-truth.jsonl", folder / "result.json", folder / "as-run/bounded_planner.py"]},
    }
    with (output / (folder.name + "-timeline.csv")).open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["time_s", "telemetry_index", "phase", "reference_delta_deg", "motion_target_delta_deg", "measured_delta_deg", "measured_error_deg", "trim_deg"])
        for row, trim in zip(turn, trims):
            writer.writerow([row["at"] - t0, row["telemetry_index"], row["phase"], deg(row["reference_yaw_rad"] - origin), deg(yaw(row["controller_target_quaternion"]) - origin), deg(row["measured_yaw_rad"] - origin), deg(goal - row["measured_yaw_rad"]), deg(trim)])
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traces", type=Path, default=Path(__file__).resolve().parents[2] / "g1_harness_20261006_followup/native-turns")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    cases = [analyze(args.traces / case, args.output) for case in CASES]
    document = {
        "scope": "Offline saved traces only; no new simulator or hardware evidence.",
        "limits": {"rate_deg_s": 10, "lead_deg": 5, "tolerance_deg": 3, "dwell_s": .5, "deadline_s": 10},
        "cases": cases,
        "measured_findings": [
            "Both deadline decisions agree with advancing-index measured dwell; positive-wrap briefly enters tolerance twice but never for 0.5 s.",
            "Both failures consume nearly the whole facing-to-measured lead allowance, with separate motion-target and measured-heading deficits.",
            "Recorded scalar reference evolution reproduces at numerical precision with the archived lead/rate/trim recurrence.",
            "Both signs and wrap crossings produce correctly directed motion-target and measured progress; conversion offset is stable in each run.",
        ],
        "inferences": [
            "Insufficient closed-loop tracking under the unchanged lead/deadline limits explains these failures; the traces do not isolate model bias, dynamic lag, plant/contact state, or internal replanning effects.",
            "Changing post-nominal heading trim alone cannot solve negative-zero while the same lead constraint remains saturated; its trim never activates.",
        ],
        "limitations": [
            "One trial per configuration; setup heading, physical state and timing are confounded.",
            "base_quat_target is the heading-corrected current motion-frame target, not a direct decoder motor output or full predicted trajectory.",
            "Frame telemetry quaternion, planner context, full predicted trajectory and exact inference-to-command timestamps are absent.",
            "Wire receipt times and physics samples are host-timestamped and asynchronous; nearby matching does not establish command identity or controller latency.",
            "Dwell uses post-tick log times; comparison to control clock is approximate by logging overhead. Both failed dwell spans are far below 0.5 s.",
        ],
        "smallest_discriminating_simulator_experiment": {
            "status": "Proposed only; not run.",
            "case": "One negative-zero repeat using the existing actual native loop and unchanged limits, startup, backward command, model files and simulator.",
            "only_change": "Observability: timestamped, indexed controller input/trajectory/active-frame capture into preallocated memory with output after the turn.",
            "capture": ["received facing vector and source sequence", "reference heading quaternion", "effective ONNX facing input and context", "generated root-yaw trajectory and frame indices", "active motion-frame quaternion", "measured base quaternion", "inference start/end and publication times"],
            "discriminants": {
                "frame_or_input_error": "Recovered world-facing request differs from the Python request beyond measured quantization/timestamp mismatch.",
                "planner_trajectory_shortfall": "Effective facing is correct, but generated future root-yaw trajectory consistently remains short.",
                "trajectory_scheduling_lag": "Future trajectory reaches facing, but active frames remain behind or are repeatedly replaced before reaching it.",
                "lower_controller_or_plant_tracking": "Active target is correct, but measured yaw persistently trails it; motor/decoder/contact evidence would then be needed to separate causes.",
            },
            "acceptance": "Use the original 3 degree / 0.5 second measured criterion and 10 second deadline; preserve hold verification regardless of outcome. One repeat diagnoses boundaries, not repeatable acceptance.",
        },
    }
    (args.output / "findings.json").write_text(json.dumps(document, indent=2) + "\n")
    for case in cases:
        print(f"{case['case']}: success={case['success']}, error={case['final_measured_error_deg']:.6f} deg, dwell={case['max_in_tolerance_span_s']:.6f} s, replay_error={case['reference_recurrence_max_error_deg']:.3g} deg")


if __name__ == "__main__":
    main()
