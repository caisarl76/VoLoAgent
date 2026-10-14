"""Compare the final lift references with saved simulated motor feedback."""
import bisect
import json
from pathlib import Path
import sys

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "g1-harness-native"))
from gear_sonic.utils.teleop.xr_upperbody_bridge import G1_UPPER_BODY_JOINT_INDICES
from gear_sonic.utils.inference.harness_profile import RIGHT_ARM_BOUNDS

ROOT = Path(__file__).parent
RUN = ROOT / "native-return-final"


if __name__ == "__main__":
    result = json.loads((RUN / "result.json").read_text())
    assert result["terminal_status"]["reason"] == "reset_timeout"
    events = [json.loads(s) for s in (RUN / "events.jsonl").read_text().splitlines()]
    end = next(r["at"] for r in events if r["event"] == "ready_terminal")
    rows = [json.loads(s) for s in (RUN / "ground-truth.jsonl").read_text().splitlines()]
    times = [r["received_monotonic"] for r in rows]
    trace = [json.loads(s) for s in (RUN / "native-trace.jsonl").read_text().splitlines()]
    commands = [r for r in trace if r["event"] == "planner_conversion"
                and r["phase"] == "RESETTING" and end - 1 <= r["at"] < end]
    assert commands
    settings = yaml.safe_load((RUN / "simulation-profile.yaml").read_text())["handover"]
    joints = list(RIGHT_ARM_BOUNDS)
    planner_indices = [G1_UPPER_BODY_JOINT_INDICES.index(i) for i in range(22, 29)]
    target = np.array([settings["raised_right_arm_joints"][name] for name in joints])
    samples = []
    for command in commands:
        index = max(0, bisect.bisect_right(times, command["at"]) - 1)
        row = rows[index]
        measured = np.asarray(row["body_q"])[22:29]
        reference = np.asarray(command["world"]["upper_body_position"])[planner_indices]
        samples.append(dict(at=command["at"], saved_sample_age_s=command["at"] - times[index],
                            measured_rad=measured.tolist(), reference_rad=reference.tolist()))
    measured = np.array([r["measured_rad"] for r in samples])
    reference = np.array([r["reference_rad"] for r in samples])
    errors = np.abs(measured - target)
    worst = int(np.argmax(np.mean(errors, axis=0)))
    report = dict(
        terminal_reason="reset_timeout", measured_ready_settling_passed=False,
        worst_raised_joint=joints[worst], raised_target_rad=float(target[worst]),
        mean_measured_rad=float(np.mean(measured[:, worst])),
        mean_commanded_rad=float(np.mean(reference[:, worst])),
        mean_measured_error_rad=float(np.mean(errors[:, worst])),
        mean_command_lead_rad=float(np.mean(np.abs(reference[:, worst] - measured[:, worst]))),
        max_saved_sample_age_s=max(r["saved_sample_age_s"] for r in samples),
        qualifying_saved_right_arm_samples=int(np.sum(np.max(errors, axis=1) <= .05)),
        measurement_scope="About 20 Hz saved simulation feedback, nearest preceding state; not synchronized command/measurement metrology.",
        tuning_applied=False, hardware_acceptance=False, samples=samples,
    )
    (ROOT / "tracking-diagnosis.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "samples"}, indent=2))
