"""Observe exact production planner feedback and references in loopback sim."""
import json
from pathlib import Path
import sys
import time

import numpy as np

from gear_sonic.scripts import run_vla_inference as loop
from gear_sonic.utils.inference import harness_control

ROOT = Path(__file__).parent
CONTROL = None
ORIGINAL_CONTROL = harness_control.HarnessControl
ORIGINAL_CONVERT = loop.planner_command_in_reference_frame


class PairedTraceControl(ORIGINAL_CONTROL):
    def __init__(self, *args, **kwargs):
        global CONTROL
        super().__init__(*args, **kwargs)
        CONTROL = self


def paired_convert(command, feedback):
    converted = ORIGINAL_CONVERT(command, feedback)
    if CONTROL is not None and CONTROL.phase == "RESETTING":
        reset = CONTROL.reset
        output = Path(sys.argv[sys.argv.index("--output") + 1])
        now = time.monotonic()
        sample = dict(
            at=now, phase=CONTROL.phase, stage=reset.stage,
            telemetry_index=feedback.get("index"),
            feedback_received_at=CONTROL.feedback_time,
            body_q_measured_motor=np.asarray(feedback["body_q_measured_motor"]).tolist(),
            left_hand_q_measured=np.asarray(feedback["left_hand_q_measured"]).tolist(),
            right_hand_q_measured=np.asarray(feedback["right_hand_q_measured"]).tolist(),
            base_quat=np.asarray(feedback["base_quat"]).tolist(),
            upper_body_reference=command.upper_body_position,
            right_hand_reference=command.right_hand_position,
            active_target=reset._target.tolist(), final_target=reset.target.tolist(),
            tracking_trim=reset._tracking_trim.tolist(),
            nominal_reference_reached=bool(reset._nominal_reached),
            lift_dwell_s=reset._lift_dwell,
        )
        with (output / "paired-reference-feedback.jsonl").open("a") as stream:
            stream.write(json.dumps(sample, allow_nan=False) + "\n")
    return converted


if __name__ == "__main__":
    harness_control.HarnessControl = PairedTraceControl
    loop.planner_command_in_reference_frame = paired_convert
    sys.path.insert(0, str(ROOT.parent / "g1_harness_20261006_followup"))
    from native_locomotion_worker import main
    main()
