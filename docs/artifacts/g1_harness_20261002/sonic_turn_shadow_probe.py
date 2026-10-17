"""Run the existing loopback probe with non-actuating turn diagnostics.

All arguments go to the original probe. The shadow candidate is logged only;
it never replaces a command or changes native limits.
"""

import importlib.util
import json
import math
from pathlib import Path
import sys

import numpy as np


def main():
    original = Path(__file__).parents[1] / "g1_harness_20261001/sonic_sim_probe.py"
    spec = importlib.util.spec_from_file_location("original_sonic_probe", original)
    probe = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(probe)
    output = Path(sys.argv[sys.argv.index("--output") + 1])
    stream = None

    class ShadowControl(probe.HarnessControl):
        def tick(self, now, feedback, received_at):
            nonlocal stream
            planner = self.planner
            previous = None if planner is None else (planner.yaw, planner.last_at)
            super().tick(now, feedback, received_at)
            if (
                previous is None or self.planner is not planner
                or planner.kind != "turn" or feedback is None
            ):
                return
            measured = probe.feedback_payload_heading_yaw(feedback)
            if measured is None:
                return
            error = probe.wrap(planner.goal - measured)
            shadow_goal = probe.wrap(
                planner.goal + np.sign(error) * planner.limits.turn_lead_rad
            )
            bounded_goal = probe.wrap(measured + float(np.clip(
                probe.wrap(shadow_goal - measured),
                -planner.limits.turn_lead_rad, planner.limits.turn_lead_rad,
            )))
            dt = min(0.05, max(0.0, now - previous[1]))
            shadow = probe.wrap(previous[0] + float(np.clip(
                probe.wrap(bounded_goal - previous[0]),
                -planner.rate * dt, planner.rate * dt,
            )))
            quat = feedback.get("base_quat_target")
            target_yaw = None
            if quat is not None and len(quat) == 4:
                w, x, y, z = np.asarray(quat) / np.linalg.norm(quat)
                target_yaw = math.atan2(2 * (w*z+x*y), 1-2*(y*y+z*z))
            if stream is None:
                stream = (output / "turn-shadow.jsonl").open("w")
            stream.write(json.dumps({
                "at": now, "phase": self.phase, "goal_yaw_rad": planner.goal,
                "measured_yaw_rad": measured, "facing_yaw_rad": planner.yaw,
                "controller_target_yaw_rad": target_yaw,
                "nominal_heading_reached": planner.nominal_heading_reached,
                "heading_trim_rad": planner.heading_trim,
                "shadow_goal_yaw_rad": shadow_goal,
                "shadow_bounded_goal_yaw_rad": bounded_goal,
                "shadow_candidate_yaw_rad": shadow,
                "shadow_minus_actual_rad": probe.wrap(shadow - planner.yaw),
                "lead_rad": probe.wrap(planner.yaw - measured),
                "shadow_lead_rad": probe.wrap(shadow - measured),
            }) + "\n")
            stream.flush()

    probe.HarnessControl = ShadowControl
    try:
        return probe.main()
    finally:
        if stream is not None:
            stream.close()


if __name__ == "__main__":
    raise SystemExit(main())
