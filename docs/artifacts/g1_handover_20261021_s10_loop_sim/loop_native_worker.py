"""Production native harness loop driving a served LeRobot policy (s10) from the simulator's head camera.

SIMULATION FIXTURE ONLY. Unlike g1_handover_20261019_loop_sim, nothing here stands in for the policy or the camera:
- run_vla_inference talks to sonic_policy_server.py through its LeRobot client, because the profile has a
  lerobot_policy section;
- the camera is the simulator's rendered D435 head view, published on the native loop's camera port.

The Isaac-GR00T client module is stubbed only because run_vla_inference imports it unconditionally; it is never
called.

Start pose: as in the 20261019 loop, the probe's own first reset_ready, before the autopilot claims, is a fixture
pre-pose to the recorded standing start pose:
- it covers the waist and both arms, while the box is 3 m away;
- hands keep their measured posture;
- it runs under the rate limit and tracking trim, but without the lead guard;
- it settles within 0.12 rad.

Every later ready return, including the autopilot's startup return, is the unmodified production decision and
motion. Ready-return references are traced with the paired tracer from g1_handover_20261014_tracking.
"""
import argparse
import json
from pathlib import Path
import sys
import time
from types import ModuleType

import numpy as np

ROOT = Path(__file__).parent
PRE_POSE_TOLERANCE_RAD = 0.12
sys.path.insert(0, str(ROOT.parent / "g1_handover_20261014_tracking"))
import native_worker as traced  # noqa: E402
from gear_sonic.scripts import run_vla_inference as loop  # noqa: E402
from gear_sonic.utils.inference import harness_control  # noqa: E402
from gear_sonic.utils.inference.standing_reset import StandingReset, _measured_joints  # noqa: E402


class PrePose(StandingReset):
    """Fixture-only approach to the start pose: rate-limited reference plus trim, no lead guard."""

    def advance(self, state, dt):
        if (not isinstance(state, dict) or isinstance(dt, bool) or not isinstance(dt, (int, float))
                or not np.isfinite(dt) or dt <= 0):
            return self.command
        try:
            measured = _measured_joints(state)
        except (TypeError, ValueError):
            return self.command
        dt = min(dt, 0.05)
        reference = self._target.copy()
        self._nominal_reached |= bool(np.max(np.abs(self._position[:17] - self._target[:17])) < 1e-6)
        if self._nominal_reached:
            error = self._target[:17] - measured[:17]
            error = np.where(np.abs(error) > self._tracking_deadband, error, 0.0)
            self._tracking_trim = np.clip(self._tracking_trim + dt * error, -0.15, 0.15)
            reference[:17] += self._tracking_trim
        self._position = self._position + np.clip(reference - self._position, -0.5 * dt, 0.5 * dt)
        return self.command

    def is_settled(self, state, joint_tolerance, yaw_tolerance_rad):
        return super().is_settled(state, max(joint_tolerance, PRE_POSE_TOLERANCE_RAD), yaw_tolerance_rad)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--endpoint", required=True)
    args = parser.parse_args()
    start = json.loads((ROOT / "start-pose.json").read_text())

    def record(event, **fields):
        with (args.output / "native-trace.jsonl").open("a") as stream:
            stream.write(json.dumps(dict(at=time.monotonic(), event=event, **fields)) + "\n")

    class UnusedIsaacClient:
        def __init__(self, **kwargs):
            raise RuntimeError("this fixture serves a LeRobot policy; the Isaac-GR00T client must not be used")

    class LoopControl(traced.PairedTraceControl):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            begin, returns = self.hooks.begin_ready_reset, []

            def begin_ready_reset(feedback, target, waypoint):
                reset = begin(feedback, target, waypoint)
                returns.append(reset.stage)
                if len(returns) == 1:  # The probe's pre-pose, not part of the loop under test.
                    reset.stage = 1
                    reset._target[:17] = start["upper_body_planner_order"]
                    reset._target[17:] = reset._position[17:]  # Hands as measured.
                    reset._final_target = reset._target.copy()
                    reset.__class__ = PrePose
                    record("fixture_start_pre_pose")
                else:
                    record("ready_return_begin", index=len(returns) - 1, production_stage=returns[-1])
                return reset

            self.hooks.begin_ready_reset = begin_ready_reset

    module = ModuleType("gr00t.policy.server_client")
    module.PolicyClient = UnusedIsaacClient
    sys.modules["gr00t.policy.server_client"] = module
    loop.planner_command_in_reference_frame = traced.paired_convert
    harness_control.HarnessControl = LoopControl
    config = loop.InferenceConfig(camera_host="127.0.0.1", camera_port=11555,
        state_zmq_host="127.0.0.1", state_zmq_port=11557,
        action_zmq_host="127.0.0.1", action_zmq_port=11556,
        keyboard_zmq_host="127.0.0.1", keyboard_zmq_port=11580,
        harness_endpoint=args.endpoint, harness_profile=str(args.profile))
    record("configuration", native_main_loop=True, served_lerobot_policy=True, rendered_head_camera=True,
           robot_connection=False)
    loop.main(config)


if __name__ == "__main__":
    main()
