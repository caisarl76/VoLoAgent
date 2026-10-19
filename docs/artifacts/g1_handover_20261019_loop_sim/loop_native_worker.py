"""Production native harness loop with a recorded-cycle stand-in policy.

SIMULATION FIXTURE ONLY. The policy client serves replay-cycle.npz: one
recorded handover cycle (reach, grasp, palm-up offer) from
handover_bottle_261019. Playback starts at the first inference of each
manipulation execution and then holds the final offer frame. Calls outside
MANIPULATING, which include prewarm, get the first frames and are never
published. The camera is a synthetic stream: the vision decisions come from
a simulator oracle in the probe, not from these images. Ready-return references
are traced with the paired tracer from g1_handover_20261014_tracking.

Start pose: the probe's own first reset_ready, before the autopilot claims, is a
fixture pre-pose to the recorded standing start pose (start-pose.json): waist
and both arms. Hands keep their measured posture, because the dataset's hand
joint order is not the controller's Dex3 order. It runs while the box is 3 m away, under the rate
limit and tracking trim but without the lead guard. Like the post-handover entry
fixture, it settles within 0.12 rad. Every later ready return, including the
autopilot's startup return, is the unmodified production decision and motion.
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

CHUNK = 40


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--endpoint", required=True)
    args = parser.parse_args()
    start = json.loads((ROOT / "start-pose.json").read_text())
    replay = np.load(ROOT / "replay-cycle.npz")
    frames = {k: replay[k] for k in ("motion_token", "left_hand_joints", "right_hand_joints")}
    playback = dict(execution=None, started=None)

    def record(event, **fields):
        with (args.output / "native-trace.jsonl").open("a") as stream:
            stream.write(json.dumps(dict(at=time.monotonic(), event=event, **fields)) + "\n")

    class Camera:
        def __init__(self, **kwargs):
            self.count = 0

        def read(self):
            self.count += 1
            return dict(timestamps={"ego_view": time.monotonic()},
                        images={"ego_view": np.full((48, 64, 3), self.count % 255, np.uint8)})

        def close(self):
            pass

    class RecordedPolicy:
        def __init__(self, **kwargs):
            pass

        def get_action(self, observation):
            control, now = traced.CONTROL, time.monotonic()
            index = 0
            if control is not None and control.phase == "MANIPULATING" and control.execution:
                execution = control.execution["execution_id"]
                if playback["execution"] != execution:
                    playback.update(execution=execution, started=now)
                    record("replay_start", execution_id=execution)
                index = int((now - playback["started"]) * 50)
            else:
                record("replay_prewarm_or_idle", phase=None if control is None else control.phase)
            rows = np.minimum(np.arange(index, index + CHUNK), len(frames["motion_token"]) - 1)
            return {k: v[rows][None].astype(np.float32) for k, v in frames.items()}, {}

        def close(self):
            pass

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
    module.PolicyClient = RecordedPolicy
    sys.modules["gr00t.policy.server_client"] = module
    loop.ComposedCameraClientSensor = Camera
    loop.planner_command_in_reference_frame = traced.paired_convert
    harness_control.HarnessControl = LoopControl
    config = loop.InferenceConfig(camera_host="127.0.0.1", camera_port=11555,
        state_zmq_host="127.0.0.1", state_zmq_port=11557,
        action_zmq_host="127.0.0.1", action_zmq_port=11556,
        keyboard_zmq_host="127.0.0.1", keyboard_zmq_port=11580,
        harness_endpoint=args.endpoint, harness_profile=str(args.profile))
    record("configuration", native_main_loop=True, recorded_cycle_policy=True,
           replay_frames=int(len(frames["motion_token"])), synthetic_camera=True, robot_connection=False)
    loop.main(config)


if __name__ == "__main__":
    main()
