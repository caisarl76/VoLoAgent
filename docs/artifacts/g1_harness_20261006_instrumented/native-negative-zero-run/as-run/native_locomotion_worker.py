"""Trace actual native-loop locomotion; scripted prewarm never actuates."""

import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
import sys
import threading
import time
from types import ModuleType

import numpy as np

from gear_sonic.scripts import run_vla_inference as loop
from gear_sonic.utils.inference import harness_control
from gear_sonic.utils.inference.bounded_planner import wrap
from gear_sonic.utils.teleop.xr_upperbody_bridge import feedback_payload_heading_yaw


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--profile', type=Path, required=True)
    parser.add_argument('--endpoint', required=True)
    args = parser.parse_args()
    lock = threading.Lock()
    control = None

    def record(event, **fields):
        with lock, (args.output/'native-trace.jsonl').open('a') as stream:
            stream.write(json.dumps(dict(at=time.monotonic(), event=event, **fields),
                                    default=lambda v: v.tolist())+'\n')

    class Camera:
        def __init__(self, **kwargs):
            self.count = 0

        def read(self):
            self.count += 1
            return dict(timestamps={'ego_view': time.monotonic()},
                        images={'ego_view': np.full((48, 64, 3), self.count % 255, np.uint8)})

        def close(self):
            pass

    class PolicyClient:
        def __init__(self, **kwargs):
            pass

        def get_action(self, observation):
            record('scripted_prewarm')
            return dict(motion_token=np.zeros((1, 40, 64), np.float32),
                        left_hand_joints=np.zeros((1, 40, 7), np.float32),
                        right_hand_joints=np.zeros((1, 40, 7), np.float32)), {}

        def close(self):
            pass

    class TraceControl(harness_control.HarnessControl):
        def __init__(self, *a, **kw):
            nonlocal control
            super().__init__(*a, **kw)
            control = self
            self.last_trace_at, self.last_trace_phase = 0.0, None

        def tick(self, now, feedback, received_at):
            planner = self.planner
            previous = None if planner is None else (planner.yaw, planner.last_at)
            super().tick(now, feedback, received_at)
            if planner is not None and planner.kind == 'turn' and previous is not None:
                measured = None if self.feedback is None else feedback_payload_heading_yaw(self.feedback)
                if measured is not None:
                    dt = min(0.05, max(0.0, now-previous[1]))
                    record('turn_sample', phase=self.phase, goal_yaw_rad=planner.goal,
                           measured_yaw_rad=measured, previous_reference_yaw_rad=previous[0],
                           reference_yaw_rad=planner.yaw, dt_s=dt,
                           telemetry_index=self.feedback_index, reason=self.reason,
                           lead_rad=wrap(planner.yaw-measured),
                           required_retraction_rate_rps=(max(0, abs(wrap(previous[0]-measured))-planner.limits.turn_lead_rad)/dt
                                                         if dt > 0 else None),
                           controller_target_quaternion=self.feedback.get('base_quat_target'))
            if self.phase != self.last_trace_phase or now-self.last_trace_at >= 0.1:
                record('status', **self.status(now))
                self.last_trace_at, self.last_trace_phase = now, self.phase

    original_convert = loop.planner_command_in_reference_frame
    last_command_at = 0.0

    def traced_convert(command, feedback):
        nonlocal last_command_at
        converted = original_convert(command, feedback)
        now = time.monotonic()
        if now-last_command_at >= 0.1:
            record('planner_conversion', phase=None if control is None else control.phase,
                   telemetry_index=feedback.get('index'), measured_heading=feedback_payload_heading_yaw(feedback),
                   world=asdict(command), reference=asdict(converted))
            last_command_at = now
        return converted

    def forbidden_latent(*a, **kw):
        record('unexpected_latent_publication_attempt')
        raise AssertionError('Scripted prewarm must never reach the action publisher')

    module = ModuleType('gr00t.policy.server_client')
    module.PolicyClient = PolicyClient
    sys.modules['gr00t.policy.server_client'] = module
    loop.ComposedCameraClientSensor = Camera
    loop.planner_command_in_reference_frame = traced_convert
    loop.pack_latent_action_message = forbidden_latent
    harness_control.HarnessControl = TraceControl
    config = loop.InferenceConfig(camera_host='127.0.0.1', camera_port=11555,
        state_zmq_host='127.0.0.1', state_zmq_port=11557,
        action_zmq_host='127.0.0.1', action_zmq_port=11556,
        keyboard_zmq_host='127.0.0.1', keyboard_zmq_port=11580,
        harness_endpoint=args.endpoint, harness_profile=str(args.profile), harness_locomotion=True)
    record('configuration', native_main_loop=True, scripted_camera=True,
           scripted_policy=True, robot_connection=False, max_turn_rate_rps=math.radians(10))
    loop.main(config)


if __name__ == '__main__':
    main()
