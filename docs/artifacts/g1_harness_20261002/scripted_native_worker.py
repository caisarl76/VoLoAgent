"""Actual VLA main loop with a scripted camera/client, for loopback faults only.

The real worker, observation preparation, epoch guards, RPC, LoopHooks and ZMQ
publisher run unchanged. Scripted latent results must never be published.
"""

import argparse
import json
from pathlib import Path
import queue
import sys
import threading
import time
from types import ModuleType, SimpleNamespace

import numpy as np

from gear_sonic.scripts import run_vla_inference as loop
from gear_sonic.utils.inference import harness_control


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--endpoint", required=True)
    args = parser.parse_args()
    trace_lock = threading.Lock()
    control = None

    def record(event, **fields):
        with trace_lock, (args.output / "native-trace.jsonl").open("a") as stream:
            stream.write(json.dumps({"at": time.monotonic(), "event": event, **fields}) + "\n")

    class Camera:
        def __init__(self, **kwargs):
            self.count = 0

        def read(self):
            self.count += 1
            return {
                "timestamps": {"ego_view": time.monotonic()},
                "images": {"ego_view": np.full((48, 64, 3), self.count % 255, np.uint8)},
            }

        def close(self):
            pass

    class PolicyClient:
        def __init__(self, **kwargs):
            self.count = 0

        def get_action(self, observation):
            self.count += 1
            number = self.count
            record("policy_started", number=number)
            (args.output / f"policy-started-{number}").touch()
            if number > 1:
                deadline = time.monotonic() + 8
                while not (args.output / f"release-policy-{number}").exists():
                    if time.monotonic() >= deadline:
                        raise TimeoutError("Scripted policy release deadline")
                    time.sleep(0.01)
            action = {
                "motion_token": np.full((1, 40, 64), 0.25, np.float32),
                "left_hand_joints": np.zeros((1, 40, 7), np.float32),
                "right_hand_joints": np.zeros((1, 40, 7), np.float32),
            }
            record("policy_returned", number=number)
            return action, {}

        def close(self):
            pass

    class TraceQueue(queue.Queue):
        def put_nowait(self, item):
            super().put_nowait(item)
            if isinstance(item, tuple) and len(item) == 3:
                record("result_queued", epoch=item[2], valid=item[0] is not None)

        def get_nowait(self):
            item = super().get_nowait()
            if isinstance(item, tuple) and len(item) == 3:
                record("result_consumed", epoch=item[2], valid=item[0] is not None,
                       captured_age_s=time.monotonic()-item[1],
                       phase=None if control is None else control.phase)
            return item

    class TraceControl(harness_control.HarnessControl):
        def __init__(self, *positional, **keyword):
            nonlocal control
            super().__init__(*positional, **keyword)
            control = self
            self.last_trace_at = 0.0
            self.last_trace_phase = None

        def tick(self, now, feedback, received_at):
            super().tick(now, feedback, received_at)
            if self.phase != self.last_trace_phase or now-self.last_trace_at >= 0.1:
                record("status", **self.status(now))
                self.last_trace_at, self.last_trace_phase = now, self.phase

        def accept_policy_result(self, epoch, captured_at, action, now):
            accepted = super().accept_policy_result(epoch, captured_at, action, now)
            record("policy_acceptance", epoch=epoch, accepted=accepted, phase=self.phase)
            return accepted

    # Replace external data sources only. Native lifecycle and guards stay real.
    module = ModuleType("gr00t.policy.server_client")
    module.PolicyClient = PolicyClient
    sys.modules["gr00t.policy.server_client"] = module
    loop.ComposedCameraClientSensor = Camera
    loop.queue = SimpleNamespace(Queue=TraceQueue, Empty=queue.Empty, Full=queue.Full)
    harness_control.HarnessControl = TraceControl

    def forbidden_latent(*positional, **keyword):
        record("unexpected_latent_publication_attempt")
        raise AssertionError("Scripted stale results reached the action publisher")

    loop.pack_latent_action_message = forbidden_latent
    config = loop.InferenceConfig(
        camera_host="127.0.0.1", camera_port=11555,
        state_zmq_host="127.0.0.1", state_zmq_port=11557,
        action_zmq_host="127.0.0.1", action_zmq_port=11556,
        keyboard_zmq_host="127.0.0.1", keyboard_zmq_port=11580,
        harness_endpoint=args.endpoint, harness_profile=str(args.profile),
    )
    record("configuration", dds_interface="lo", scripted_camera=True, scripted_policy=True,
           native_main_loop=True, robot_connection=False)
    loop.main(config)


if __name__ == "__main__":
    main()
