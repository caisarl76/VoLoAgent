"""Production native harness (run_vla_inference + LoopController) on the simulator, with a native-side message log.

SIMULATION FIXTURE ONLY. Adapted from g1_handover_20261021_s10_loop_sim/loop_native_worker.py:
- no pre-pose and no ready-return tracer: the autonomous loop never uses the StandingReset ready return;
- the Isaac-GR00T client module is stubbed only because run_vla_inference imports it unconditionally;
- message log (native-messages.jsonl): every message the process sends on the action socket (pose, planner,
  command) with its monotonic send time, and every LoopController.tick with its phase, stage and label. Nothing
  here changes what is sent or when; the log is written by a background thread every 0.25 s.
"""
import argparse
import base64
import collections
import json
from pathlib import Path
import sys
import threading
import time
from types import ModuleType

import zmq

from gear_sonic.scripts import run_vla_inference as loop  # noqa: E402
from gear_sonic.utils.inference import loop_control  # noqa: E402

TOPICS = (b"pose", b"planner", b"command")
HEADER_SIZE = 1280


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--endpoint", required=True)
    args = parser.parse_args()
    records = collections.deque()

    def writer():
        with (args.output / "native-messages.jsonl").open("a") as stream:
            while True:
                time.sleep(0.25)
                while records:
                    stream.write(json.dumps(records.popleft()) + "\n")
                stream.flush()

    threading.Thread(target=writer, daemon=True).start()
    send = zmq.Socket.send

    def logged_send(self, data, *a, **kw):
        if isinstance(data, (bytes, bytearray, memoryview)):
            data = bytes(data)
            topic = next((t for t in TOPICS if data.startswith(t)), None)
            if topic is not None:
                record = dict(k="send", t=time.monotonic(), topic=topic.decode())
                payload = data[len(topic) + HEADER_SIZE:]
                if topic == b"pose":
                    record["b64"] = base64.b64encode(payload).decode()
                elif topic == b"command":
                    record["start_stop_planner"] = list(payload[:3])
                records.append(record)
        return send(self, data, *a, **kw)

    zmq.Socket.send = logged_send
    tick = loop_control.LoopController.tick

    def logged_tick(self, now, *a, **kw):
        token, hands, label = tick(self, now, *a, **kw)
        records.append(dict(k="tick", now=now, t=time.monotonic(), phase=self.phase, stage=self.stage,
                            label=label, trial=self.trial_no))
        return token, hands, label

    loop_control.LoopController.tick = logged_tick

    class UnusedIsaacClient:
        def __init__(self, **kwargs):
            raise RuntimeError("this fixture serves a LeRobot policy; the Isaac-GR00T client must not be used")

    module = ModuleType("gr00t.policy.server_client")
    module.PolicyClient = UnusedIsaacClient
    sys.modules["gr00t.policy.server_client"] = module
    config = loop.InferenceConfig(camera_host="127.0.0.1", camera_port=11555,
        state_zmq_host="127.0.0.1", state_zmq_port=11557,
        action_zmq_host="127.0.0.1", action_zmq_port=11556,
        keyboard_zmq_host="127.0.0.1", keyboard_zmq_port=11580,
        harness_endpoint=args.endpoint, harness_profile=str(args.profile))
    loop.main(config)


if __name__ == "__main__":
    main()
