"""Record the deploy's g1_debug telemetry on the robot PC for a supervised stage (read-only).

Usage (PC2 harness venv, PYTHONPATH=the harness folder):
    record_telemetry.py OUT.npz [--seconds 40]

Subscribes to g1_debug on localhost:5557 and stores every new message until --seconds pass or Ctrl-C:
receive time, index, measured joints (motor order), the deploy's current joint targets, Dex3 hands, base orientation,
planner_reference_active and the hold flag. It sends nothing.
"""
import argparse
import time

import numpy as np

from gear_sonic.utils.data_collection.zmq_state_subscriber import ZMQStateSubscriber

FIELDS = ("body_q_measured_motor", "body_q_target", "left_hand_q", "right_hand_q", "base_quat",
          "planner_reference_active", "harness_planner_hold_enabled")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("out")
    parser.add_argument("--seconds", type=float, default=40.0)
    a = parser.parse_args()
    subscriber = ZMQStateSubscriber(host="localhost", port=5557)
    rows, last, end = {k: [] for k in ("t", "index", *FIELDS)}, None, time.time() + a.seconds
    try:
        while time.time() < end:
            msg = subscriber.get_msg()
            if msg is None or msg.get("index") == last:
                time.sleep(0.002)
                continue
            last = msg.get("index")
            rows["t"].append(time.time())
            rows["index"].append(last)
            for key in FIELDS:
                value = np.asarray(msg.get(key, []), np.float64).reshape(-1)
                rows[key].append(value)
    except KeyboardInterrupt:
        pass
    np.savez_compressed(a.out, **{k: np.array(v, dtype=object if k in FIELDS else np.float64) for k, v in rows.items()})
    print(f"recorded {len(rows['t'])} messages over {rows['t'][-1] - rows['t'][0]:.1f} s" if rows["t"] else "no messages")


if __name__ == "__main__":
    main()
