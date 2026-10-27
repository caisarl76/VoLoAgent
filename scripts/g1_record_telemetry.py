"""Record the deploy's g1_debug telemetry on the robot PC during a supervised session (read-only).

Usage (PC2, harness venv, PYTHONPATH=the native loop folder):
    g1_record_telemetry.py OUT.npz [--seconds 1800]

Subscribes to g1_debug on localhost:5557 and stores every new message until --seconds pass or it is stopped: receive
time, index, measured joints (motor order), the deploy's joint targets, Dex3 hands, base orientation,
planner_reference_active and the hold flag. It sends nothing.

Stopping: SIGTERM (`kill PID`) or SIGINT (Ctrl-C) ends the recording and saves it. Both are handled explicitly, because a
recorder started with `nohup ... &` from a non-interactive shell inherits SIGINT/SIGHUP as ignored. The file is written
to OUT.tmp and renamed, and a second recorder for the same OUT refuses to start (lock file OUT.lock).
"""
import argparse
import fcntl
import os
import signal
import sys
import time
from pathlib import Path

import numpy as np

from gear_sonic.utils.data_collection.zmq_state_subscriber import ZMQStateSubscriber

FIELDS = ("body_q_measured_motor", "body_q_target", "left_hand_q", "right_hand_q", "base_quat",
          "planner_reference_active", "harness_planner_hold_enabled")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("out", type=Path)
    parser.add_argument("--seconds", type=float, default=1800.0)
    a = parser.parse_args()
    lock = open(a.out.with_name(a.out.name + ".lock"), "w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        sys.exit(f"another recorder is already writing {a.out}; stop it (kill PID) or choose another file name")
    lock.write(str(os.getpid()))
    lock.flush()

    stop = []
    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(sig, lambda signum, frame: stop.append(signum))
    print(f"recording {a.out} for up to {a.seconds:g} s (pid {os.getpid()}; stop with: kill {os.getpid()})", flush=True)

    subscriber = ZMQStateSubscriber(host="localhost", port=5557)
    rows, last, end = {k: [] for k in ("t", "index", *FIELDS)}, None, time.time() + a.seconds
    while time.time() < end and not stop:
        msg = subscriber.get_msg()
        if msg is None or msg.get("index") == last:
            time.sleep(0.002)
            continue
        last = msg.get("index")
        rows["t"].append(time.time())
        rows["index"].append(last)
        for key in FIELDS:
            rows[key].append(np.asarray(msg.get(key, []), np.float64).reshape(-1))

    tmp = a.out.with_name(a.out.name + ".tmp")
    with open(tmp, "wb") as f:
        np.savez_compressed(f, **{k: np.array(v, dtype=object if k in FIELDS else np.float64) for k, v in rows.items()})
    os.replace(tmp, a.out)
    why = f"signal {stop[0]}" if stop else "time limit"
    span = f"{rows['t'][-1] - rows['t'][0]:.1f} s" if rows["t"] else "no messages"
    print(f"saved {a.out}: {len(rows['t'])} messages over {span} ({why})", flush=True)


if __name__ == "__main__":
    main()
