"""Read-only probe of the Dex3 hand pressure pads: does a bottle on the open palm show, and does its removal?

Usage (PC2, a Python with unitree_sdk2py, e.g. ~/dimos/.venv/bin/python):
    g1_dex3_pressure_probe.py OUT.npz [--hand right] [--iface enP8p1s0]

Subscribes to rt/dex3/<hand>/state (HandState_) on DDS domain 0 and records every message: receive time, each pad's 12
pressure and temperature values, the pad 'lost' flags and the 7 motor positions. It never publishes. While it runs, type a
label and Enter to mark what is happening (e.g. `empty`, `bottle on`, `taken`); a live line shows each pad's pressure sum.
`q` + Enter, Ctrl-C or `kill PID` stops it and saves OUT.npz, then prints the mean total pressure per labelled segment.
"""
import argparse
import os
import select
import signal
import sys
import threading
import time
from pathlib import Path

import numpy as np
from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import HandState_


def main():
    p = argparse.ArgumentParser()
    p.add_argument("out", type=Path)
    p.add_argument("--hand", choices=("right", "left"), default="right")
    p.add_argument("--iface", default="enP8p1s0")
    a = p.parse_args()

    rows, lock = [], threading.Lock()

    def on_state(msg):
        pads = msg.press_sensor_state
        with lock:
            rows.append((time.time(),
                         np.array([list(s.pressure) for s in pads], np.float32),
                         np.array([list(s.temperature) for s in pads], np.float32),
                         np.array([s.lost for s in pads], np.int64),
                         np.array([m.q for m in msg.motor_state], np.float32)))

    ChannelFactoryInitialize(0, a.iface)
    sub = ChannelSubscriber(f"rt/dex3/{a.hand}/state", HandState_)
    sub.Init(on_state, 10)

    stop, marks = [], []
    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(sig, lambda signum, frame: stop.append(signum))
    print(f"pid {os.getpid()}: recording rt/dex3/{a.hand}/state -> {a.out}. Type a label + Enter to mark; q + Enter "
          "stops.", flush=True)
    last_print = 0.0
    while not stop:
        if select.select([sys.stdin], [], [], 0.1)[0]:
            line = sys.stdin.readline()
            if not line or line.strip() == "q":
                break
            if line.strip():
                marks.append((time.time(), line.strip()))
                print(f"  mark '{line.strip()}'", flush=True)
        if time.time() - last_print > 0.5:
            with lock:
                latest, n = (rows[-1], len(rows)) if rows else (None, 0)
            if latest is None:
                print("  waiting for hand state ...", flush=True)
            else:
                sums = latest[1].sum(axis=1)
                print(f"  {n} msgs | pad sums {np.array2string(sums, precision=0, max_line_width=200)} | "
                      f"total {sums.sum():.0f} | lost {latest[3].tolist()}", flush=True)
            last_print = time.time()

    with lock:
        data = list(rows)
    if not data:
        sys.exit("no hand state received: is the robot on and the interface right?")
    t = np.array([r[0] for r in data])
    out = {"t": t, "pressure": np.stack([r[1] for r in data]), "temperature": np.stack([r[2] for r in data]),
           "lost": np.stack([r[3] for r in data]), "motor_q": np.stack([r[4] for r in data]),
           "mark_t": np.array([m[0] for m in marks], np.float64), "mark_label": np.array([m[1] for m in marks], dtype=str)}
    tmp = a.out.with_name(a.out.name + ".tmp")
    with open(tmp, "wb") as f:
        np.savez_compressed(f, **out)
    os.replace(tmp, a.out)
    rate = (len(t) - 1) / (t[-1] - t[0]) if len(t) > 1 else 0.0
    print(f"saved {a.out}: {len(t)} messages, {rate:.0f} Hz, pads {out['pressure'].shape[1]} x 12", flush=True)
    total = out["pressure"].sum(axis=(1, 2))
    edges = list(out["mark_t"]) + [t[-1] + 1]
    for (t0, label), t1 in zip(marks, edges[1:]):
        seg = total[(t >= t0) & (t < t1)]
        if len(seg):
            print(f"  {label:20s} {len(seg):5d} msgs  total pressure mean {seg.mean():10.1f}  min {seg.min():10.1f}  "
                  f"max {seg.max():10.1f}", flush=True)


if __name__ == "__main__":
    main()
