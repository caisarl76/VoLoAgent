"""Read-only probe for a "bottle taken" signal: right-arm motor torques (rt/lowstate) and the Dex3 hand pads.

Usage (PC2, a Python with unitree_sdk2py, e.g. ~/dimos/.venv/bin/python):
    g1_dex3_pressure_probe.py OUT.npz [--hand right] [--iface enP8p1s0] [--labels idle,bottle,taken]

Run it in its own terminal beside a loop session. It subscribes on DDS domain 0 to rt/lowstate (the arm's 7 motors:
tau_est, q, dq) and rt/dex3/<hand>/state (each pad's 12 pressure values, 'lost' flags, the 7 finger positions). It never
publishes. Marking needs only Enter: each empty Enter marks the next label, the first once and then the rest cycling
(idle, bottle, taken, bottle, taken, ...). Typing a word + Enter marks that word instead. A live line shows the arm torques
and the pad total. `q` + Enter, Ctrl-C or `kill PID` stops it and saves OUT.npz, then prints per-label means.
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
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import HandState_, LowState_

ARM = {"right": range(22, 29), "left": range(15, 22)}  # G1 motor order: shoulder pitch/roll/yaw, elbow, wrist roll/pitch/yaw
JOINTS = ("sh_pitch", "sh_roll", "sh_yaw", "elbow", "wr_roll", "wr_pitch", "wr_yaw")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("out", type=Path)
    p.add_argument("--hand", choices=("right", "left"), default="right")
    p.add_argument("--iface", default="enP8p1s0")
    p.add_argument("--labels", default="idle,bottle,taken", help="Enter marks these in turn: the first once, then the rest cycle")
    a = p.parse_args()
    labels = [s.strip() for s in a.labels.split(",") if s.strip()]
    arm_idx = list(ARM[a.hand])

    arm, hand, lock = [], [], threading.Lock()

    def on_low(msg):
        ms = msg.motor_state
        with lock:
            arm.append((time.time(), [ms[i].tau_est for i in arm_idx], [ms[i].q for i in arm_idx], [ms[i].dq for i in arm_idx]))

    def on_hand(msg):
        pads = msg.press_sensor_state
        with lock:
            hand.append((time.time(), [list(s.pressure) for s in pads], [s.lost for s in pads], [m.q for m in msg.motor_state]))

    ChannelFactoryInitialize(0, a.iface)
    subs = [ChannelSubscriber("rt/lowstate", LowState_), ChannelSubscriber(f"rt/dex3/{a.hand}/state", HandState_)]
    subs[0].Init(on_low, 10)
    subs[1].Init(on_hand, 10)

    stop, marks = [], []
    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(sig, lambda signum, frame: stop.append(signum))

    def next_label():
        k = len([m for m in marks if m[2]])  # Enter-only marks so far
        return labels[k] if k < len(labels) else labels[1 + (k - 1) % (len(labels) - 1)] if len(labels) > 1 else labels[0]

    print(f"pid {os.getpid()}: recording {a.hand} arm torques + Dex3 pads -> {a.out}", flush=True)
    print(f"Enter marks the next label (now: '{next_label()}'); a word + Enter marks that word; q + Enter stops.", flush=True)
    last_print = 0.0
    while not stop:
        if select.select([sys.stdin], [], [], 0.1)[0]:
            line = sys.stdin.readline()
            if not line or line.strip() == "q":
                break
            word = line.strip()
            label = word or next_label()
            marks.append((time.time(), label, not word))
            print(f"  >>> mark '{label}'   (next Enter: '{next_label()}')", flush=True)
        if time.time() - last_print > 0.5:
            with lock:
                la, lh = (arm[-1] if arm else None), (hand[-1] if hand else None)
            if la is None:
                print("  waiting for rt/lowstate ...", flush=True)
            else:
                tau = " ".join(f"{n} {v:+6.2f}" for n, v in zip(JOINTS, la[1]))
                pad = f" | pads {np.sum(lh[1]):.0f}" if lh else " | no hand state"
                print(f"  tau_est N*m: {tau}{pad}", flush=True)
            last_print = time.time()

    with lock:
        arm_rows, hand_rows = list(arm), list(hand)
    if not arm_rows:
        sys.exit("no rt/lowstate received: is the robot on and the interface right?")
    out = {"arm_t": np.array([r[0] for r in arm_rows]), "arm_tau": np.array([r[1] for r in arm_rows], np.float32),
           "arm_q": np.array([r[2] for r in arm_rows], np.float32), "arm_dq": np.array([r[3] for r in arm_rows], np.float32),
           "arm_joints": np.array(JOINTS), "mark_t": np.array([m[0] for m in marks], np.float64),
           "mark_label": np.array([m[1] for m in marks], dtype=str)}
    if hand_rows:
        out |= {"hand_t": np.array([r[0] for r in hand_rows]), "pad_pressure": np.array([r[1] for r in hand_rows], np.float32),
                "pad_lost": np.array([r[2] for r in hand_rows], np.int64), "hand_q": np.array([r[3] for r in hand_rows], np.float32)}
    tmp = a.out.with_name(a.out.name + ".tmp")
    with open(tmp, "wb") as f:
        np.savez_compressed(f, **out)
    os.replace(tmp, a.out)
    t = out["arm_t"]
    print(f"saved {a.out}: {len(t)} arm msgs ({(len(t) - 1) / max(t[-1] - t[0], 1e-9):.0f} Hz), {len(hand_rows)} hand msgs",
          flush=True)
    edges = [m[0] for m in marks] + [t[-1] + 1]
    for (t0, label, _), t1 in zip(marks, edges[1:]):
        sel = (t >= t0) & (t < t1)
        if sel.any():
            tau = " ".join(f"{v:+6.2f}" for v in out["arm_tau"][sel].mean(axis=0))
            print(f"  {label:12s} {sel.sum():6d} msgs  mean tau_est [{tau}]", flush=True)


if __name__ == "__main__":
    main()
