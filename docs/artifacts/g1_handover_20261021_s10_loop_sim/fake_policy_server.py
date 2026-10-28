"""sonic_policy_server.py's protocol with a recorded cycle instead of a model: a plumbing check for run_s10_loop.py.

Usage: fake_policy_server.py --port 5590 --output DIR

Answers like sonic_policy_server.py (ZMQ REP, msgpack): "info" reports s10's input shapes; "chunk" returns 40 rows
(token 64, Dex3 hands 14) of the recorded handover cycle in g1_handover_20261019_loop_sim/replay-cycle.npz, from the
row matching the time since the episode's first request. A request after 2 s without any starts a new episode, so
the prewarm request and each manipulation start from row 0. Every 10th request's state, task and decoded head image
are saved to OUTPUT, to show what the native loop sends.
"""
import argparse
import json
from pathlib import Path
import time

import cv2
import msgpack
import numpy as np
import zmq

ROOT = Path(__file__).parent
KEY = "observation.images.egocentric"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    a = parser.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    replay = np.load(ROOT.parent / "g1_handover_20261019_loop_sim/replay-cycle.npz")
    rows = np.hstack([replay["motion_token"], replay["left_hand_joints"], replay["right_hand_joints"]]).astype(np.float32)
    sock = zmq.Context().socket(zmq.REP)
    sock.bind(f"tcp://127.0.0.1:{a.port}")
    print(f"[fake server] ready on port {a.port}: {len(rows)} recorded rows", flush=True)
    episode_start, last, count = None, -1e9, 0
    log = (a.output / "requests.jsonl").open("a")
    while True:
        request = msgpack.unpackb(sock.recv(), raw=False)
        now = time.monotonic()
        if request["op"] == "info":
            reply = dict(image_keys=[KEY], n_obs=1, shapes={KEY: [3, 480, 640]}, model="fake: replay-cycle.npz")
        else:
            if now - last > 2.0:
                episode_start = now
            last, count = now, count + 1
            start = min(int((now - episode_start) * 50), len(rows) - 1)
            chunk = rows[np.minimum(np.arange(start, start + 40), len(rows) - 1)]
            image = cv2.imdecode(np.frombuffer(request["images"][0][KEY], np.uint8), cv2.IMREAD_COLOR)
            log.write(json.dumps(dict(at=now, request=count, row=start, task=request["task"],
                                      state=request["states"][0], image_shape=list(image.shape))) + "\n")
            log.flush()
            if count % 10 == 1:
                cv2.imwrite(str(a.output / f"request-{count:04d}.jpg"), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            reply = dict(chunk=chunk.tobytes(), shape=list(chunk.shape), latency_s=0.0)
        sock.send(msgpack.packb(reply, use_bin_type=True))


if __name__ == "__main__":
    main()
