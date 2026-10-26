"""Parity of the native LeRobot policy path with lerobot's sonic_policy_streamer.py, from a real s10 robot session.

Usage (the lerobot venv, which has the streamer's dependencies):
    ~/work/lerobot/.venv/bin/python parity_check.py SESSION_RAW_DIR OUT.json [--trial N]

No GPU or robot is needed. If the native client sends byte-identical requests and plays the returned chunks
identically, the served model's behaviour is the same by construction. This script compares, against the
streamer's own code (not a copy):

1. state: stream_backends.observation_state vs lerobot_state on random g1_debug messages, in both Dex3 right-hand
   orders;
2. request: the raw msgpack bytes that the streamer's RemotePolicy.chunk and LeRobotSonicPolicyClient.get_action
   send for the session's real head-camera frames and 28D states (a capturing fake server answers both);
3. output: the 50 Hz tokens and hands from the streamer's SonicBackend (ChunkResampler cross-fade and send()'s
   slew limit) vs the native TokenStream. Both get the same chunk sequence at the session's own replan and arrival
   ticks, with the s10 settings (0.3 s blend, 0.05 token step). The chunks are 40-row windows of the session's
   logged output: the raw chunks were not logged, and any realistic chunk sequence exercises the same code.
"""
import argparse
import json
from pathlib import Path
import sys
import threading
from types import SimpleNamespace

import cv2
import msgpack
import numpy as np
import zmq

NATIVE = Path("/home/jihun/work/VoLoAgent/.worktrees/g1-harness-native")
STREAMER = Path.home() / "work/lerobot/examples/g1_dex3_training"
sys.path[:0] = [str(NATIVE), str(STREAMER)]
import sonic_policy_streamer as streamer  # noqa: E402
import stream_backends as backends  # noqa: E402

from gear_sonic.utils.inference.lerobot_sonic_policy import (  # noqa: E402
    RIGHT_SWAP,
    LeRobotSonicPolicyClient,
    TokenStream,
    lerobot_state,
)

KEY = "observation.images.egocentric"
TASK = "pick drink bottle from the table and handover"
BLEND_S, MAX_TOKEN_STEP, FPS, ROWS = 0.3, 0.05, 50, 40


class CapturingServer:
    """sonic_policy_server.py's protocol; keeps every request's raw bytes."""

    def __init__(self, chunk):
        self.chunk, self.raw = chunk, []
        self.sock = zmq.Context.instance().socket(zmq.REP)
        self.sock.bind("tcp://127.0.0.1:*")
        self.addr = self.sock.getsockopt(zmq.LAST_ENDPOINT).decode()
        threading.Thread(target=self.serve, daemon=True).start()

    def serve(self):
        while self.sock.poll(3000):
            raw = self.sock.recv()
            self.raw.append(raw)
            request = msgpack.unpackb(raw, raw=False)
            reply = (dict(image_keys=[KEY], n_obs=1, shapes={KEY: [3, 480, 640]}, model="capture")
                     if request["op"] == "info"
                     else dict(chunk=self.chunk.tobytes(), shape=list(self.chunk.shape), latency_s=0.0))
            self.sock.send(msgpack.packb(reply, use_bin_type=True))


def state_parity(rng):
    worst = {}
    for order, permutation in (("dataset", np.arange(7)), ("swap", RIGHT_SWAP)):
        backends.RIGHT_ORDER[:] = permutation  # The streamer's --dex3-right-order switch.
        diffs = []
        for _ in range(200):
            msg = dict(body_q=rng.normal(size=29), left_hand_q=rng.normal(size=7), right_hand_q=rng.normal(size=7))
            diffs.append(np.max(np.abs(backends.observation_state(msg) - lerobot_state(msg, permutation))))
        worst[order] = float(max(diffs))
    backends.RIGHT_ORDER[:] = np.arange(7)
    return worst


def request_parity(log, frames):
    jpeg, offsets, t_recv = frames["jpeg"], frames["offsets"], frames["t_recv"]
    picks = np.linspace(0, len(t_recv) - 1, 20).astype(int)
    server = CapturingServer(np.zeros((ROWS, 78), np.float32))
    remote = streamer.RemotePolicy(server.addr, 3.0)
    host, port = server.addr.removeprefix("tcp://").rsplit(":", 1)
    client = LeRobotSonicPolicyClient(host, int(port), 3000)
    identical = 0
    for i in picks:
        image = cv2.imdecode(jpeg[offsets[i]:offsets[i + 1]], cv2.IMREAD_COLOR)  # As received; no conversion.
        tick = int(np.clip(np.searchsorted(log["wall"], t_recv[i]), 0, len(log["wall"]) - 1))
        state = log["state"][tick]
        before = len(server.raw)
        remote.chunk([state], [{KEY: image}], TASK)
        client.get_action({"lerobot": dict(state=state, image=image, task=TASK)})
        streamer_bytes, native_bytes = server.raw[before:before + 2]
        identical += streamer_bytes == native_bytes
    client.close()
    return dict(frames=len(picks), byte_identical=int(identical), image_shape=list(image.shape))


def output_parity(log):
    episode = np.flatnonzero(log["phase"] == "episode")
    t_ep, chunk_t0 = log["t_ep"][episode], log["chunk_t0"][episode]
    rows_all = np.hstack([log["token"][episode], log["hands"][episode]]).astype(np.float32)
    arrivals = np.r_[0, np.flatnonzero(np.diff(chunk_t0) != 0) + 1]

    def window(t0):
        start = int(np.searchsorted(t_ep, t0 - 1e-9))
        rows = rows_all[start:start + ROWS]
        return np.vstack([rows, np.repeat(rows[-1:], ROWS - len(rows), 0)])

    start_token = rows_all[0, :64] - 0.3  # A planner token the slew limit must climb from.
    captured = []
    args = SimpleNamespace(max_token_step=MAX_TOKEN_STEP, action_space="sonic78")
    backend = backends.SonicBackend(args, SimpleNamespace(send=lambda _: None), None,
                                    lambda token, hands, *rest: captured.append(np.r_[token, hands]), FPS, None)
    backend.last_sent = start_token.copy()
    native = TokenStream(FPS, BLEND_S, MAX_TOKEN_STEP)
    native.reset(start_token=start_token)
    native_out, next_arrival = [], 0
    for i, t in enumerate(t_ep):
        if next_arrival < len(arrivals) and i == arrivals[next_arrival]:
            chunk = window(chunk_t0[i])
            backend.set_chunk(chunk, None, chunk_t0[i], BLEND_S, t)
            native.set_chunk(chunk, chunk_t0[i], t)
            next_arrival += 1
        backend.episode_tick(t, i, chunk_t0[i])
        token, hands = native.at(t)
        native_out.append(np.r_[token, hands])
    diff = np.abs(np.array(captured) - np.array(native_out))
    return dict(ticks=len(t_ep), chunks=len(arrivals), max_abs_token_diff=float(diff[:, :64].max()),
                max_abs_hand_diff=float(diff[:, 64:].max()), streamer_slewed_ticks=int(backend.slewed))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("session", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--trial", type=int, default=0)
    a = parser.parse_args()
    log = np.load(a.session / f"trial_{a.trial:03d}.npz", allow_pickle=True)
    frames = np.load(a.session / f"trial_{a.trial:03d}_frames.npz", allow_pickle=True)
    result = dict(session=str(a.session), trial=a.trial, model=str(log["model"]), task=TASK,
                  settings=dict(chunk_blend_s=BLEND_S, max_token_step=MAX_TOKEN_STEP, fps=FPS),
                  state=state_parity(np.random.default_rng(0)), request=request_parity(log, frames),
                  output=output_parity(log))
    a.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
