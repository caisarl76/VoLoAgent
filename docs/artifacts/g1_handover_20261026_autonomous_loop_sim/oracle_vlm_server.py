"""Scene-truth oracle behind the OpenAI-compatible chat API: answers the two re-arm prompts from the simulator.

SIMULATION FIXTURE ONLY. The simulated bottle is an opaque white cylinder, not the clear PET bottle the re-arm
prompts were validated on, so the primary run takes its re-arm decisions from here. The `loop` command still runs
RearmMonitor unchanged, with --vlm-base-url pointing at this server.

- upright_check prompt (`bottle_upright_on_case`): upright = the bottle stands on the case (fixture state on_box);
  hand_at_bottle = a hand holds it (in_hand or on_palm).
- object_on_case prompt: true when the bottle is anywhere on the case top (standing, or dropped and lying there).
The answer is the state at request time (bottle-state.json, 20 Hz), not at the frame's capture; the frame is at
most camera_max_age_s (0.5 s) old. The image is ignored. Every answer is logged to oracle.jsonl.
"""
import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import threading
import time
import uuid

from scene_s10 import CASE_M


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8302)
    parser.add_argument("--box-front-m", type=float, required=True)
    parser.add_argument("--box-height-m", type=float, required=True)
    args = parser.parse_args()
    lock = threading.Lock()
    depth, width = CASE_M

    def on_case(truth):
        x, y, z = truth["bottle_position"]
        return (not truth["box_away"] and args.box_front_m <= x <= args.box_front_m + depth
                and abs(y) <= width / 2 and args.box_height_m - 0.01 <= z <= args.box_height_m + 0.25)

    def answer(text):
        truth = json.loads((args.output / "bottle-state.json").read_text())
        bottle = truth["bottle"]
        if "object_on_case" in text:
            reply = {"object_on_case": bottle in ("on_box", "dropped", "fallen") and on_case(truth)}
        elif "bottle_upright_on_case" in text:
            reply = {"bottle_upright_on_case": bottle == "on_box" and not truth["box_away"],
                     "hand_at_bottle": bottle in ("in_hand", "on_palm")}
        else:
            raise ValueError("unknown prompt")
        with lock, (args.output / "oracle.jsonl").open("a") as stream:
            stream.write(json.dumps(dict(at=time.monotonic(), bottle=bottle, sim_time=truth["sim_time"],
                                         position=truth["bottle_position"], reply=reply)) + "\n")
        return json.dumps(reply)

    class Handler(BaseHTTPRequestHandler):
        def _send(self, code, body):
            data = json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_POST(self):
            try:
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
                text = " ".join(part.get("text", "") for m in body["messages"] if m.get("role") == "user"
                                for part in (m["content"] if isinstance(m["content"], list) else [])
                                if part.get("type") == "text")
                reply = answer(text)
            except (ValueError, KeyError, TypeError, OSError) as exc:
                return self._send(400, {"error": {"message": f"{type(exc).__name__}: {exc}"}})
            self._send(200, {"id": f"chatcmpl-{uuid.uuid4().hex}", "object": "chat.completion",
                             "created": int(time.time()), "model": body.get("model") or "sim-oracle",
                             "choices": [{"index": 0, "finish_reason": "stop",
                                          "message": {"role": "assistant", "content": reply}}]})

        def log_message(self, *args):
            pass

    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
