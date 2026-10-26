"""Local OpenAI-compatible server for Qwen3-VL-2B: the G1 re-arm monitor's VLM (one RTX 3060, bf16).

Usage (lerobot venv, transformers 5.x):
  CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 python scripts/g1_local_vlm_server.py --model SNAPSHOT --port 8301
Request handling follows docs/artifacts/g1_vlm_upright_check_20261026/upright_check.py.
"""
import argparse
import base64
import binascii
import io
import json
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from PIL import Image

MODEL_NAME = "qwen3-vl-2b"
MAX_NEW_TOKENS = 60


class BadRequest(ValueError):
    pass


def parse_messages(messages):
    """-> (images, text, system): image parts decoded to RGB, empty system dropped."""
    images, texts, system = [], [], []
    for message in messages:
        content = message.get("content")
        if isinstance(content, str):
            content = [{"type": "text", "text": content}]
        if message.get("role") == "system":
            system += [p.get("text", "") for p in content or [] if p.get("type") == "text"]
            continue
        for part in content or []:
            if part.get("type") == "text":
                texts.append(part.get("text", ""))
            elif part.get("type") == "image_url":
                url = (part.get("image_url") or {}).get("url", "")
                if not url.startswith("data:image/") or ";base64," not in url:
                    raise BadRequest("image_url must be a base64 data URL")
                try:
                    raw = base64.b64decode(url.split(";base64,", 1)[1], validate=True)
                    images.append(Image.open(io.BytesIO(raw)).convert("RGB"))
                except (binascii.Error, OSError):
                    raise BadRequest("image_url is not a decodable image")
    if not images:
        raise BadRequest("request has no image")
    return images, "\n".join(texts), "\n".join(s for s in system if s)


def build_app(generate):
    """generate(images, text, system="") -> str. One request at a time (lock around generate)."""
    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def _send(self, code, body):
            data = json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path.rstrip("/") == "/v1/models":
                self._send(200, {"object": "list", "data": [{"id": MODEL_NAME, "object": "model"}]})
            else:
                self._send(404, {"error": {"message": "not found"}})

        def do_POST(self):
            if self.path.rstrip("/") != "/v1/chat/completions":
                return self._send(404, {"error": {"message": "not found"}})
            try:
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
                images, text, system = parse_messages(body["messages"])
            except (BadRequest, ValueError, KeyError, TypeError, AttributeError) as exc:
                return self._send(400, {"error": {"message": str(exc) or "bad request", "type": "invalid_request_error"}})
            try:
                with lock:
                    reply = generate(images, text, system)
            except Exception as exc:
                return self._send(500, {"error": {"message": f"{type(exc).__name__}: {exc}"}})
            self._send(200, {
                "id": f"chatcmpl-{uuid.uuid4().hex}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": body.get("model") or MODEL_NAME,
                "choices": [{"index": 0, "message": {"role": "assistant", "content": reply}, "finish_reason": "stop"}],
            })

        def log_message(self, *args):
            pass

    return Handler


def free_gpu_gb(device):
    import torch

    return torch.cuda.mem_get_info(device)[0] / 1e9


def load_generate(model_path, device):
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    processor = AutoProcessor.from_pretrained(model_path)
    model = AutoModelForImageTextToText.from_pretrained(model_path, dtype=torch.bfloat16, device_map=device).eval()

    def generate(images, text, system=""):
        # ponytail: images then text, as the monitor sends and upright_check validated; part order is not preserved.
        messages = [{"role": "user", "content": [{"type": "image", "image": i} for i in images] + [{"type": "text", "text": text}]}]
        if system:
            messages.insert(0, {"role": "system", "content": [{"type": "text", "text": system}]})
        inputs = processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
        ).to(device)
        with torch.inference_mode():
            ids = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
        return processor.batch_decode(ids[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0].strip()

    return generate


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model", required=True)
    p.add_argument("--host", default="127.0.0.1", help="0.0.0.0 to serve the PC2 orchestrator over the lab LAN")
    p.add_argument("--port", type=int, default=8301)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--min-free-gb", type=float, default=4.4)
    args = p.parse_args(argv)
    free = free_gpu_gb(args.device)
    if free < args.min_free_gb:
        sys.exit(f"{args.device} has {free:.1f} GB free, below --min-free-gb {args.min_free_gb}; not loading the model")
    generate = load_generate(args.model, args.device)
    httpd = ThreadingHTTPServer((args.host, args.port), build_app(generate))
    print(f"serving {MODEL_NAME} on http://{args.host}:{args.port}/v1", flush=True)
    httpd.serve_forever()


if __name__ == "__main__":
    main()
