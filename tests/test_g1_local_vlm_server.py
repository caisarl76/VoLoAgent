import base64
import importlib.util
import io
import json
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest
from openai import OpenAI
from PIL import Image

_spec = importlib.util.spec_from_file_location(
    "g1_local_vlm_server", Path(__file__).parents[1] / "scripts" / "g1_local_vlm_server.py"
)
server = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(server)


def data_url(size=(8, 6)):
    buf = io.BytesIO()
    Image.new("RGB", size).save(buf, format="JPEG")
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


class Serve:
    def __init__(self, generate):
        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), server.build_app(generate))
        self.url = f"http://127.0.0.1:{self.httpd.server_port}/v1"

    def __enter__(self):
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()
        return self

    def __exit__(self, *exc):
        self.httpd.shutdown()
        self.httpd.server_close()


def post(url, body):
    req = urllib.request.Request(
        url + "/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"}
    )
    return urllib.request.urlopen(req)


def test_chat_completion_round_trip_with_fake_generate():
    calls = []

    def fake(images, text, system=""):
        calls.append((images, text, system))
        return '{"ok": true}'

    with Serve(fake) as s:
        client = OpenAI(api_key="local", base_url=s.url, max_retries=0)
        response = client.chat.completions.create(
            model="qwen",
            messages=[
                {"role": "system", "content": "be brief"},
                {"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": data_url()}},
                    {"type": "text", "text": "is it upright?"},
                ]},
            ],
            temperature=0.0,
        )
        assert [m.id for m in client.models.list().data] == ["qwen3-vl-2b"]
    assert response.object == "chat.completion"
    assert response.choices[0].message.role == "assistant"
    assert response.choices[0].message.content == '{"ok": true}'
    assert response.choices[0].finish_reason == "stop"
    images, text, system = calls[0]
    assert images[0].mode == "RGB" and images[0].size == (8, 6)
    assert (text, system) == ("is it upright?", "be brief")


def test_empty_system_message_is_dropped():
    calls = []
    with Serve(lambda images, text, system="": calls.append(system) or "x") as s:
        OpenAI(api_key="local", base_url=s.url, max_retries=0).chat.completions.create(
            model="qwen",
            messages=[
                {"role": "system", "content": ""},
                {"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": data_url()}},
                    {"type": "text", "text": "q"},
                ]},
            ],
        )
    assert calls == [""]


@pytest.mark.parametrize("part", [
    {"type": "text", "text": "no image"},
    {"type": "image_url", "image_url": {"url": "http://example.com/a.jpg"}},
])
def test_rejects_request_without_image(part):
    with Serve(lambda *a: "never") as s:
        with pytest.raises(urllib.error.HTTPError) as err:
            post(s.url, {"model": "m", "messages": [{"role": "user", "content": [part]}]})
    assert err.value.code == 400
    assert "error" in json.loads(err.value.read())


def test_refuses_start_when_gpu_memory_low(monkeypatch, capsys):
    monkeypatch.setattr(server, "free_gpu_gb", lambda device: 1.0)
    monkeypatch.setattr(server, "load_generate", lambda *a: pytest.fail("model loaded"))
    with pytest.raises(SystemExit) as exit_:
        server.main(["--model", "m", "--min-free-gb", "4.4"])
    assert exit_.value.code != 0
    assert "1.0" in str(exit_.value.code) + capsys.readouterr().err
