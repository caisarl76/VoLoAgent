import base64, json, time, urllib.request
from pathlib import Path
from vlm_orchestrator.harness.g1.rearm_monitor import EMPTY_PROMPT
print("EMPTY_PROMPT =", EMPTY_PROMPT)
ALT = ('You see the head camera of a humanoid robot looking down at a black case. The case has a built-in carrying handle and '
       'white labels on its lid; they are part of the case. Answer only with JSON {"object_on_case": true|false}. '
       "object_on_case is true if a bottle or any other loose object stands or lies on top of the black case.")
def ask(img, text):
    url = "data:image/jpeg;base64," + base64.b64encode(img.read_bytes()).decode()
    body = {"model": "q", "messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": url}}, {"type": "text", "text": text}]}]}
    req = urllib.request.Request("http://127.0.0.1:8302/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=60).read())["choices"][0]["message"]["content"].replace("\n", " ")
for _ in range(120):
    try: urllib.request.urlopen("http://127.0.0.1:8302/v1/models", timeout=2); break
    except Exception: time.sleep(1)
for f in sorted(Path("docs/artifacts/g1_vlm_upright_check_20261026/frames").glob("*")):
    print(f"{f.stem:14s} ALT: {ask(f, ALT)}", flush=True)
