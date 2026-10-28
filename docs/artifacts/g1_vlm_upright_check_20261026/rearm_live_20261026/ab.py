import base64, json, sys, time, urllib.request, re
from pathlib import Path
sys.path.insert(0, "docs/artifacts/g1_vlm_upright_check_20261026")
src = Path("docs/artifacts/g1_vlm_upright_check_20261026/upright_check.py").read_text()
VAL = eval(re.search(r"PROMPT = (\(.*?\))\n", src, re.S).group(1))
from vlm_orchestrator.harness.g1.rearm_monitor import PROMPT as REARM
frames = sorted(Path("docs/artifacts/g1_vlm_upright_check_20261026/frames").glob("*"))
def ask(img, text):
    url = "data:image/jpeg;base64," + base64.b64encode(img.read_bytes()).decode()
    body = {"model": "q", "messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": url}}, {"type": "text", "text": text}]}]}
    req = urllib.request.Request("http://127.0.0.1:8302/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=60).read())["choices"][0]["message"]["content"]
for _ in range(120):
    try: urllib.request.urlopen("http://127.0.0.1:8302/v1/models", timeout=2); break
    except Exception: time.sleep(1)
for f in frames:
    print(f.stem, "| VAL:", ask(f, VAL).replace("\n", " "), "| REARM:", ask(f, REARM).replace("\n", " "), flush=True)
