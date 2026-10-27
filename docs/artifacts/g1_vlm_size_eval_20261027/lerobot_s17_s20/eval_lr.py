"""Ask the loop's three prompts on each labelled s17-s20 frame. Usage: eval_lr.py BASE_URL NAME"""
import base64, json, sys, time, urllib.request
from pathlib import Path
from vlm_orchestrator.harness.g1.rearm_monitor import EMPTY_PROMPT, PROMPT, TAKEN_PROMPT
base, name = sys.argv[1:3]
here = Path(__file__).parent
items = json.load(open(here / "items.json"))
def ask(path, text):
    url = "data:image/jpeg;base64," + base64.b64encode(Path(path).read_bytes()).decode()
    body = {"model": name, "messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": url}}, {"type": "text", "text": text}]}],
            "temperature": 0.0, "max_tokens": 64}
    r = urllib.request.Request(base + "/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    raw = json.loads(urllib.request.urlopen(r, timeout=120).read())["choices"][0]["message"]["content"]
    try: return json.loads(raw[raw.index("{"):raw.rindex("}") + 1])
    except ValueError: return {}
t0 = time.monotonic()
for x in items:
    x["pred"] = {**ask(x["path"], PROMPT), **ask(x["path"], EMPTY_PROMPT), **ask(x["path"], TAKEN_PROMPT)}
(here / f"eval-{name}.json").write_text(json.dumps(items, indent=0))
print("done", len(items), f"{time.monotonic() - t0:.0f} s")
