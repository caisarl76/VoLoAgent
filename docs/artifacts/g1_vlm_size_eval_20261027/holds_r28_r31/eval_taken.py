"""Score a VLM on the r28-r31 hold frames with the loop's TAKEN_PROMPT. Usage: eval_taken.py BASE_URL NAME"""
import base64, json, os, sys, time, urllib.request
from pathlib import Path
from vlm_orchestrator.harness.g1.rearm_monitor import TAKEN_PROMPT
base, name = sys.argv[1:3]
EXTRA = json.loads(os.environ.get("VLM_EXTRA_BODY", "{}"))
here = Path(__file__).parent
items = json.load(open(here / "items.json"))
def ask(path):
    url = "data:image/jpeg;base64," + base64.b64encode(Path(path).read_bytes()).decode()
    body = {"model": os.environ.get("VLM_MODEL", "m"), "messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": url}}, {"type": "text", "text": TAKEN_PROMPT}]}],
            "temperature": 0.0, "max_tokens": 64, **EXTRA}
    r = urllib.request.Request(base + "/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    raw = json.loads(urllib.request.urlopen(r, timeout=120).read())["choices"][0]["message"]["content"]
    try: return json.loads(raw[raw.index("{"):raw.rindex("}") + 1]).get("bottle_on_robot_hand")
    except ValueError: return None
lat = []
for x in items:
    t = time.monotonic(); x["pred"] = ask(x["path"]); lat.append(time.monotonic() - t)
fy = [i for i, x in enumerate(items) if x["label"] == "on" and x["pred"] is False]
fn = [i for i, x in enumerate(items) if x["label"] == "off" and x["pred"] is not False]
holds = {}
for i, x in enumerate(items): holds.setdefault(f"{x['session']}_{x['trial']}", []).append(i)
timeline = {h: "".join({"on": "o", "off": "_", "uncertain": "~"}[items[i]["label"]] for i in ix) + "  " +
            "".join({True: "H", False: "-", None: "?"}[items[i]["pred"]] for i in ix) for h, ix in holds.items()}
res = {"model": name, "n_on": sum(x["label"] == "on" for x in items), "n_off": sum(x["label"] == "off" for x in items),
       "false_taken": fy, "missed": len(fn), "latency_median": sorted(lat)[len(lat) // 2], "timeline": timeline}
(here / f"eval-{name}.json").write_text(json.dumps({**res, "preds": [x["pred"] for x in items]}, indent=1))
print(json.dumps(res, indent=1))
