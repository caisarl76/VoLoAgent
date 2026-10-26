import base64, json, sys, time, urllib.request
from pathlib import Path
from openai import OpenAI
sys.path.insert(0, "tests")
from vlm_orchestrator.harness.g1.rearm_monitor import RearmMonitor
from vlm_orchestrator.harness.g1.registry import HarnessLimits
import test_g1_monitor as tm
from dataclasses import replace
A = Path("docs/artifacts/g1_vlm_upright_check_20261026")
labels = {r["frame"]: r for r in json.loads((A / "qwen3-vl-2b.json").read_text())["rows"]}
for _ in range(120):
    try: urllib.request.urlopen("http://127.0.0.1:8302/v1/models", timeout=2); break
    except Exception: time.sleep(1)
api = OpenAI(api_key="local", base_url="http://127.0.0.1:8302/v1", timeout=10, max_retries=0)
def vlm(system, content):
    return api.chat.completions.create(model="q", messages=[{"role": "system", "content": system}, {"role": "user", "content": content}], temperature=0.0).choices[0].message.content
ok = {"empty": 0, "bottle": 0}
for f in sorted((A / "frames").glob("*")):
    snap = replace(tm.snapshot(f.stem, now=time.monotonic()), jpeg_rgb_b64=base64.b64encode(f.read_bytes()).decode()) if hasattr(tm.snapshot(f.stem), "jpeg_rgb_b64") else None
    row, out = labels[f.stem], {}
    for step in ("bottle", "empty"):
        m = RearmMonitor(vlm, time.monotonic, HarnessLimits() if callable(HarnessLimits) else None)
        snap = replace(snap, received_at=time.monotonic()) if hasattr(snap, "received_at") else snap
        m.begin(step, snap)
        d = m.check(replace(snap, received_at=time.monotonic()))
        out[step] = d.outcome; (step == "empty" and f.stem.endswith("last") and print("   ", d.reason, json.dumps(getattr(d, "answers", None))))
    want_bottle = "yes" if (row["label_upright"] and not row["label_hand"]) else "no"
    ok["bottle"] += out["bottle"] == want_bottle
    print(f"{f.stem:14s} label up={row['label_upright']!s:5} hand={row['label_hand']!s:5} | bottle={out['bottle']:7} (want {want_bottle}) | empty={out['empty']}", flush=True)
print("bottle-step combined", ok["bottle"], "/ 17")
