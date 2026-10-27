"""Score a VLM server on labelled real robot frames with the loop's production prompts (2026-10-27).

Usage (VoLoAgent root): PYTHONPATH=. .venv/bin/python eval_vlm.py BASE_URL LABELS.json SELECTED.json OUT.json
Fields come from rearm_monitor.PROMPT (bottle_upright_on_case, hand_at_bottle), EMPTY_PROMPT (object_on_case) and
TAKEN_PROMPT (bottle_on_robot_hand). Gate errors that move the robot wrongly are reported separately:
  rearm_false_yes  - the bottle step says yes (upright and no hand) when the frame has no upright bottle or a human hand at it
  taken_false_yes  - the taken check says the bottle is off the robot hand while it is still on it
"""
import base64, json, sys, time, urllib.request
from pathlib import Path
from vlm_orchestrator.harness.g1.rearm_monitor import EMPTY_PROMPT, PROMPT, TAKEN_PROMPT

base, labels, selected, out = sys.argv[1], *map(Path, sys.argv[2:5])
import os
MODEL = os.environ.get("VLM_MODEL", "m")  # vLLM checks the served model name
EXTRA = json.loads(os.environ.get("VLM_EXTRA_BODY", "{}"))  # e.g. {"chat_template_kwargs": {"enable_thinking": false}}
L = {x["id"]: x for x in json.loads(labels.read_text())}
S = {x["id"]: x for x in json.loads(selected.read_text())}

def ask(path, text):
    url = "data:image/jpeg;base64," + base64.b64encode(Path(path).read_bytes()).decode()
    body = {"model": MODEL, "messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": url}}, {"type": "text", "text": text}]}],
            "temperature": 0.0, "max_tokens": 64, **EXTRA}  # greedy as the loop's client; vLLM defaults to sampling
    t = time.monotonic()
    r = urllib.request.Request(base + "/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    raw = json.loads(urllib.request.urlopen(r, timeout=120).read())["choices"][0]["message"]["content"]
    try:
        return json.loads(raw[raw.index("{"):raw.rindex("}") + 1]), time.monotonic() - t
    except ValueError:
        return {}, time.monotonic() - t

rows, lat = [], []
for i in sorted(L):
    p = str(Path(selected.parent.parent) / S[i]["path"]) if not Path(S[i]["path"]).is_absolute() else S[i]["path"]
    up, t1 = ask(p, PROMPT); em, t2 = ask(p, EMPTY_PROMPT); tk, t3 = ask(p, TAKEN_PROMPT); lat += [t1, t2, t3]
    rows.append({"id": i, "kind": S[i]["kind"], "pred": {**up, **em, **tk}, "label": L[i]})
acc = {}
for f, lf in [("bottle_upright_on_case", "bottle_upright_on_case"), ("object_on_case", "object_on_case"), ("bottle_on_robot_hand", "bottle_on_robot_hand")]:
    ok = [r["pred"].get(f) is r["label"][lf] for r in rows]
    acc[f] = f"{sum(ok)}/{len(ok)}"
# hand_at_bottle in PROMPT counts robot or human hands: compare to (human hand) or (bottle on robot hand)
ok = [r["pred"].get("hand_at_bottle") is (r["label"]["human_hand_at_bottle"] or r["label"]["bottle_on_robot_hand"]) for r in rows]
acc["hand_at_bottle"] = f"{sum(ok)}/{len(ok)}"
RE = [r for r in rows if r["kind"].startswith("rearm")]; TK = [r for r in rows if r["kind"] == "taken"]  # each check is scored where it runs
rearm_fy = [r["id"] for r in RE if r["pred"].get("bottle_upright_on_case") is True and r["pred"].get("hand_at_bottle") is False
            and not (r["label"]["bottle_upright_on_case"] and not r["label"]["human_hand_at_bottle"])]
rearm_fn = [r["id"] for r in RE if r["label"]["bottle_upright_on_case"] and not r["label"]["human_hand_at_bottle"] and not r["label"]["bottle_on_robot_hand"]
            and not (r["pred"].get("bottle_upright_on_case") is True and r["pred"].get("hand_at_bottle") is False)]
taken_fy = [r["id"] for r in TK if r["label"]["bottle_on_robot_hand"] and r["pred"].get("bottle_on_robot_hand") is False]
taken_fn = [r["id"] for r in TK if not r["label"]["bottle_on_robot_hand"] and r["pred"].get("bottle_on_robot_hand") is not False]
res = {"base_url": base, "n": len(rows), "n_rearm": len(RE), "n_taken": len(TK), "n_taken_on_hand": sum(r["label"]["bottle_on_robot_hand"] for r in TK), "field_accuracy": acc,
       "rearm_false_yes": rearm_fy, "rearm_missed_yes": rearm_fn, "taken_false_yes": taken_fy, "taken_missed_yes": taken_fn,
       "latency_s_median": sorted(lat)[len(lat) // 2], "latency_s_p95": sorted(lat)[int(len(lat) * 0.95)], "rows": rows}
out.write_text(json.dumps(res, indent=1))
print(json.dumps({k: (v if not isinstance(v, list) else (len(v), v[:12])) for k, v in res.items() if k != "rows"}, indent=1))
