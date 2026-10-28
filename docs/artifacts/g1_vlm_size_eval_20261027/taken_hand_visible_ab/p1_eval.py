"""Score the taken prompt (current) vs a hand-visible version on hold frames. Usage: p1_eval.py BASE_URL"""
import base64, json, sys, urllib.request
from pathlib import Path
from vlm_orchestrator.harness.g1.rearm_monitor import TAKEN_PROMPT
V2 = ('You see the head camera of a humanoid robot. Answer only with JSON '
      '{"robot_hand_visible": true|false, "bottle_on_robot_hand": true|false}. robot_hand_visible is true if the '
      "robot's own hand (the black and silver metal gripper) is at least partly in the image. bottle_on_robot_hand "
      "is true if a drink bottle rests on or is held by the robot's own hand, including while a person touches it.")
base = sys.argv[1]
def ask(path, text):
    url = "data:image/jpeg;base64," + base64.b64encode(Path(path).read_bytes()).decode()
    body = {"model": "qwen3-vl-4b", "messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": url}}, {"type": "text", "text": text}]}], "temperature": 0.0, "max_tokens": 64}
    raw = json.loads(urllib.request.urlopen(urllib.request.Request(base + "/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=120).read())["choices"][0]["message"]["content"]
    try: return json.loads(raw[raw.index("{"):raw.rindex("}") + 1])
    except ValueError: return {}
A = [x for x in json.load(open("/home/jihun/.claude/jobs/eec392e6/tmp/takeset/items.json")) if x["label"] != "uncertain"]
B = json.load(open("/home/jihun/.claude/jobs/eec392e6/tmp/setB.json"))
out = {}
for name, items in (("A r28-r31", A), ("B R32-critical", B)):
    rows = []
    for x in items:
        cur = ask(x["path"], TAKEN_PROMPT).get("bottle_on_robot_hand")
        v2 = ask(x["path"], V2)
        rows.append({**x, "cur_yes": cur is False, "v2_vis": v2.get("robot_hand_visible"), "v2_on": v2.get("bottle_on_robot_hand"),
                     "v2_yes": v2.get("robot_hand_visible") is True and v2.get("bottle_on_robot_hand") is False})
    on = [r for r in rows if r["label"] == "on"]; off = [r for r in rows if r["label"] in ("off", "empty")]
    print(f"== {name}: {len(on)} bottle-on frames, {len(off)} empty-hand frames")
    for k in ("cur", "v2"):
        print(f"  {k:3s} false taken {sum(r[k+'_yes'] for r in on):3d}/{len(on)}   empty-hand yes {sum(r[k+'_yes'] for r in off):3d}/{len(off)}")
    print("  v2 hand not visible: on-frames", sum(r["v2_vis"] is False for r in on), " empty-frames", sum(r["v2_vis"] is False for r in off))
    out[name] = rows
json.dump(out, open("/home/jihun/.claude/jobs/eec392e6/tmp/p1_eval.json", "w"), indent=0, default=str)
by = {}
for r in out["B R32-critical"]:
    by.setdefault((r["session"], r["trial"], r["label"]), []).append(("Y" if r["v2_yes"] else ("u" if r["v2_vis"] is False else "n")) + ("Y" if r["cur_yes"] else "n"))
for k, v in by.items(): print(" ", k, "v2/cur:", " ".join(v))
