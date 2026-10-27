import base64, json, sys, urllib.request, collections
from pathlib import Path
base, model = sys.argv[1], sys.argv[2]
LB = Path("/home/jihun/.claude/jobs/eec392e6/tmp/labels")
L = {x["id"]: x for x in json.load(open(LB / "labels.json"))}; S = {x["id"]: x for x in json.load(open(LB / "selected.json"))}
P = ('Image 1 was taken by the robot\'s head camera at the end of a handover, while the robot held a drink bottle out on its '
     'own hand (the black and silver metal gripper). Image 2 is the same camera now. Answer only with JSON '
     '{"bottle_still_on_robot_hand": true|false}. bottle_still_on_robot_hand is true if in image 2 the bottle still rests on or '
     'is held by the robot\'s hand, including while a person touches it; false only if the robot\'s hand no longer holds it.')
def enc(p): return {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(open(p, "rb").read()).decode()}}
def ask(ref, cur):
    body = {"model": model, "temperature": 0, "max_tokens": 64, "chat_template_kwargs": {"enable_thinking": False},
            "messages": [{"role": "user", "content": [enc(ref), enc(cur), {"type": "text", "text": P}]}]}
    raw = json.loads(urllib.request.urlopen(urllib.request.Request(base + "/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=120).read())["choices"][0]["message"]["content"]
    try: return json.loads(raw[raw.index("{"):raw.rindex("}") + 1]).get("bottle_still_on_robot_hand")
    except ValueError: return None
root = LB.parent
TK = [i for i in sorted(L) if S[i]["kind"] == "taken"]
by_trial = collections.defaultdict(list)
for i in TK: by_trial[(S[i]["session"], S[i]["path"].split("/")[-2])].append(i)
fy, fn = [], []
for key, ids in by_trial.items():
    ref = next((i for i in ids if L[i]["bottle_on_robot_hand"]), ids[0])  # the hold's first on-hand frame
    for i in ids:
        v = ask(root / S[ref]["path"], root / S[i]["path"]); on = L[i]["bottle_on_robot_hand"]
        if on and v is not True: fy.append(i)
        if not on and v is not False: fn.append(i)
print(model, "two-image: FALSE TAKEN", len(fy), fy, "| missed taken", len(fn), "of", sum(not L[i]["bottle_on_robot_hand"] for i in TK), fn[:15])
