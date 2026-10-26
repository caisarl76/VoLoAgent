import base64, json, sys, urllib.request, io, collections
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw
S = Path(sys.argv[1])
P = ('You see the head camera of a humanoid robot. Answer only with JSON {"bottle_on_robot_hand": true|false}. '
     "bottle_on_robot_hand is true if a drink bottle rests on or is held by the robot's own hand (the black and silver "
     "metal gripper), including while a person touches it; false if the robot's hand is empty or not visible.")
def ask(jpeg):
    url = "data:image/jpeg;base64," + base64.b64encode(jpeg).decode()
    body = {"model": "q", "messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": url}}, {"type": "text", "text": P}]}]}
    r = urllib.request.Request("http://127.0.0.1:8301/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    raw = json.loads(urllib.request.urlopen(r, timeout=60).read())["choices"][0]["message"]["content"]
    try: return json.loads(raw[raw.index("{"):raw.rindex("}") + 1]).get("bottle_on_robot_hand")
    except ValueError: return "bad:" + raw
res = collections.Counter(); keep = []
for i in range(3):
    z = np.load(S / f"trial_{i:03d}.npz"); f = np.load(S / f"trial_{i:03d}_frames.npz")
    t = f["t_recv"]; raw = f["jpeg"].tobytes(); off = f["offsets"]; ws, we = float(z["wall_start"]), float(z["wall_stop"])
    grasp_t = None
    # positives: last 1.0 s; negatives: first 2.0 s of the episode (hand empty, reaching)
    for label, sel in (("on_hand", (t > we - 1.0) & (t <= we + 0.1)), ("empty", (t >= ws) & (t < ws + 2.0))):
        for k in np.flatnonzero(sel)[::3]:
            a = ask(raw[off[k]:off[k + 1]]); res[(label, a)] += 1
            if (label == "on_hand") != (a is True): keep.append((i, label, round(float(t[k] - ws), 2), a))
print("P:", P)
for k, v in sorted(res.items(), key=str): print(v, k)
print("misses:", keep[:12])
