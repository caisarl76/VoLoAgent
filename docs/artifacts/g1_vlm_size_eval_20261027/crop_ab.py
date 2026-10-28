import base64, io, json, sys, urllib.request
from pathlib import Path
from PIL import Image
L = {x["id"]: x for x in json.load(open(sys.argv[2]))}; S = {x["id"]: x for x in json.load(open(sys.argv[3]))}
base = sys.argv[1]
P = 'Answer only with JSON {"bottle_visible": true|false}. bottle_visible is true if any part of a drink bottle (clear plastic, yellow-green cap) is visible in this image.'
BOXES = {"tr55": (0.55, 0.0, 1.0, 0.55), "tr45": (0.45, 0.0, 1.0, 0.6)}
def ask(img, text):
    b = io.BytesIO(); img.save(b, "JPEG", quality=95)
    body = {"model": "m", "messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(b.getvalue()).decode()}}, {"type": "text", "text": text}]}]}
    raw = json.loads(urllib.request.urlopen(urllib.request.Request(base + "/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=120).read())["choices"][0]["message"]["content"]
    try: return json.loads(raw[raw.index("{"):raw.rindex("}") + 1]).get("bottle_visible")
    except ValueError: return None
TK = [i for i in sorted(L) if S[i]["kind"] == "taken"]
for name, (x0, y0, x1, y1) in BOXES.items():
    fy, fn, rows = [], [], {}
    for i in TK:
        im = Image.open(Path(sys.argv[3]).parent.parent / S[i]["path"]).convert("RGB"); w, h = im.size
        v = ask(im.crop((int(x0 * w), int(y0 * h), int(x1 * w), int(y1 * h))), P); rows[i] = v
        on = L[i]["bottle_on_robot_hand"]
        if on and v is not True: fy.append(i)        # said no bottle (taken) while on the hand: dangerous
        if not on and v is not False: fn.append(i)   # said bottle while empty: only delays
    print(name, "FALSE TAKEN", len(fy), fy, "| missed taken", len(fn), "of", sum(not L[i]["bottle_on_robot_hand"] for i in TK), fn[:20])
    json.dump(rows, open(Path(sys.argv[2]).parent / f"crop-{name}-{base.rsplit(':',1)[1].split('/')[0]}.json", "w"))
