import json, sys, collections
items = json.load(open(sys.argv[1]))
S = collections.defaultdict(lambda: collections.Counter())
for x in items:
    p, c = x["pred"], x["label"]
    rearm_yes = p.get("bottle_upright_on_case") is True and p.get("hand_at_bottle") is False
    empty_yes = p.get("bottle_upright_on_case") is False and p.get("hand_at_bottle") is False and p.get("object_on_case") is False
    on = p.get("bottle_on_robot_hand")
    S[c]["n"] += 1; S[c]["rearm_yes"] += rearm_yes; S[c]["empty_yes"] += empty_yes
    S[c]["on_true"] += on is True; S[c]["on_false"] += on is False; S[c]["on_none"] += on is None
print(f"{'class':5s} {'n':>4s} {'re-arm bottle yes':>18s} {'empty yes':>10s} {'on_hand T/F/?':>14s}")
for c in "CGHPEU":
    s = S[c]; print(f"{c:5s} {s['n']:4d} {s['rearm_yes']:18d} {s['empty_yes']:10d} {s['on_true']:6d}/{s['on_false']}/{s['on_none']}")
bad = lambda f: [(x["id"], x["session"], x["trial"], x["t"]) for x in items if f(x)]
print("re-arm false yes (G/H/P/E):", bad(lambda x: x["label"] in "GHPE" and x["pred"].get("bottle_upright_on_case") is True and x["pred"].get("hand_at_bottle") is False))
print("taken false (H says not on hand):", bad(lambda x: x["label"] == "H" and x["pred"].get("bottle_on_robot_hand") is False))
print("empty false yes (C/G/H/P):", bad(lambda x: x["label"] in "CGHP" and x["pred"].get("bottle_upright_on_case") is False and x["pred"].get("hand_at_bottle") is False and x["pred"].get("object_on_case") is False))
