import base64, json, sys, urllib.request, itertools
from pathlib import Path
from vlm_orchestrator.harness.g1.rearm_monitor import TAKEN_PROMPT
here=Path(__file__).parent; d=json.load(open(here/"dense.json"))
for x in d:
    url="data:image/jpeg;base64,"+base64.b64encode(Path(x["path"]).read_bytes()).decode()
    body={"model":"qwen3-vl-4b","messages":[{"role":"user","content":[{"type":"image_url","image_url":{"url":url}},{"type":"text","text":TAKEN_PROMPT}]}],"temperature":0.0,"max_tokens":64}
    raw=json.loads(urllib.request.urlopen(urllib.request.Request(sys.argv[1]+"/chat/completions",json.dumps(body).encode(),{"Content-Type":"application/json"}),timeout=120).read())["choices"][0]["message"]["content"]
    try: x["on"]=json.loads(raw[raw.index("{"):raw.rindex("}")+1]).get("bottle_on_robot_hand")
    except ValueError: x["on"]=None
json.dump(d,open(here/"dense-4b.json","w"),indent=0)
runs={}
for k,g in itertools.groupby(d,key=lambda x:(x["session"],x["trial"])):
    g=list(g); best=cur=0; s=""
    for x in g:
        cur=cur+1 if x["on"] is False else 0; best=max(best,cur); s+={True:"H",False:"-",None:"?"}[x["on"]]
    runs[k]=(best,s)
false=sum(x["on"] is False for x in d)
print("dense on-hand frames",len(d),"false taken",false)
for k,(b,s) in runs.items():
    if "-" in s: print(k,"longest run",b,s)
print("trials whose longest false run >= 3:",[k for k,(b,s) in runs.items() if b>=3])
