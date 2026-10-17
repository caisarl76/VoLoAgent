"""Render the saved G1 evidence as figures and a self-contained HTML report.

No network calls, credentials, robot imports or control commands are used.
Run with the workstation's Isaac-GR00T Python (it already has Matplotlib).
"""

import base64
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


OUT = Path(__file__).resolve().parent
SOURCE = OUT.parent / "g1_harness_20261002"
COLORS = {
    "ink": "#172b40", "muted": "#657587", "green": "#187b56",
    "red": "#bd3f37", "blue": "#267998", "amber": "#a56913",
    "line": "#dce3ec", "background": "#f4f7fb",
}


def read_json(path):
    return json.loads(path.read_text())


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def data_url(path):
    mime = "image/png" if path.suffix == ".png" else "image/jpeg"
    return f"data:{mime};base64," + base64.b64encode(path.read_bytes()).decode()


def trial(name):
    folder = SOURCE / name
    events_path = next((folder / "mission").glob("*/events.jsonl"))
    events = read_jsonl(events_path)
    start = next(row["started_at"] for row in events if row["event"] == "skill_start")
    end = next(row["at"] for row in events if row["event"] == "mission_result")
    frames = {row["frame_id"]: row for row in events if row["event"] == "monitor_frame"}
    decisions = []
    for row in events:
        if row["event"] != "monitor_decision":
            continue
        saved = events_path.parent / Path(frames[row["frame_id"]]["path"]).name
        raw = json.loads(row["raw_response"])
        decisions.append({
            "captured_s": row["captured_at"] - start,
            "decided_s": row["decided_at"] - start,
            "raw_status": raw["status"], "monitor_outcome": row["outcome"],
            "reason": raw["reason"], "image": data_url(saved),
            "source_frame": str(saved.relative_to(SOURCE)),
        })
    truth = [row for row in read_jsonl(folder / "bottle-ground-truth.jsonl")
             if start <= row["at"] <= end]
    assert truth and all(set(row["bottle_contacts"]) == {"source_table"} for row in truth)
    return {
        "duration_s": end - start, "decisions": decisions,
        "truth": [{"time_s": row["at"] - start, "position": row["bottle_position"]}
                  for row in truth],
        "result": read_json(folder / "result.json"),
        "source_contacts_only": True,
    }


def style_axis(ax):
    ax.set_facecolor("white")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("bottom", "left"):
        ax.spines[side].set_color(COLORS["line"])
    ax.tick_params(colors=COLORS["muted"], labelsize=10, length=0, pad=8)
    ax.grid(axis="x", color=COLORS["line"], linewidth=0.6)
    ax.set_axisbelow(True)


def timeline(ax, trials):
    style_axis(ax)
    for y, trial_name in ((1, "original"), (0, "corrected")):
        t = trials[trial_name]
        ax.hlines(y, 0, t["duration_s"], color=COLORS["line"], linewidth=4)
        for d in t["decisions"]:
            color = COLORS["red"] if d["raw_status"] == "complete" else COLORS["blue"]
            ax.scatter(d["decided_s"], y, s=32, color=color, zorder=3)
    ax.axvline(trials["corrected"]["duration_s"], color=COLORS["amber"], ls="--", lw=1.5)
    ax.set(xlim=(-2, 126), ylim=(-0.55, 1.7), yticks=[0, 1],
           yticklabels=["Corrected", "Original"])
    ax.text(16, 1.17, "2 false-complete frame replies", color=COLORS["red"], fontsize=10)
    ax.text(27, 0.19, "44 in-progress replies", color=COLORS["blue"], fontsize=10)
    ax.text(121.5, 0.5, "Stop\nat 120 s", color=COLORS["amber"], fontsize=10, va="center")
    ax.set_title("Live frame verdicts before and after the criterion fix", loc="left", pad=17,
                 fontweight="bold", fontsize=13)


def bottle_distance(ax, corrected, layout):
    style_axis(ax)
    center = layout["destination_stool"]["center_xy"]
    rows = corrected["truth"]
    distance = [((r["position"][0] - center[0]) ** 2 +
                 (r["position"][1] - center[1]) ** 2) ** 0.5 for r in rows]
    ax.plot([r["time_s"] for r in rows], distance, color=COLORS["red"], lw=2.5)
    ax.set(xlim=(0, 121), ylim=(0, 0.74), yticks=[0, 0.3, 0.6], ylabel="Meters",
           xlabel="Seconds after manipulation starts")
    ax.text(5, 0.39, "Bottle stayed ~61 cm from stool center\nEvery recorded contact was with the source table",
            fontsize=10, color=COLORS["ink"])
    ax.set_title("Independent object truth in the corrected trial", loc="left", pad=9,
                 fontweight="bold", fontsize=12)


def turn_bars(ax, turns):
    style_axis(ax)
    labels = ["+15°", "−15°", "−15° near boundary\n(no crossing)",
              "−15° crossing boundary", "+15° crossing boundary"]
    for i, t in enumerate(turns):
        color = COLORS["green"] if t["outcome"] == "passed" else COLORS["red"]
        ax.barh(i, t["error_deg"], color=color, height=0.55)
        ax.text(t["error_deg"] + 0.12, i, f'{t["error_deg"]:.2f}°', va="center",
                fontsize=10, color=color, fontweight="bold")
    ax.axvline(3, color=COLORS["amber"], ls="--", lw=1.6)
    ax.text(3.1, -0.65, "3° limit", color=COLORS["amber"], fontsize=10)
    ax.set(yticks=range(5), yticklabels=labels, xlim=(0, 11), ylim=(4.7, -0.95),
           xlabel="Measured heading error at trial end (degrees)")
    ax.set_title("Turning still needs work", loc="left", pad=14, fontsize=13, fontweight="bold")


def figures(data):
    plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": COLORS["ink"],
                        "axes.labelcolor": COLORS["muted"], "svg.fonttype": "none"})
    fig = plt.figure(figsize=(16, 10.5), facecolor=COLORS["background"])
    fig.text(0.04, 0.95, "G1 harness: the loop runs, placement remains unproven",
             fontsize=23, fontweight="bold")
    fig.text(0.04, 0.915, "Saved simulation results · 2 October 2026 · No physical robot was used",
             fontsize=12, color=COLORS["muted"])
    cards = [("50 / 50", "G1 software tests", "green"),
             ("23 / 23", "Native harness / launcher tests", "green"),
             ("44", "Corrected live in-progress replies", "blue"),
             ("No placement", "Both live trials kept the bottle on source", "red")]
    for i, (value, label, color) in enumerate(cards):
        x = 0.04 + i * 0.238
        fig.patches.append(FancyBboxPatch((x, 0.805), 0.218, 0.078,
                           boxstyle="round,pad=0.007", transform=fig.transFigure,
                           facecolor="white", edgecolor=COLORS["line"], linewidth=0.8))
        fig.text(x + 0.012, 0.845, value, fontsize=20, fontweight="bold", color=COLORS[color])
        fig.text(x + 0.012, 0.820, label, fontsize=9.4, color=COLORS["muted"])
    fig.text(0.04, 0.766,
             "VoLoAgent → registered prompt → real VLA checkpoint → SONIC → MuJoCo + camera → Genon monitor",
             fontsize=11, color=COLORS["muted"])
    scene = fig.add_axes([0.04, 0.427, 0.395, 0.292])
    scene.imshow(mpimg.imread(SOURCE / "bottle-scene/overview.png"))
    scene.axis("off")
    scene.set_title("Sample scene — mapping and camera checks pass", fontsize=13,
                    fontweight="bold", loc="left", pad=9)
    fig.text(0.055, 0.385,
             "Both surfaces: 80 cm high. Table: 50 cm forward / 20 cm left.\n"
             "Stool: 35 cm forward / 60 cm right. Bottle: provisional geometry.",
             fontsize=10, color=COLORS["muted"], linespacing=1.55)
    timeline(fig.add_axes([0.535, 0.585, 0.415, 0.13]), data["trials"])
    bottle_distance(fig.add_axes([0.535, 0.425, 0.415, 0.105]),
                    data["trials"]["corrected"], data["layout"])
    turn_bars(fig.add_axes([0.177, 0.095, 0.282, 0.235]), data["turns"])
    status = fig.add_axes([0.535, 0.095, 0.43, 0.235])
    status.axis("off")
    status.text(0, 1.02, "What is checked, and what remains", fontsize=13, fontweight="bold")
    lines = [
        ("PASS", "Policy stall interrupts with confirmed hold", "green"),
        ("PASS", "Fresh late result during reset is discarded", "green"),
        ("PASS", "Recorded vision: 8 negatives, 0 false-completes", "green"),
        ("OPEN", "Calibrated grasp + released placement", "red"),
        ("OPEN", "Repeatable turns + carried-bottle occlusion", "amber"),
        ("OPEN", "172 training episodes still need outcome review", "amber"),
        ("PENDING", "Supervised hardware; dashboard deferred", "muted"),
    ]
    for i, (label, text, color) in enumerate(lines):
        y = 0.88 - i * 0.13
        status.text(0, y, label, color=COLORS[color], fontsize=10, fontweight="bold")
        status.text(0.17, y, text, fontsize=10)
    fig.text(0.04, 0.025,
             "The corrected criterion resolves the observed false-success case. These small sets do not establish general reliability.",
             fontsize=10, color=COLORS["muted"])
    fig.savefig(OUT / "results-overview.png", dpi=150)
    fig.savefig(OUT / "results-overview.pdf")
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(10.5, 4.5), layout="constrained", facecolor="white")
    turn_bars(ax, data["turns"])
    fig.savefig(OUT / "turning.svg")
    plt.close(fig)
    svg_path = OUT / "turning.svg"
    svg_path.write_text("\n".join(line.rstrip() for line in svg_path.read_text().splitlines()) + "\n")


HTML = r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; img-src data:; style-src 'unsafe-inline'; script-src 'unsafe-inline'; connect-src 'none'; base-uri 'none'">
<title>G1 harness — saved validation results</title>
<style>
:root{color-scheme:light;--ink:#172b40;--muted:#657587;--line:#dce3ec;--green:#187b56;--red:#bd3f37;--blue:#267998;--amber:#a56913}
*{box-sizing:border-box}body{margin:0;background:#f4f7fb;color:var(--ink);font:16px/1.55 system-ui,sans-serif}
main{max-width:1220px;margin:auto;padding:44px 28px 70px}h1{font-size:clamp(27px,4vw,42px);line-height:1.2;letter-spacing:-1px;margin:12px 0}h2{font-size:23px;margin:0 0 12px}h3{font-size:18px;margin:0 0 8px}p{margin:8px 0 16px}.eyebrow{font-size:13px;letter-spacing:1.4px;text-transform:uppercase;color:var(--muted)}.muted{color:var(--muted)}.small{font-size:13px}.green{color:var(--green)}.red{color:var(--red)}.blue{color:var(--blue)}.amber{color:var(--amber)}
.cards{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin:27px 0}.card,.panel{background:white;border:1px solid var(--line);border-radius:16px;padding:22px}.card strong{display:block;font-size:30px;letter-spacing:-.6px}.card span{font-size:13px;color:var(--muted)}.panel{margin:20px 0}.row{display:grid;grid-template-columns:1fr 1fr;gap:24px}.row .panel{margin:0}.full{width:100%;display:block;border-radius:10px}.notice{padding:15px 18px;border-left:4px solid var(--amber);background:#fff8ed;border-radius:0 8px 8px 0}.tabs{display:flex;gap:8px;margin:16px 0;flex-wrap:wrap}button{font:inherit;color:var(--ink);background:white;border:1px solid var(--line);border-radius:9px;padding:9px 15px;cursor:pointer}button:hover{border-color:var(--blue)}button[aria-pressed=true]{background:var(--ink);color:white;border-color:var(--ink)}button:disabled{opacity:.45;cursor:default}button:focus-visible,input:focus-visible{outline:3px solid #66afd0;outline-offset:3px}.viewer{display:grid;grid-template-columns:1.2fr 1fr;gap:23px}.controls{display:flex;gap:12px;align-items:center;margin:15px 0}input[type=range]{width:100%;accent-color:var(--blue)}.pill{display:inline-block;font-size:12px;font-weight:700;border-radius:99px;padding:5px 10px;background:#eaf4f8}.pill.red{background:#fff0ee}.pill.amber{background:#fff4df}.key{display:block;margin-top:12px;color:var(--muted);font-size:12px;text-transform:uppercase;letter-spacing:.7px}.quote{font-size:18px;padding:15px 0;border-bottom:1px solid var(--line);line-height:1.45}dl{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin:14px 0}dt{font-size:12px;color:var(--muted)}dd{margin:2px 0;font-weight:600}table{border-collapse:collapse;width:100%;font-size:14px}td,th{padding:10px 8px;text-align:left;border-bottom:1px solid var(--line)}th{font-size:12px;color:var(--muted)}ul{padding-left:20px}li{margin:9px 0}details{margin-top:15px}summary{cursor:pointer;color:var(--blue)}code{background:#eef2f7;padding:2px 5px;border-radius:4px;font-size:12px;overflow-wrap:anywhere}a{color:var(--blue)}.reviewbar{height:18px;display:flex;border-radius:6px;overflow:hidden;background:#e8edf3;margin:18px 0}.reviewbar .success{background:var(--green);width:calc(100% / 174)}.reviewbar .failure{background:var(--red);width:calc(100% / 174)}.reviewbar .unknown{flex:1;background:#d7e0eb}.legend{display:flex;gap:18px;flex-wrap:wrap;font-size:13px}.legend b{font-size:18px}.step{display:flex;gap:14px;align-items:baseline;padding:9px 0}.step .tag{width:70px;flex-shrink:0;font-size:12px;font-weight:700}.footer{margin-top:30px;font-size:12px;color:var(--muted)}
@media(max-width:760px){main{padding:26px 16px}.cards{grid-template-columns:1fr 1fr}.row,.viewer{grid-template-columns:1fr}.card,.panel{padding:17px}.card strong{font-size:26px}}
@media print{body{background:white}main{padding:0}.panel,.card{break-inside:avoid}.tabs,.controls{display:none}}
</style></head><body><main>
<div class="eyebrow">Unitree G1 · saved validation · 02 October 2026</div>
<h1>The control loop runs.<br>The bottle task still does not complete.</h1>
<p class="muted">VoLoAgent, the real VLA checkpoint, SONIC, MuJoCo, the simulator camera and the Genon monitor ran together. The bottle remained on the source table in both live trials.</p>
<div class="cards"><div class="card"><strong class="green">50 / 50</strong><span>G1 tests passed · no skips</span></div><div class="card"><strong class="green">23 / 23</strong><span>Native harness / launcher tests passed</span></div><div class="card"><strong class="blue">44</strong><span>Corrected live in-progress replies</span></div><div class="card"><strong class="red">No placement</strong><span>Two live trials · source contact only</span></div></div>
<section class="panel"><h2>The session at a glance</h2><img class="full" src="__OVERVIEW__" alt="Validation overview with sample scene, live verdict timeline, stationary bottle truth and turning error bars"><p class="small muted">This is a report of saved simulation evidence. All owned trial processes were stopped. No physical robot was used.</p><a href="results-overview.png">PNG figure</a> · <a href="results-overview.pdf">PDF figure</a></section>
<section class="panel"><h2>Inspect the live camera evidence</h2><p>The original criterion called a bottle on the source table complete. The corrected criterion names the <b>separate green stool</b> and requires transfer and release onto it. The trained VLA prompt is unchanged.</p>
<div class="tabs" aria-label="Choose recorded trial"><button id="original" aria-pressed="false">Original criterion · false success</button><button id="corrected" aria-pressed="true">Corrected criterion · waiting then stop</button></div>
<div class="viewer"><div><img id="frame" class="full" alt="Saved ego-camera frame"><div class="controls"><button id="previous" aria-label="Previous frame">←</button><input id="scrubber" type="range" min="0" value="0" aria-label="Choose saved frame"><button id="next" aria-label="Next frame">→</button></div><p id="frame-counter" class="small muted" aria-live="polite"></p></div><div><div id="verdict" class="pill"></div><p id="reason" class="quote"></p><dl><div><dt>Model frame verdict</dt><dd id="raw-status"></dd></div><div><dt>Monitor verdict after two-frame rule</dt><dd id="monitor-status"></dd></div><div><dt>Frame captured at</dt><dd id="captured"></dd></div><div><dt>Reply received at</dt><dd id="decided"></dd></div></dl><div class="notice"><b>Independent truth:</b> every object-contact sample during both missions was with <code>source_table</code>. The bottle was not on the green stool.</div><p id="trial-result" class="small muted"></p></div></div>
<details><summary>What this comparison establishes</summary><p class="small">The original trial contains three frame judgments, two of which falsely say complete. The second false-complete frame satisfied the monitor's confirmation rule. The corrected trial has 44 in-progress replies and a 120.159-second task interruption with planner hold confirmed. These are correlated observations in two trials, not a general reliability estimate. The separate three-frame replay substitutes its first saved frame for the original unsaved initial reference.</p></details></section>
<div class="row"><section class="panel"><h2>Turning remains unaccepted</h2><div>__TURNING__</div><p class="small muted">Threshold: 3° error, held for 0.5 s. Lead limit: 5°. Rate: 10°/s. Deadline: 10 s. The negative crossing trial stops on a lead/rate conflict; the negative near-boundary trial times out.</p><p class="small">Earlier signed turns also failed at 3.314° / 3.761°. Three passing follow-up trials do not establish repeatability.</p></section><section class="panel"><h2>Sample workstation layout</h2><img class="full" src="__SCENE__" alt="Rendered sample scene with G1, source table, cyan bottle and separate green stool"><table><thead><tr><th>Item</th><th>Position from initial G1 pelvis XY</th><th>Height</th></tr></thead><tbody><tr><td>Source table</td><td>50 cm forward, 20 cm left</td><td>80 cm</td></tr><tr><td>Green stool</td><td>35 cm forward, 60 cm right</td><td>80 cm</td></tr><tr><td>Bottle</td><td>45 cm forward, centered laterally</td><td>15 cm tall</td></tr></tbody></table><p class="small muted">This uses your sample-scene option, rather than reconstructing the current ~1 m setup. Right/left are the robot's. Bottle mass (~42 g), appearance and friction are provisional. Mapping, passive support, fingertip collision filtering and camera transport pass; grasp is unvalidated.</p></section></div>
<div class="row" style="margin-top:24px"><section class="panel"><h2>Completion and interruption checks</h2><div class="step"><span class="tag green">PASS</span><span>Policy stall → interruption and confirmed planner hold</span></div><div class="step"><span class="tag green">PASS</span><span>Fresh late result during reset → discarded, no latent publication</span></div><div class="step"><span class="tag green">PASS</span><span>Recorded vision → 8 negatives, 0 false-completes</span></div><div class="step"><span class="tag green">PASS</span><span>Recorded release → one two-frame confirmation</span></div><p class="small muted">The late result is discarded by the paused queue drain; it does not isolate the active epoch comparison. The live corrected run proves a negative case, not successful placement. Stool contact alone cannot certify stable released placement.</p></section><section class="panel"><h2>Training outcome review</h2><p><b>179</b> source episodes → <b>174</b> retained episodes.<br><b>1</b> trained prompt: <code>pick drink bottle and place it on the right table</code></p><div class="reviewbar" aria-label="1 reviewed placement, 1 reviewed failure, 172 unreviewed"><span class="success"></span><span class="failure"></span><span class="unknown"></span></div><div class="legend"><span class="green"><b>1</b> placement</span><span class="red"><b>1</b> failure</span><span class="muted"><b>172</b> unreviewed</span></div><p class="small muted">Episode 18: session-inspected placement onto the user-confirmed destination. Episode 5: user-confirmed unrecovered failure, retained in training. These labels do not estimate dataset success rate.</p><p class="small">Priority review: source episode 14, 544.88 s, contains 9.93% of retained frames and many stationary views. Frame share does not establish optimizer sampling probability or checkpoint causality.</p></section></div>
<section class="panel"><h2>Remaining work</h2><ul><li>Calibrate bottle appearance, mass, friction and workstation geometry; demonstrate grasp and released placement with independent stable top-support evidence.</li><li>Resolve inconsistent negative/wrapped turns while keeping the existing limits.</li><li>Validate fully hidden carried-bottle footage.</li><li>Review the 172 remaining outcomes and idle segments before choosing another training run.</li><li>Run supervised hardware acceptance after the relevant gates and explicit hardware, safety-zone and E-stop readiness confirmation.</li></ul><p class="small muted">Standing reset preserves heading at reset entry. Returning to a saved floor location still needs localization. Dashboard implementation remains assigned to another session.</p></section>
<div class="footer">Generated only from the saved validation artifact set. No API calls, robot connection or live status polling. <a href="../g1_harness_20261002/README.md">Source report</a>. Agent commit d9fe414; native dependency fix 73560de. Source-file hashes are in visualization-data.json.</div>
</main><script>
const data=__DATA__;
let selected='corrected',index=0;
const $=id=>document.getElementById(id);
function render(){
 const t=data.trials[selected],d=t.decisions[index];
 $('original').setAttribute('aria-pressed',String(selected==='original'));
 $('corrected').setAttribute('aria-pressed',String(selected==='corrected'));
 $('frame').src=d.image;
 $('frame').alt=`Saved ${selected} trial ego-camera frame ${index+1}, ${d.captured_s.toFixed(2)} seconds after manipulation start`;
 $('scrubber').max=t.decisions.length-1;$('scrubber').value=index;
 $('previous').disabled=index===0;$('next').disabled=index===t.decisions.length-1;
 $('frame-counter').textContent=`Frame ${index+1} / ${t.decisions.length} · captured at ${d.captured_s.toFixed(2)} s`;
 const falseClaim=d.raw_status==='complete';
 $('verdict').className='pill '+(falseClaim?'red':'blue');
 $('verdict').textContent=falseClaim?'False completion claim — bottle still on source':'In progress — completion not established';
 $('reason').textContent=d.reason;
 $('raw-status').textContent=d.raw_status.replaceAll('_',' ');
 $('monitor-status').textContent=d.monitor_outcome.replaceAll('_',' ');
 $('captured').textContent=d.captured_s.toFixed(2)+' s';
 $('decided').textContent=d.decided_s.toFixed(2)+' s';
 $('trial-result').textContent=selected==='original'?'Coordinator falsely completed, then reset. Independent placement failed. Mission ended at '+t.duration_s.toFixed(3)+' s.':'Coordinator interrupted at '+t.duration_s.toFixed(3)+' s; planner hold confirmed. No automatic retry or reset.';
}
for(const name of ['original','corrected'])$(name).addEventListener('click',()=>{selected=name;index=0;render()});
$('scrubber').addEventListener('input',e=>{index=Number(e.target.value);render()});
$('previous').addEventListener('click',()=>{index=Math.max(0,index-1);render()});
$('next').addEventListener('click',()=>{index=Math.min(data.trials[selected].decisions.length-1,index+1);render()});
render();
</script></body></html>'''


def main():
    manifest = read_json(SOURCE / "artifact-files.sha256.json")
    for name, digest in manifest.items():
        assert hashlib.sha256((SOURCE / name).read_bytes()).hexdigest() == digest, name
    data = {
        "trials": {"original": trial("live-original-criterion"),
                   "corrected": trial("live-green-stool-criterion")},
        "turns": read_json(SOURCE / "turn-diagnosis.json")["trials"],
        "layout": read_json(SOURCE / "scene_layout.json"),
        "software": read_json(SOURCE / "software-checks.json"),
        "training": read_json(SOURCE / "training-review/summary.json"),
    }
    original = data["trials"]["original"]
    corrected = data["trials"]["corrected"]
    assert len(original["decisions"]) == 3
    assert sum(d["raw_status"] == "complete" for d in original["decisions"]) == 2
    assert len(corrected["decisions"]) == 44
    assert all(d["raw_status"] == "in_progress" for d in corrected["decisions"])
    assert corrected["result"]["final_status"]["hold_confirmed"]
    figures(data)
    svg = (OUT / "turning.svg").read_text()
    svg = svg[svg.index("<svg"):]
    svg = svg.replace('width="756pt" height="324pt"', 'width="100%" height="auto"')
    # Restrict the HTML payload to its frame viewer data; never include credentials.
    viewer_data = {"trials": {name: {"duration_s": t["duration_s"], "decisions": t["decisions"]}
                              for name, t in data["trials"].items()}}
    html = (HTML.replace("__OVERVIEW__", data_url(OUT / "results-overview.png"))
            .replace("__SCENE__", data_url(SOURCE / "bottle-scene/overview.png"))
            .replace("__TURNING__", svg)
            .replace("__DATA__", json.dumps(viewer_data).replace("</", "<\\/")))
    (OUT / "report.html").write_text(html)
    for t in data["trials"].values():
        for d in t["decisions"]:
            del d["image"]
    data["source_file_hashes"] = manifest
    data["source_agent_commit"] = "d9fe414"
    data["figure_library"] = f"Matplotlib {matplotlib.__version__}"
    (OUT / "visualization-data.json").write_text(json.dumps(data, indent=2) + "\n")
    print(json.dumps({"html": str(OUT / "report.html"), "original_frames": 3,
                      "corrected_frames": 44, "source_hashes_verified": len(manifest)}))


if __name__ == "__main__":
    main()
