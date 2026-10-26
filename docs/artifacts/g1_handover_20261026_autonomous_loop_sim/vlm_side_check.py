"""Side check: ask the real local Qwen3-VL server the two re-arm prompts on rendered simulator frames.

Usage: vlm_side_check.py RUN_DIR [--base-url http://127.0.0.1:8301/v1]   (VoLoAgent venv, server already up)

Frames, labelled by the oracle's scene truth at the time the runner fetched them (evidence jpgs) or by the
simulator's ground truth (trial frames npz, nearest 20 Hz sample):
- upright: bottle standing on the case, no hand (bottle-step frames the oracle answered yes, leftover frames);
- empty:   empty case (empty-step frames the oracle answered yes);
- held:    the right hand holds the bottle (trial frames while the fixture state is in_hand).
The prompts are vlm_orchestrator.harness.g1.rearm_monitor's PROMPT and EMPTY_PROMPT, sent image first as
RearmMonitor sends them; no tuning. Writes RUN/vlm-side-check.json and the frames to RUN/vlm-side-check/.
"""
import argparse
import base64
import bisect
import json
from pathlib import Path
import time

import numpy as np
from openai import OpenAI

from vlm_orchestrator.harness.g1.rearm_monitor import EMPTY_PROMPT, PROMPT

PER_CLASS = 3


def jsonl(path):
    return [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]


def spread(items, n):
    return [items[i] for i in np.linspace(0, len(items) - 1, n).round().astype(int)] if len(items) > n else items


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("--base-url", default="http://127.0.0.1:8301/v1")
    args = parser.parse_args()
    run, session = args.run.resolve(), args.run.resolve() / "session"
    picks = {"upright": [], "empty": [], "held": []}
    for path in sorted(session.glob("trial_*_evidence.jsonl")):
        for e in jsonl(path):
            if e["kind"] != "rearm_decision" or not e.get("frame_path"):
                continue
            jpg = (session / e["frame_path"]).read_bytes()
            if e["step"] == "bottle" and e["outcome"] == "yes":
                picks["upright"].append((jpg, f"{e['frame_path']} (bottle step, oracle yes)"))
            elif e["step"] == "empty" and e["outcome"] == "yes":
                picks["empty"].append((jpg, f"{e['frame_path']} (empty step, oracle yes)"))
            elif e["step"] == "empty" and e["outcome"] == "no" and e.get("answers", {}).get("upright_check", {}) \
                    .get("parsed", {}).get("bottle_upright_on_case"):
                picks["upright"].append((jpg, f"{e['frame_path']} (leftover bottle, empty step, oracle no)"))
    events = jsonl(run / "events.jsonl")
    wall_minus_mono = np.median([e["wall"] - e["at"] for e in events])
    truth = jsonl(run / "ground-truth.jsonl")
    times = [r["received_monotonic"] for r in truth]
    for path in sorted(session.glob("trial_*_frames.npz")):
        with np.load(path) as f:
            for i, t in enumerate(f["t_recv"]):
                k = min(bisect.bisect_left(times, t - wall_minus_mono), len(truth) - 1)
                if truth[k]["bottle"] == "in_hand":
                    picks["held"].append((f["jpeg"][f["offsets"][i]:f["offsets"][i + 1]].tobytes(),
                                          f"{path.name}[{i}] (fixture in_hand)"))
    api = OpenAI(api_key="local", base_url=args.base_url, timeout=30, max_retries=0)
    out_dir = run / "vlm-side-check"
    out_dir.mkdir(exist_ok=True)
    rows = []
    for label, items in picks.items():
        for n, (jpg, source) in enumerate(spread(items, PER_CLASS)):
            (out_dir / f"{label}_{n}.jpg").write_bytes(jpg)
            url = "data:image/jpeg;base64," + base64.b64encode(jpg).decode()
            row = dict(label=label, frame=f"{label}_{n}.jpg", source=source)
            for name, prompt in (("upright_check", PROMPT), ("object_on_case", EMPTY_PROMPT)):
                start = time.monotonic()
                reply = api.chat.completions.create(model="qwen3-vl-2b", temperature=0.0, messages=[
                    {"role": "user", "content": [{"type": "image_url", "image_url": {"url": url}},
                                                 {"type": "text", "text": prompt}]}]).choices[0].message.content
                row[name] = dict(raw=reply, seconds=round(time.monotonic() - start, 2))
            try:
                up = json.loads((r := row["upright_check"]["raw"])[r.index("{"):r.rindex("}") + 1])
                obj = json.loads((r := row["object_on_case"]["raw"])[r.index("{"):r.rindex("}") + 1])
                row["bottle_step_yes"] = bool(up["bottle_upright_on_case"] and not up["hand_at_bottle"])
                row["empty_step_yes"] = bool(not up["bottle_upright_on_case"] and not up["hand_at_bottle"]
                                             and not obj["object_on_case"])
                row["agrees_with_truth"] = {"upright": row["bottle_step_yes"] and not row["empty_step_yes"],
                                            "empty": row["empty_step_yes"] and not row["bottle_step_yes"],
                                            "held": not row["bottle_step_yes"] and not row["empty_step_yes"]}[label]
            except (ValueError, KeyError, TypeError):
                row["agrees_with_truth"] = None
            rows.append(row)
            print(json.dumps(row))
    summary = dict(base_url=args.base_url, available={k: len(v) for k, v in picks.items()}, rows=rows,
                   agree=sum(1 for r in rows if r["agrees_with_truth"]), total=len(rows))
    (run / "vlm-side-check.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: summary[k] for k in ("available", "agree", "total")}))


if __name__ == "__main__":
    main()
