"""Where real-robot handover episodes end relative to the measured desk top.

Usage: check_offer_heights.py  (reads the s07/s08 eval drafts read-only; writes results.json here)

The recorded state has both arms and hands but no legs, waist or pelvis tilt.
This applies only the ready-return "hand above desk top" test from native
ready_clearance.py, assuming:
- the last frame (operator stop, after the hand-off) as the end of the offer;
- an upright pelvis and a zero waist;
- a few pelvis heights below an 0.85 m desk.
Run with the native worktree's Python.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from gear_sonic.data.robot_model.instantiation.g1 import instantiate_g1_robot_model
from gear_sonic.utils.inference import ready_clearance as rc

ROOT = Path(__file__).parent
DRAFTS = Path("/mnt/data/jihun/g1_eval_sessions/_visualizer/workspace/drafts")
SESSIONS = {"s07": "0f94fe9ad104498b947eee25b9dcc4ca", "s08": "2ee431e22b504954a811b051e5bae108"}
DESK_TOP_M = 0.85
PELVIS_HEIGHTS_M = (0.74, 0.76, 0.787)  # Robot's measured range, and the simulator's standing height.


def main():
    model = instantiate_g1_robot_model(waist_location="lower_and_upper_body")
    names = model.joint_names
    arm_motors = [names[i] for i in model.get_body_actuated_joint_indices()][15:29]
    right_hand = [names[i] for i in model.get_hand_actuated_joint_indices("right")]
    boxes = rc.right_arm_link_boxes(model)
    report = {}
    for session, draft in SESSIONS.items():
        meta = DRAFTS / draft / "meta"
        info = json.loads((meta / "info.json").read_text())
        index = {n: i for i, n in enumerate(info["features"]["observation.state"]["names"])}
        labels = json.loads((meta / "eval_labels.json").read_text())["episodes"]
        model_path = json.loads((meta / "eval_session.json").read_text())["model"]
        frames = pd.read_parquet(DRAFTS / draft / "data/chunk-000/file-000.parquet")
        episodes = {}
        for episode, rows in frames.groupby("episode_index"):
            last = np.stack(rows["observation.state"].to_numpy())[-1]
            body = np.zeros(29)
            body[15:29] = [last[index[n]] for n in arm_motors]
            hand = np.array([last[index[n]] for n in right_hand])
            level = boxes(body, hand)
            lowest = float(min(level[i][0][2] for i in rc._HAND_LINKS))
            episodes[str(episode)] = dict(
                verdict=labels[str(episode)]["vlm_verdict"], frames=len(rows),
                lowest_hand_above_pelvis_m=lowest,
                hand_minus_desk_top_m={f"{p:.3f}": lowest - (DESK_TOP_M - p) for p in PELVIS_HEIGHTS_M},
            )
        report[session] = dict(draft=str(DRAFTS / draft), model=model_path, episodes=episodes)
    (ROOT / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    for session, data in report.items():
        print(session, data["model"].split("/g1_models/")[-1])
        for episode, e in data["episodes"].items():
            cells = " ".join(f"{p}:{d * 1000:+.0f}mm" for p, d in e["hand_minus_desk_top_m"].items())
            print(f"  ep{episode} {e['verdict']:<7} lowest {e['lowest_hand_above_pelvis_m']:+.3f} m | {cells}")


if __name__ == "__main__":
    main()
