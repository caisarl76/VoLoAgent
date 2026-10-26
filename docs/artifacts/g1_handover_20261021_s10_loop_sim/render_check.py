"""Side-by-side of the simulated D435 head view and a real s10 head frame (camera and scene fidelity check).

Usage (simulator Python, MUJOCO_GL=egl, from g1_handover_20261012_acceptance for scene.py):
    render_check.py GROUND_TRUTH.jsonl SESSION_RAW_DIR OUT.jpg [--box-front-m 0.28]

Renders scene_s10.restyle_scene's d435_head camera at the last standing sample before the box arrives in a
20261019 loop run (the recorded start pose), with the case and bottle placed as run_s10_loop.py places them.
"""
import argparse
import json
from pathlib import Path
import sys

import cv2
import mujoco
import numpy as np

ROOT = Path(__file__).parent
sys.path[:0] = [str(ROOT.parent / "g1_handover_20261012_acceptance"), str(ROOT)]
from scene import build_scene  # noqa: E402
from scene_s10 import BOTTLE_HALF_HEIGHT_M, BOTTLE_PELVIS_M, restyle_scene  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("ground_truth", type=Path)
    parser.add_argument("session", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--box-front-m", type=float, default=0.28)
    parser.add_argument("--box-height-m", type=float, default=0.845)
    a = parser.parse_args()
    scene_path = a.out.with_suffix(".xml")
    build_scene(scene_path, desk_front_m=a.box_front_m, desk_height_m=a.box_height_m)
    restyle_scene(scene_path, a.box_front_m, a.box_height_m, json.loads((ROOT / "d435i_head.json").read_text()))
    model = mujoco.MjModel.from_xml_path(str(scene_path))
    data = mujoco.MjData(model)
    rows = [json.loads(line) for line in a.ground_truth.open()]
    restarts = [i for i in range(1, len(rows)) if rows[i]["sim_time"] < rows[i - 1]["sim_time"] - 1]
    standing = [r for r in rows[restarts[-1]:] if r["box_away"]][-1]
    data.qpos[:] = standing["qpos"]
    w, x, y, z = data.qpos[3:7]
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    forward, left = BOTTLE_PELVIS_M
    adr = model.joint("task_bottle_free").qposadr[0]
    data.qpos[adr:adr + 3] = [data.qpos[0] + np.cos(yaw) * forward - np.sin(yaw) * left,
                              data.qpos[1] + np.sin(yaw) * forward + np.cos(yaw) * left,
                              a.box_height_m + BOTTLE_HALF_HEIGHT_M + 0.001]
    data.qpos[adr + 3:adr + 7] = [1, 0, 0, 0]
    mujoco.mj_forward(model, data)
    renderer = mujoco.Renderer(model, 480, 640)
    renderer.update_scene(data, camera="d435_head")
    sim = renderer.render()  # RGB
    frames = np.load(a.session / "trial_000_frames.npz")
    jpeg, offsets = frames["jpeg"], frames["offsets"]
    real = cv2.imdecode(jpeg[offsets[0]:offsets[1]], cv2.IMREAD_COLOR)  # As received: RGB-ordered bytes.
    side = np.hstack([cv2.cvtColor(sim, cv2.COLOR_RGB2BGR), cv2.cvtColor(real, cv2.COLOR_RGB2BGR)])
    cv2.putText(side, "sim d435_head", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    cv2.putText(side, "real s10 trial 0, first frame", (650, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    cv2.imwrite(str(a.out), side)
    print(a.out, "standing sample at sim time", standing["sim_time"])


if __name__ == "__main__":
    main()
