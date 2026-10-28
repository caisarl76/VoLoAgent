"""Render the d435_head view at the loop's ready hold (sim-03 ground truth, bottle on the case) with the right arm
shifted outward (shoulder roll), to find where the right gripper leaves the image. Geometry only (kinematic qpos)."""
import gzip, json, sys
from pathlib import Path
import mujoco, numpy as np
ART = Path(sys.argv[1]); OUT = Path(sys.argv[2])
sys.path[:0] = [str(ART / "g1_handover_20261012_acceptance"), str(ART / "g1_handover_20261026_autonomous_loop_sim")]
from scene import build_scene
from scene_s10 import restyle_scene
OUT.mkdir(parents=True, exist_ok=True)
xml = OUT / "scene.xml"
build_scene(xml, desk_front_m=0.28, desk_height_m=0.845)
restyle_scene(xml, 0.28, 0.845, json.loads((ART / "g1_handover_20261026_autonomous_loop_sim/d435i_head.json").read_text()))
model = mujoco.MjModel.from_xml_path(str(xml)); data = mujoco.MjData(model)
rows = [json.loads(l) for l in gzip.open(ART / "g1_handover_20261026_autonomous_loop_sim/runs/sim-03/ground-truth.jsonl.gz")]
ready = [r for r in rows if r["placed"] and r["grasps"] == 0 and not r["box_away"]][20]
R = ["right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint", "right_elbow_joint",
     "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint"]
renderer = mujoco.Renderer(model, 480, 640)
import cv2
hand_geoms = [g for g in range(model.ngeom) if "right" in (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[g]) or "")
              and any(k in mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[g]) for k in ("hand", "wrist", "thumb", "index", "middle"))]
results = {}
import itertools
J = {k: model.joint(k).qposadr[0] for k in R}
def visible(deltas):
    data.qpos[:] = ready["qpos"]
    for k, v in deltas.items(): data.qpos[J[k]] += v
    mujoco.mj_forward(model, data)
    renderer.enable_segmentation_rendering(); renderer.update_scene(data, camera="d435_head"); seg = renderer.render(); renderer.disable_segmentation_rendering()
    return int(np.isin(seg[..., 0], hand_geoms).sum())
C = {
 "A_yaw-0.40": {"right_shoulder_yaw_joint": -0.40},
 "B_roll-0.15_yaw-0.40": {"right_shoulder_roll_joint": -0.15, "right_shoulder_yaw_joint": -0.40},
 "C_pitch-0.20_wyaw-0.50": {"right_shoulder_pitch_joint": -0.20, "right_wrist_yaw_joint": -0.50},
}
margin = {"A_yaw-0.40": {"right_shoulder_yaw_joint": -0.30}, "B_roll-0.15_yaw-0.40": {"right_shoulder_roll_joint": -0.15, "right_shoulder_yaw_joint": -0.30},
          "C_pitch-0.20_wyaw-0.50": {"right_shoulder_pitch_joint": -0.20, "right_wrist_yaw_joint": -0.40}}
results = {}
for name, d in C.items():
    vis = visible(d); vis_m = visible(margin[name])
    import cv2
    renderer.update_scene(data, camera="d435_head")
    data.qpos[:] = ready["qpos"]
    for k, v in d.items(): data.qpos[J[k]] += v
    mujoco.mj_forward(model, data); renderer.update_scene(data, camera="d435_head"); cv2.imwrite(str(OUT / f"{name}.jpg"), cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR))
    # a wide view to see where the hand went
    cam = mujoco.MjvCamera(); cam.lookat[:] = data.qpos[:3] + [0.35, -0.1, 0.25]; cam.distance, cam.azimuth, cam.elevation = 1.6, 200, -20
    renderer.update_scene(data, camera=cam); cv2.imwrite(str(OUT / f"{name}_wide.jpg"), cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR))
    q = [round(float(data.qpos[J[j]]), 3) for j in R]
    results[name] = {"deltas": d, "hand_pixels": vis, "hand_pixels_0.1rad_less": vis_m, "right_arm_q_sim": q}
    print(name, results[name])
(OUT / "results.json").write_text(json.dumps(results, indent=1))
