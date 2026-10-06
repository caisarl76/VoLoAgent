"""Compare the existing camera with a URDF-mount proxy at one frozen pose.

The URDF link is not proven to be the RGB optical origin. Intrinsics remain
the original simulator values; this does not establish device calibration.
"""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import cv2
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from gear_sonic.data.robot_model.instantiation.g1 import instantiate_g1_robot_model
from gear_sonic.scripts.run_vla_inference import prepare_observation_from_sensors


def rows(path):
    return [json.loads(s) for s in path.read_text().splitlines()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--urdf", type=Path, required=True)
    parser.add_argument("--trial", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    joint = ET.parse(args.urdf).getroot().find("joint[@name='d435_joint']")
    assert joint is not None and joint.attrib["type"] == "fixed"
    parent = joint.find("parent").attrib["link"]
    origin = joint.find("origin")
    position = np.fromstring(origin.attrib["xyz"], sep=" ")
    rpy = np.fromstring(origin.attrib["rpy"], sep=" ")
    link_rotation = Rotation.from_euler("xyz", rpy).as_matrix()
    # Convention hypothesis: link +X is forward, +Y left, +Z up.
    # MuJoCo camera +X is right, +Y up and -Z forward.
    camera_axes = np.array([[0, 0, -1], [-1, 0, 0], [0, 1, 0]])
    rotation = link_rotation @ camera_axes
    np.testing.assert_allclose(-rotation[:, 2], link_rotation[:, 0])
    quat = Rotation.from_matrix(rotation).as_quat()[[3, 0, 1, 2]]
    events_path = next((args.trial / "mission").glob("*/events.jsonl"))
    started = next(x["at"] for x in rows(events_path) if x["event"] == "skill_start")
    state_path = args.trial / "ground-truth.jsonl"
    object_path = args.trial / "bottle-ground-truth.jsonl"
    source = min(rows(state_path), key=lambda x: abs(x["at"] - started))
    obj = min(rows(object_path), key=lambda x: abs(x["at"] - source["at"]))
    models, arrays, camera_parameters = [], {}, {}
    robot = instantiate_g1_robot_model(waist_location="lower_and_upper_body")
    for name in ["existing", "urdf_mount_proxy"]:
        folder = args.output / name
        folder.mkdir()
        xml = ET.parse(args.scene.parent / "robot.xml")
        camera = next(e for e in xml.getroot().iter("camera") if e.get("name") == "head_camera")
        if name == "urdf_mount_proxy":
            camera.attrib.pop("euler", None)
            camera.set("pos", " ".join(map(str, position)))
            camera.set("quat", " ".join(map(str, quat)))
        xml.write(folder / "robot.xml")
        shutil.copy2(args.scene, folder / "scene.xml")
        model = mujoco.MjModel.from_xml_path(str((folder / "scene.xml").resolve()))
        data = mujoco.MjData(model)
        cam = model.camera("head_camera").id
        assert model.body(model.cam_bodyid[cam]).name == parent
        data.qpos[:7] = source["floating_base_pose"]
        data.qpos[model.jnt_qposadr[model.actuator_trnid[:, 0]]] = np.r_[
            source["body_q"], source["left_hand_q"], source["right_hand_q"]]
        q = model.joint("task_bottle_free").qposadr[0]
        data.qpos[q:q + 7] = [*obj["bottle_position"], 1, 0, 0, 0]
        mujoco.mj_forward(model, data)
        with mujoco.Renderer(model, height=480, width=640) as renderer:
            renderer.update_scene(data, camera="head_camera")
            rgb = renderer.render().copy()
        assert cv2.imwrite(str(folder / "ego_view.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        camera_client = SimpleNamespace(read=lambda rgb=rgb: {
            "images": {"ego_view": rgb}, "timestamps": {"ego_view": 1.0}})
        sensor_state = {key: np.asarray(source[key]) for key in ["body_q", "left_hand_q", "right_hand_q"]}
        sensor_state["base_quat"] = np.asarray(source["floating_base_pose"][3:7])
        subscriber = SimpleNamespace(get_msg=lambda: sensor_state)
        observation = prepare_observation_from_sensors(
            camera_client, subscriber, robot, "pick drink bottle and place it on the right table")
        arrays.update({f"{name}__state__{k}": v for k, v in observation["state"].items()})
        arrays[f"{name}__image"] = observation["video"]["ego_view"]
        camera_parameters[name] = {"position_in_parent_m": model.cam_pos[cam].tolist(),
                                   "quaternion_wxyz": model.cam_quat[cam].tolist(),
                                   "fovy_deg": float(model.cam_fovy[cam])}
        models.append(model)
    invariants = ["body_mass", "body_inertia", "body_ipos", "body_iquat", "geom_type",
                  "geom_size", "geom_pos", "geom_quat", "geom_contype", "geom_conaffinity",
                  "geom_friction", "jnt_qposadr", "actuator_trnid", "actuator_gainprm"]
    for key in invariants:
        np.testing.assert_array_equal(getattr(models[0], key), getattr(models[1], key))
    for key in observation["state"]:
        np.testing.assert_array_equal(arrays[f"existing__state__{key}"],
                                      arrays[f"urdf_mount_proxy__state__{key}"])
    np.savez_compressed(args.output / "inputs.npz", **arrays)
    relative = Rotation.from_quat(models[0].cam_quat[cam][[1, 2, 3, 0]]).inv() * Rotation.from_matrix(rotation)
    result = {"camera_model_user_confirmed": "RealSense D435i, official G1 head mount",
              "robot_actuated": False, "physics_advanced": False, "camera_parent": parent,
              "urdf_joint_origin": origin.attrib, "camera_parameters": camera_parameters,
              "mount_translation_difference_m": float(np.linalg.norm(models[0].cam_pos[cam] - position)),
              "mount_rotation_difference_deg": float(np.degrees(relative.magnitude())),
              "invariant_fields": invariants, "state_inputs_identical": True,
              "state_skew_from_skill_start_s": source["at"] - started,
              "object_state_skew_s": obj["at"] - source["at"],
              "full_scene_calibrated": False, "rgb_intrinsics_available": False,
              "limitations": ["URDF link-to-RGB optical transform is assumed, not measured.",
                              "Original simulator fovy is retained; no nominal D435i field of view is substituted.",
                              "Root/table placement and backdrop still differ from recordings.",
                              "Saved 10 Hz pose is approximate, not an exact policy request."],
              "sources": {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in [Path(__file__), args.scene, args.scene.parent / "robot.xml",
                                    args.urdf, state_path, object_path, events_path]}}
    (args.output / "d435-joint-as-read.urdf").write_text(ET.tostring(joint, encoding="unicode") + "\n")
    (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
