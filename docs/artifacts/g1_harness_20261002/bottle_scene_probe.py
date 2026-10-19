"""Build and verify a provisional bottle scene; no DDS or policy commands.

The robot comes from the SONIC scene without joint/actuator/camera changes.
Geometry dimensions are assumptions, not a reconstruction of the real station.
Contact checks freeze the robot and do not demonstrate grasping or placement.
"""

import argparse
import hashlib
import json
from pathlib import Path
import time
import xml.etree.ElementTree as ET

import cv2
import msgpack
import mujoco
import numpy as np
import zmq

from gear_sonic.camera.sensor_server import ImageMessageSchema
from gear_sonic.utils.mujoco_sim.base_sim import DefaultEnv
from gear_sonic.utils.mujoco_sim.configs import SimLoopConfig
from gear_sonic.utils.mujoco_sim.image_publish_utils import ImagePublishProcess


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--native", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--camera-port", type=int, default=11555)
    parser.add_argument("--layout", type=Path, default=Path(__file__).with_name("scene_layout.json"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    source = args.native / "gear_sonic/data/robot_model/model_data/g1"
    original = source / "scene_43dof.xml"
    robot_source = source / "g1_29dof_with_hand.xml"
    result = {"outcome": "failed", "robot_connection": False, "dds_started": False,
              "checkpoint_used": False, "placement_validated": False,
              "geometry_measured": False, "native_checkout": str(args.native),
              "sources": {str(p): digest(p) for p in (original, robot_source)},
              "camera_port": args.camera_port}
    renderer = publisher = context = subscriber = None
    try:
        layout = json.loads(args.layout.read_text())
        table, stool, obj = (layout[key] for key in ("source_table", "destination_stool", "bottle"))
        def vector(values):
            return " ".join(str(float(value)) for value in values)

        result["layout"] = layout
        result["layout_sha256"] = digest(args.layout)
        (args.output / "layout.json").write_text(json.dumps(layout, indent=2) + "\n")
        robot = ET.parse(robot_source)
        robot.getroot().find("compiler").set("meshdir", str(source / "meshes"))
        robot.write(args.output / "robot.xml")
        scene = ET.parse(original)
        root = scene.getroot()
        root.set("model", "g1_sonic_bottle_provisional")
        root.find("include").set("file", "robot.xml")
        world = root.find("worldbody")
        ET.SubElement(world, "geom", name="source_table", type="box",
                      pos=vector([*table["center_xy"], table["height"]/2]),
                      size=vector([*table["half_size_xy"], table["height"]/2]), rgba="0.7 0.6 0.5 1")
        ET.SubElement(world, "geom", name="right_green_stool", type="cylinder",
                      pos=vector([*stool["center_xy"], stool["height"]/2]),
                      size=vector([stool["radius"], stool["height"]/2]), rgba="0.15 0.65 0.15 1")
        source_position = [*obj["source_xy"], table["height"] + obj["half_height"] + 0.015]
        destination_position = [*stool["center_xy"], stool["height"] + obj["half_height"] + 0.015]
        bottle = ET.SubElement(world, "body", name="task_bottle", pos=vector(source_position))
        ET.SubElement(bottle, "freejoint", name="task_bottle_free")
        ET.SubElement(bottle, "geom", name="task_bottle_geom", type="cylinder",
                      size=vector([obj["radius"], obj["half_height"]]), density=str(obj["density"]), rgba="0.3 0.7 0.8 1",
                      friction="0.95 0.03 0.003", solref="0.01 1", condim="4")
        scene_path = args.output / "scene.xml"
        scene.write(scene_path)
        config = SimLoopConfig(interface="sim", enable_onscreen=False, enable_offscreen=False)
        wbc = config.load_wbc_yaml()
        wbc.update(ROBOT_SCENE=str(scene_path), ENABLE_ELASTIC_BAND=False)
        env = DefaultEnv(wbc, onscreen=False, offscreen=False)
        model, data = env.mj_model, env.mj_data
        reference = mujoco.MjModel.from_xml_path(str(original))
        assert (model.nq, model.nv, model.nu) == (57, 55, 43)
        assert [model.actuator(i).name for i in range(model.nu)] == [
            reference.actuator(i).name for i in range(reference.nu)]
        np.testing.assert_array_equal(model.actuator_trnid, reference.actuator_trnid)
        np.testing.assert_array_equal(model.jnt_qposadr[:reference.njnt], reference.jnt_qposadr)
        np.testing.assert_array_equal(model.jnt_dofadr[:reference.njnt], reference.jnt_dofadr)
        for indices in (env.body_joint_index, env.left_hand_index, env.right_hand_index):
            np.testing.assert_array_equal(model.jnt_qposadr[indices], indices + env.qpos_offset - 1)
        camera = model.camera("head_camera").id
        ref_camera = reference.camera("head_camera").id
        for field in ("cam_bodyid", "cam_pos", "cam_quat", "cam_fovy"):
            np.testing.assert_array_equal(getattr(model, field)[camera], getattr(reference, field)[ref_camera])
        result["mapping"] = {"outcome": "passed", "nq": model.nq, "nv": model.nv, "nu": model.nu,
                             "body_joints": len(env.body_joint_index), "left_hand_joints": len(env.left_hand_index),
                             "right_hand_joints": len(env.right_hand_index),
                             "camera_preserved": True,
                             "right_hand_actuators": [model.actuator(i).name for i in range(36, 43)]}
        bottle_joint = model.joint("task_bottle_free")
        qa, va = bottle_joint.qposadr[0], bottle_joint.dofadr[0]
        bottle_geom = model.geom("task_bottle_geom").id
        # Kinematic robot for passive object/contact checks, not controlled motion.
        robot_qpos = data.qpos[:reference.nq].copy()
        robot_qpos[2] = 0.786

        def place(position):
            data.qpos[:reference.nq] = robot_qpos
            data.qvel[:] = 0
            data.qpos[qa:qa+7] = [*position, 1, 0, 0, 0]
            mujoco.mj_forward(model, data)

        def contacts():
            names = []
            for c in data.contact:
                pair = (c.geom1, c.geom2)
                if bottle_geom in pair:
                    other = pair[1] if pair[0] == bottle_geom else pair[0]
                    names.append(model.geom(other).name or model.body(model.geom_bodyid[other]).name)
            return names

        result["passive_contacts"] = {}
        for name, position, surface in (
            ("source", source_position, "source_table"),
            ("destination", destination_position, "right_green_stool"),
        ):
            place(position)
            for _ in range(round(2 / model.opt.timestep)):
                data.qpos[:reference.nq] = robot_qpos
                data.qvel[:reference.nv] = 0
                mujoco.mj_step(model, data)
            measured = data.qpos[qa:qa+3].copy()
            touching = contacts()
            assert surface in touching, (name, touching)
            assert np.linalg.norm(data.qvel[va:va+6]) < 0.01
            assert np.linalg.norm(measured[:2] - position[:2]) < 0.01
            result["passive_contacts"][name] = {"outcome": "passed", "position": measured.tolist(),
                                                  "contacts": touching, "velocity_norm": float(np.linalg.norm(data.qvel[va:va+6]))}
        # Test collision filtering for each hand, by deliberately overlapping
        # the bottle with a fingertip. This does not test grasp stability.
        result["hand_collision_filter"] = {}
        for side in ("left", "right"):
            place(source_position)
            finger = model.body(f"{side}_hand_index_1_link").id
            place(data.xpos[finger] + np.array([0.01, 0, 0]))
            touching = contacts()
            assert any(side + "_hand" in name for name in touching), touching
            result["hand_collision_filter"][side] = touching

        place([*obj["source_xy"], table["height"] + obj["half_height"] + 0.001])
        renderer = mujoco.Renderer(model, height=480, width=640)
        renderer.update_scene(data, camera="head_camera")
        ego = renderer.render().copy()
        assert np.std(ego) > 10
        cv2.imwrite(str(args.output / "ego_view.png"), cv2.cvtColor(ego, cv2.COLOR_RGB2BGR))
        overview = mujoco.MjvCamera()
        overview.lookat[:] = [0.25, -0.3, 0.65]
        overview.distance, overview.azimuth, overview.elevation = 2.8, -125, -25
        renderer.update_scene(data, camera=overview)
        cv2.imwrite(str(args.output / "overview.png"), cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR))

        # Real native MuJoCo image subprocess -> actual camera decoder schema.
        context = zmq.Context()
        subscriber = context.socket(zmq.SUB)
        subscriber.setsockopt(zmq.LINGER, 0)
        subscriber.setsockopt(zmq.SUBSCRIBE, b"")
        subscriber.connect(f"tcp://127.0.0.1:{args.camera_port}")
        publisher = ImagePublishProcess({"ego_view": {"height": 480, "width": 640}},
                                        image_dt=1/30, zmq_port=args.camera_port)
        publisher.start_process()
        timestamps = []
        deadline = time.monotonic() + 15
        while len(timestamps) < 3 and time.monotonic() < deadline:
            publisher.update_shared_memory({"ego_view_image": ego})
            if subscriber.poll(100):
                message = ImageMessageSchema.deserialize(msgpack.unpackb(subscriber.recv(), raw=False))
                image = message.images["ego_view"]
                assert image.shape == (480, 640, 3) and image.dtype == np.uint8
                assert np.mean(np.abs(image.astype(float)-ego)) < 8
                timestamp = message.timestamps["ego_view"]
                assert abs(time.time() - timestamp) < 1
                if not timestamps or timestamp > timestamps[-1]:
                    timestamps.append(timestamp)
        assert len(timestamps) == 3, "Three advancing native camera messages required"
        result["camera_transport"] = {"outcome": "passed", "frames": 3, "timestamps": timestamps,
                                      "shape": [480, 640, 3], "camera": "head_camera"}
        result["outputs"] = {p.name: digest(p) for p in args.output.iterdir() if p.is_file()}
        result["outcome"] = "passed_scene_smoke_only"
    except Exception as exc:
        result["reason"] = str(exc)
        raise
    finally:
        if publisher is not None:
            publisher.stop()
        if subscriber is not None:
            subscriber.close()
        if context is not None:
            context.term()
        if renderer is not None:
            renderer.close()
        (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result))


if __name__ == "__main__":
    main()
