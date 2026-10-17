"""Compare recorded/native inputs and scene sensitivity without actuation.

Starts only an owned checkpoint server on GPU 1. Never opens camera, state,
action, DDS or harness command endpoints. Saved simulator state is reconstructed
offline and is approximate (nearest 10 Hz truth sample to a saved monitor frame).
"""

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time
from types import SimpleNamespace

import cv2
import numpy as np
import pyarrow.parquet as pq

from gear_sonic.data.robot_model.instantiation.g1 import instantiate_g1_robot_model
from gear_sonic.scripts.run_vla_inference import prepare_observation_from_sensors
from gr00t.policy.server_client import PolicyClient


PROMPT = "pick drink bottle and place it on the right table"
ORIGIN = Path("/home/jihun/work/GR00T-WholeBodyControl")
DATASET = Path("/mnt/data/jihun/datasets/G1_WBT_GR00T/pnp_bottle_260916")
CHECKPOINT = Path("/mnt/data/jihun/models/pnp_bottle_260916_n17_raw_gpu7/checkpoint-20000")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def observation(image, q, gravity, modality):
    return {
        "video": {"ego_view": image[None, None]},
        "state": {**{key: q[entry["start"]:entry["end"]].astype(np.float32)[None, None]
                     for key, entry in modality["state"].items() if "original_key" not in entry},
                  "projected_gravity": gravity.astype(np.float32)[None, None]},
        "language": {"annotation.human.task_description": [[PROMPT]]},
    }


def from_native(image, state, robot):
    camera = SimpleNamespace(read=lambda: {"images": {"ego_view": image}, "timestamps": {"ego_view": 1.0}})
    subscriber = SimpleNamespace(get_msg=lambda: copy.deepcopy(state))
    return prepare_observation_from_sensors(camera, subscriber, robot, PROMPT)


def state_from_recorded(q, quat, robot):
    return {"body_q": q[robot.get_body_actuated_joint_indices()].copy(),
            "left_hand_q": q[robot.get_hand_actuated_joint_indices("left")].copy(),
            "right_hand_q": q[robot.get_hand_actuated_joint_indices("right")].copy(),
            "base_quat": quat}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--saved-live", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--inputs-only", action="store_true")
    parser.add_argument("--require-parity", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output/'as-run-policy_input_probe.py').write_bytes(Path(__file__).read_bytes())
    robot = instantiate_g1_robot_model(waist_location="lower_and_upper_body")
    modality_path = Path("/tmp/pnp_bottle_260916_gr00t_raw_20260930/meta/modality.json")
    modality = json.loads(modality_path.read_text())
    processor = json.loads((CHECKPOINT/"processor_config.json").read_text())
    state_keys = processor["processor_kwargs"]["modality_configs"]["unitree_g1_sonic"]["state"]["modality_keys"]
    indices = {key: robot.get_joint_group_indices(key) for key in state_keys if key != "projected_gravity"}
    for key, actual in indices.items():
        expected = modality["state"][key]
        assert actual == list(range(expected["start"], expected["end"])), (key, actual, expected)

    parquet = DATASET/"data/chunk-000/episode_000018.parquet"
    video = DATASET/"videos/chunk-000/observation.images.ego_view/episode_000018.mp4"
    table = pq.read_table(parquet)
    columns = {key: np.asarray(table[key].to_pylist()) for key in
               ["observation.state", "observation.root_orientation", "observation.projected_gravity",
                "action.motion_token", "teleop.left_hand_joints", "teleop.right_hand_joints"]}
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS)
    assert fps == 50.0
    cases, parity, targets = {}, [], {}
    for seconds in [5, 10, 15, 20, 30, 38]:
        frame = int(seconds*fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
        ok, bgr = cap.read()
        assert ok
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        q = columns["observation.state"][frame]
        quat = columns["observation.root_orientation"][frame]
        original = observation(rgb, q, columns["observation.projected_gravity"][frame], modality)
        native = from_native(rgb, state_from_recorded(q, quat, robot), robot)
        deltas = {key: float(np.max(np.abs(native["state"][key]-original["state"][key]))) for key in state_keys}
        assert all(delta < 1e-5 for key, delta in deltas.items() if key != "left_hand"), deltas
        if args.require_parity:
            assert all(delta < 1e-5 for delta in deltas.values()), deltas
        assert np.array_equal(native["video"]["ego_view"], original["video"]["ego_view"])
        parity.append({"frame": frame, "seconds": seconds, "max_absolute_state_delta": deltas,
                       "left_hand_middle_index_coupling_delta": deltas["left_hand"],
                       "legacy_overwrite_delta": float(np.max(np.abs(q[24:26]-q[22:24])))})
        name = f"recorded-{seconds:02d}s"
        cases[name] = original
        targets[name] = {key: columns[column][frame:frame+40][None].astype(np.float32) for key, column in
                         [("motion_token", "action.motion_token"), ("left_hand_joints", "teleop.left_hand_joints"),
                          ("right_hand_joints", "teleop.right_hand_joints")]}
        cv2.imwrite(str(args.output/f"{name}.jpg"), bgr)
        if seconds == 10:
            cases["recorded-10s-native-preparation"] = native
    cap.release()

    mission = next((args.saved_live/"mission").iterdir())
    events_path = mission/"events.jsonl"
    events = [json.loads(line) for line in events_path.read_text().splitlines()]
    decision = next(row for row in events if row["event"] == "monitor_decision")
    # The runner hashes the encoded JPEG bytes for its filename.
    snapshots = [row for row in events if row["event"] == "monitor_frame"]
    snap = next(row for row in snapshots if row.get("frame_id") == decision["frame_id"])
    image_path = mission/Path(snap["path"]).name
    sim_rgb = cv2.cvtColor(cv2.imread(str(image_path)), cv2.COLOR_BGR2RGB)
    truth_path = args.saved_live/"ground-truth.jsonl"
    truth = [json.loads(line) for line in truth_path.read_text().splitlines()]
    sample = min(truth, key=lambda row: abs(row["at"]-decision["captured_at"]))
    sim_state = {key: np.asarray(sample[key], dtype=np.float64) for key in ["body_q", "left_hand_q", "right_hand_q"]}
    sim_state["base_quat"] = np.asarray(sample["floating_base_pose"][3:7])
    cases["sim-saved-native"] = from_native(sim_rgb, sim_state, robot)
    for name, state_source, image_source in [
        ("recorded-10s-state-sim-image", "recorded-10s", "sim-saved-native"),
        ("sim-state-recorded-10s-image", "sim-saved-native", "recorded-10s"),
    ]:
        cases[name] = copy.deepcopy(cases[state_source])
        cases[name]["video"] = cases[image_source]["video"]

    summary = {"robot_actuated": False, "server_gpu": 1, "checkpoint": str(CHECKPOINT), "prompt": PROMPT,
               "state_group_indices_match_dataset": True, "state_parity": parity,
               "sim_state_frame_skew_s": sample["at"]-decision["captured_at"],
               "simulation_image_is_saved_monitor_jpeg": True, "same_query_repeat_count": args.repeats,
               "sources": {str(p): digest(p) for p in [Path(__file__), modality_path, parquet, video, events_path,
                                                      truth_path, image_path, CHECKPOINT/"processor_config.json"]}}
    (args.output/"inputs.json").write_text(json.dumps(summary, indent=2)+'\n')
    arrays = {f"{name}__state__{key}": value for name, obs in cases.items() for key, value in obs["state"].items()}
    arrays.update({f"{name}__image": obs["video"]["ego_view"] for name, obs in cases.items()})
    np.savez_compressed(args.output/"inputs.npz", **arrays)
    if args.inputs_only:
        print(json.dumps({"inputs_only": True, "state_parity": parity}), flush=True)
        return

    argv = [str(Path("/home/jihun/work/Isaac-GR00T/.venv/bin/python")),
            str(ORIGIN/"gear_sonic/scripts/serve_pnp_bottle_offline.py"), "--model-path", str(CHECKPOINT),
            "--embodiment-tag", "UNITREE_G1_SONIC", "--device", "cuda:0", "--host", "127.0.0.1", "--port", "15558"]
    outputs, rows = {}, []
    with (args.output/"policy.log").open("w") as log:
        server = subprocess.Popen(argv, stdout=log, stderr=log, start_new_session=True,
                                  env={**os.environ, "PYTHONPATH": "/home/jihun/work/Isaac-GR00T", "CUDA_VISIBLE_DEVICES": "1",
                                       "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "HF_HUB_OFFLINE": "1",
                                       "TRANSFORMERS_OFFLINE": "1", "PYTHONUNBUFFERED": "1", "TOKENIZERS_PARALLELISM": "false"})
        try:
            deadline = time.monotonic()+240
            while "Server is ready and listening" not in (args.output/"policy.log").read_text():
                if server.poll() is not None:
                    raise RuntimeError(f"Owned policy server exited: {server.returncode}")
                if time.monotonic() > deadline:
                    raise TimeoutError("Owned policy server readiness timed out")
                time.sleep(0.2)
            client = PolicyClient(host="127.0.0.1", port=15558, timeout_ms=120000)
            print("Policy-only replay ready", flush=True)
            for name, obs in cases.items():
                for repeat in range(args.repeats):
                    started = time.monotonic()
                    action, _ = client.get_action(obs)
                    row = {"case": name, "repeat": repeat, "latency_s": time.monotonic()-started, "actions": {}}
                    for key, width in [("motion_token", 64), ("left_hand_joints", 7), ("right_hand_joints", 7)]:
                        a = np.asarray(action[key])
                        assert a.shape == (1, 40, width) and np.isfinite(a).all(), (key, a.shape)
                        outputs[f"{name}__{repeat}__{key}"] = a
                        metrics = {"min_per_joint": a.min(axis=(0,1)).tolist(), "max_per_joint": a.max(axis=(0,1)).tolist()}
                        if name in targets:
                            metrics["recorded_target_mae"] = float(np.abs(a-targets[name][key]).mean())
                        row["actions"][key] = metrics
                    rows.append(row)
                    print(json.dumps({"case": name, "repeat": repeat, "latency_s": row["latency_s"]}), flush=True)
            np.savez_compressed(args.output/"actions.npz", **outputs)
            summary["queries"] = rows
        finally:
            if server.poll() is None:
                os.killpg(server.pid, signal.SIGINT)
                try:
                    server.wait(10)
                except subprocess.TimeoutExpired:
                    os.killpg(server.pid, signal.SIGKILL)
                    server.wait()
            summary["owned_server_exit_code"] = server.returncode
            (args.output/"result.json").write_text(json.dumps(summary, indent=2)+'\n')


if __name__ == "__main__":
    main()
