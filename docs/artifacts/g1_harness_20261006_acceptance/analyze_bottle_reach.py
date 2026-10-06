"""Reconstruct sampled hand/bottle distances offline; do not advance physics."""

import argparse
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np


def rows(path):
    return [json.loads(s) for s in path.read_text().splitlines()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trial", type=Path, required=True)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    states = rows(args.trial / "ground-truth.jsonl")
    objects = rows(args.trial / "bottle-ground-truth.jsonl")
    events_path = next((args.trial / "mission").glob("*/events.jsonl"))
    events = rows(events_path)
    start = next(x["at"] for x in events if x["event"] == "skill_start")
    end = next(x["at"] for x in events if x["event"] == "mission_result")
    model = mujoco.MjModel.from_xml_path(str(args.scene.resolve()))
    data = mujoco.MjData(model)
    bottle_q = model.joint("task_bottle_free").qposadr[0]
    bottle_geom = model.geom("task_bottle_geom").id
    hands = {side: [i for i in range(model.ngeom)
                    if ("hand" in model.body(model.geom_bodyid[i]).name
                        or "wrist" in model.body(model.geom_bodyid[i]).name)
                    and model.body(model.geom_bodyid[i]).name.startswith(side)
                    and model.geom_contype[i] != 0] for side in ["left", "right"]}
    assert all(hands.values())
    actuator_q = model.jnt_qposadr[model.actuator_trnid[:, 0]]
    object_times = np.asarray([o["at"] for o in objects])
    records = []
    for state in states:
        if not start <= state["at"] <= end:
            continue
        index = int(np.argmin(np.abs(object_times - state["at"])))
        obj = objects[index]
        assert max(abs(v) for v in obj["bottle_position"][:2]) < 5
        # Only translation was saved. This is an upright-cylinder reconstruction,
        # verified against its stationary source-table trace, not a contact proof.
        data.qpos[:7] = state["floating_base_pose"]
        data.qpos[actuator_q] = np.r_[state["body_q"], state["left_hand_q"], state["right_hand_q"]]
        data.qpos[bottle_q:bottle_q + 7] = [*obj["bottle_position"], 1, 0, 0, 0]
        mujoco.mj_forward(model, data)
        distances = {side: min(float(mujoco.mj_geomDistance(model, data, i, bottle_geom, 2, None))
                              for i in geoms) for side, geoms in hands.items()}
        records.append({"elapsed_s": state["at"] - start,
                        "object_state_skew_s": obj["at"] - state["at"],
                        "base_position": state["floating_base_pose"][:3],
                        "bottle_position": obj["bottle_position"],
                        "left_wrist_position": data.body("left_wrist_yaw_link").xpos.tolist(),
                        "right_wrist_position": data.body("right_wrist_yaw_link").xpos.tolist(),
                        "hand_bottle_collision_distance_m": distances,
                        "body_q": state["body_q"],
                        "left_hand_q": state["left_hand_q"], "right_hand_q": state["right_hand_q"]})
    assert records
    result = {"robot_actuated": False, "physics_advanced": False,
              "reconstructed_samples": len(records), "sample_rate_hz": 10,
              "bottle_orientation_assumed_upright": True,
              "maximum_object_state_skew_s": max(abs(r["object_state_skew_s"]) for r in records),
              "minimum_sampled_hand_bottle_distance_m": {
                  side: min(r["hand_bottle_collision_distance_m"][side] for r in records)
                  for side in hands},
              "minimum_distance_sample_s": {
                  side: min(records, key=lambda r: r["hand_bottle_collision_distance_m"][side])["elapsed_s"]
                  for side in hands},
              "bottle_position_range_m": np.ptp([r["bottle_position"] for r in records], axis=0).tolist(),
              "base_position_range_m": np.ptp([r["base_position"] for r in records], axis=0).tolist(),
              "limitations": ["Robot and object samples are asynchronous; exact capture-state parity is not claimed.",
                              "10 Hz distances can miss intermediate minima and are not contact acceptance evidence.",
                              "Assumed object orientation is suitable only for this stationary bottle trace."],
              "sources": {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in [Path(__file__), args.scene, args.scene.parent / "robot.xml",
                                    args.trial / "ground-truth.jsonl",
                                    args.trial / "bottle-ground-truth.jsonl", events_path]}}
    (args.output / "samples.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
    (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
