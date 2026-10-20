"""Headless MuJoCo scene for the handover loop: box, bottle and a scripted person.

SIMULATION FIXTURE ONLY; controller communication stays on loopback.
- Box: the scene's 'desk' geoms (top at --box-height-m). They stay 3 m away during
  grounded startup and the start pre-pose. Each `box-restore-N` request brings them
  back only if the conservative arm-to-box gap is positive; box-restore-N.json
  records the result.
- Bottle grasp: a standing bottle is a fixed prop without contacts, because the
  replayed reach sweeps the hand sideways through it (the simulated pelvis turns
  more than the recorded one). It attaches to the right palm when the right hand
  closes within ATTACH_RADIUS_M of it. Opening the hand palm-up (wrist roll > PALM_UP_RAD)
  leaves it resting on the palm. Opening otherwise drops it to physics, with contacts.
- Person: `take-bottle-N` removes a palm-resting bottle; `place-bottle-N` stands a
  new bottle on the box. A bottle stands at --bottle-forward-m/--bottle-left-m in
  the robot's current pelvis heading frame (where the open-loop replay closes the
  hand), and at least at --bottle-min-x-m so it stays on the box.
- bottle-state.json is the oracle's ground truth (20 Hz). ground-truth.jsonl keeps
  the full state. box_contacts lists robot bodies in MuJoCo contact with the box;
  the shared `clearance` distance reads 0.0 falsely for some mesh/box pairs.
"""
import argparse
import json
from pathlib import Path
import signal
import sys
import time

import mujoco
import numpy as np

from gear_sonic.data.robot_model.instantiation.g1 import instantiate_g1_robot_model
from gear_sonic.scripts.run_sim_loop import SimWrapper
from gear_sonic.utils.mujoco_sim.configs import SimLoopConfig

ART = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ART / "g1_handover_20261012_acceptance"), str(ART / "g1_handover_20261013_lift")]
from geometry import closest_bounds_gap  # noqa: E402
from scene import closest_clearance  # noqa: E402

sys.path.insert(0, "/home/jihun/work/.enpire-runtime/g1-bottle-sim-20261006")
from grounded_startup import apply_grounded_startup  # noqa: E402
from scene_fixture import configure_hand_order  # noqa: E402

DESK = ["desk_top", *(f"desk_leg_{i}" for i in range(4))]
AWAY_M = 3.0
ATTACH_RADIUS_M = 0.10
PALM_UP_RAD = 1.2
PARKED = (AWAY_M, 1.0, 0.11)  # Where the person keeps a taken bottle.
WRIST_ROLL = 26  # Motor index of right_wrist_roll_joint.
# right_hand_palm_link is a fixed URDF frame (right_hand_palm_joint), not a MuJoCo body.
PALM_OFFSET = np.array([0.0415, -0.003, 0.0])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--box-height-m", type=float, required=True)
    parser.add_argument("--bottle-forward-m", type=float, required=True)
    parser.add_argument("--bottle-left-m", type=float, required=True)
    parser.add_argument("--bottle-min-x-m", type=float, required=True)
    parser.add_argument("--deadline-s", type=float, default=600.0)
    args = parser.parse_args()
    out = args.output
    config = SimLoopConfig(interface="sim", enable_onscreen=False, enable_offscreen=False,
                           enable_real_device=False, data_collection=False)
    wbc = config.load_wbc_yaml()
    assert wbc["INTERFACE"] == "lo" and wbc["DOMAIN_ID"] == 0
    wbc["ROBOT_SCENE"], wbc["ENV_NAME"] = str(args.scene), config.env_name
    sim = SimWrapper(instantiate_g1_robot_model(), config.env_name, wbc, onscreen=False, offscreen=False).sim
    env = sim.sim_env
    model, data = env.mj_model, env.mj_data
    mapping = configure_hand_order(env)
    joint = model.joint("task_bottle_free")
    qadr, vadr = joint.qposadr[0], joint.dofadr[0]
    palm = model.body("right_wrist_yaw_link").id
    bottle_z = args.box_height_m + 0.10 + 0.001
    desk_geoms = {model.geom(name).id for name in DESK}
    bottle_geoms = [g for g in range(model.ngeom) if model.geom_bodyid[g] == model.jnt_bodyid[joint.id]]
    contacts = {g: (int(model.geom_contype[g]), int(model.geom_conaffinity[g])) for g in bottle_geoms}
    state = dict(automatic_resets=0, startup_resets=0, physics_steps=0)
    fixture = dict(box_away=False, bottle="on_box", rel=None, attach_distance_m=None, taken=0, placed=0,
                   grasps=0, restore_requests=0, events=[])
    applying_startup, sample_at = False, 0.0
    original_step, original_reset = env.sim_step, env.reset
    deadline = time.monotonic() + args.deadline_s

    def event(name, **fields):
        fixture["events"].append(dict(sim_time=float(data.time), event=name, **fields))

    def stop(*_):
        sim._running = False

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)

    def reset():
        state["startup_resets" if applying_startup else "automatic_resets"] += 1
        return original_reset()

    def set_bottle(position, quat=(1.0, 0.0, 0.0, 0.0)):
        data.qpos[qadr:qadr + 3], data.qpos[qadr + 3:qadr + 7] = position, quat
        data.qvel[vadr:vadr + 6] = 0

    def bottle_contacts(enabled):
        for g, (contype, conaffinity) in contacts.items():
            model.geom_contype[g], model.geom_conaffinity[g] = (contype, conaffinity) if enabled else (0, 0)

    def stand_bottle():
        w, x, y, z = data.qpos[3:7]
        yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        c, s = np.cos(yaw), np.sin(yaw)
        forward, left = args.bottle_forward_m, args.bottle_left_m
        fixture["home"] = np.array([max(data.qpos[0] + c * forward - s * left, args.bottle_min_x_m),
                                    data.qpos[1] + s * forward + c * left, bottle_z])
        set_bottle(fixture["home"])

    def shift_desk(dx):
        for name in DESK:
            model.geom_pos[model.geom(name).id][0] += dx

    def palm_pose():
        rotation = data.xmat[palm].reshape(3, 3).copy()
        return data.xpos[palm] + rotation @ PALM_OFFSET, rotation

    def closure():
        return float(np.abs(env.obs["right_hand_q"]).mean())

    def requests(prefix, count):
        return (out / f"{prefix}-{count + 1}").exists()

    def fixture_step():
        bottle = fixture["bottle"]
        position = data.qpos[qadr:qadr + 3].copy()
        if fixture["box_away"]:
            if requests("box-restore", fixture["restore_requests"]):
                fixture["restore_requests"] += 1
                shift_desk(-AWAY_M)
                stand_bottle()
                mujoco.mj_forward(model, data)
                gap = closest_bounds_gap(model, data, ["desk_top"])
                restored = gap["gap_m"] > 0
                if not restored:  # Never drop the box into the arm.
                    shift_desk(AWAY_M)
                    set_bottle(fixture["home"] + np.array([AWAY_M, 0.0, 0.0]))
                fixture["box_away"] = not restored
                (out / f"box-restore-{fixture['restore_requests']}.json").write_text(json.dumps(dict(
                    restored=restored, sim_time=float(data.time), box_gap=gap), indent=2) + "\n")
                event("box_restore", restored=restored, gap_m=gap["gap_m"], bottle=fixture["home"].tolist())
            return
        if bottle == "on_box":
            set_bottle(fixture["home"])
        elif bottle == "taken":
            set_bottle(PARKED)  # Contact-free, so it would otherwise fall forever.
        if bottle in ("in_hand", "on_palm"):
            origin, rotation = palm_pose()
            rel_position, rel_rotation = fixture["rel"]
            quat = np.zeros(4)
            mujoco.mju_mat2Quat(quat, (rotation @ rel_rotation).ravel())
            set_bottle(origin + rotation @ rel_position, quat)
        if bottle == "on_box" and closure() > 0.3:
            origin, rotation = palm_pose()
            distance = float(np.linalg.norm(position - origin))
            if distance < ATTACH_RADIUS_M:
                bottle_rotation = np.zeros(9)
                mujoco.mju_quat2Mat(bottle_rotation, data.qpos[qadr + 3:qadr + 7])
                fixture.update(bottle="in_hand", attach_distance_m=distance,
                               rel=(rotation.T @ (position - origin), rotation.T @ bottle_rotation.reshape(3, 3)))
                fixture["grasps"] += 1
                event("grasp", palm_bottle_distance_m=distance)
        elif bottle == "in_hand" and closure() < 0.2:
            roll = float(env.obs["body_q"][WRIST_ROLL])
            if roll > PALM_UP_RAD:
                fixture["bottle"] = "on_palm"
                event("release_palm_up", wrist_roll_rad=roll)
            else:
                fixture["bottle"] = "dropped"
                bottle_contacts(True)
                event("release_not_palm_up", wrist_roll_rad=roll)
        elif bottle == "on_palm" and requests("take-bottle", fixture["taken"]):
            fixture["taken"] += 1
            fixture["bottle"] = "taken"
            set_bottle(PARKED)
            event("person_took_bottle", cycle=fixture["taken"])
        elif bottle == "taken" and requests("place-bottle", fixture["placed"]):
            fixture["placed"] += 1
            fixture["bottle"] = "on_box"
            stand_bottle()
            bottle_contacts(False)
            event("person_placed_bottle", cycle=fixture["placed"], position=fixture["home"].tolist())
        elif bottle == "dropped" and position[2] < args.box_height_m - 0.05:
            fixture["bottle"] = "fallen"
            event("bottle_fell", position=position.tolist())

    def startup(sim_):
        receipt = apply_grounded_startup(sim_)
        bottle_contacts(False)
        shift_desk(AWAY_M)
        set_bottle((AWAY_M, 0.0, bottle_z))
        mujoco.mj_forward(model, data)
        fixture["box_away"] = True
        return dict(receipt, box_moved_away_m=AWAY_M)

    env.reset = reset
    with (out / "ground-truth.jsonl").open("x") as stream:
        def step():
            nonlocal applying_startup, sample_at
            if (out / "startup-request").exists() and state["startup_resets"] == 0:
                applying_startup = True
                try:
                    receipt = startup(sim)
                finally:
                    applying_startup = False
                (out / "startup-receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
            original_step()
            state["physics_steps"] += 1
            if (out / "startup-receipt.json").exists():
                fixture_step()
            now = time.monotonic()
            if now >= sample_at:
                sample_at = now + 0.05
                origin, _ = palm_pose()
                truth = dict(sim_time=float(data.time), received_monotonic=now, bottle=fixture["bottle"],
                             box_away=fixture["box_away"], bottle_position=data.qpos[qadr:qadr + 3].tolist(),
                             palm_position=origin.tolist(), right_hand_closure=closure(),
                             right_wrist_roll=float(env.obs["body_q"][WRIST_ROLL]),
                             grasps=fixture["grasps"], taken=fixture["taken"], placed=fixture["placed"],
                             attach_distance_m=fixture["attach_distance_m"])
                temporary = out / "bottle-state.json.tmp"
                temporary.write_text(json.dumps(truth))
                temporary.replace(out / "bottle-state.json")
                foot_contacts, box_contacts = [], []
                for contact in data.contact[:data.ncon]:
                    ids = [int(model.geom_bodyid[g]) for g in (contact.geom1, contact.geom2)]
                    names = [model.body(i).name or "" for i in ids]
                    if 0 in ids and any("ankle" in n or "foot" in n for n in names):
                        foot_contacts.append(names)
                    box = [g in desk_geoms for g in (contact.geom1, contact.geom2)]
                    if any(box) and not all(box):
                        box_contacts.append(names[box.index(False)])
                clearance = closest_clearance(model, data, [*DESK, "person_proxy_geom"])
                stream.write(json.dumps(dict(truth, qpos=data.qpos.tolist(), body_q=env.obs["body_q"].tolist(),
                    right_hand_q=env.obs["right_hand_q"].tolist(), band_enabled=bool(env.elastic_band.enable),
                    foot_floor_contacts=foot_contacts, box_contacts=sorted(set(box_contacts)), clearance=clearance, **state), allow_nan=False) + "\n")
                stream.flush()
            if now >= deadline or (out / "stop-simulator").exists():
                stop()
        env.sim_step = step
        (out / "simulator-ready.json").write_text(json.dumps(dict(interface="lo", domain_id=0,
            hand_mapping=mapping, headless=True, mujoco_version=mujoco.__version__), indent=2) + "\n")
        try:
            sim.start()
        finally:
            sim.close()
            (out / "simulator-result.json").write_text(json.dumps(dict(state, fixture_events=fixture["events"]),
                                                                  indent=2) + "\n")


if __name__ == "__main__":
    main()
