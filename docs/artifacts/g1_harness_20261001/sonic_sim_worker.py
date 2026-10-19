"""Headless loopback simulator with ground-truth samples and band release.

Only sonic_sim_probe.py launches this worker. The release file switches off
the simulator's virtual support after the controller has started.
"""

import argparse
import json
import math
from pathlib import Path
import time

import mujoco

from gear_sonic.data.robot_model.instantiation.g1 import instantiate_g1_robot_model
from gear_sonic.scripts.run_sim_loop import SimWrapper
from gear_sonic.utils.mujoco_sim.configs import SimLoopConfig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--initial-yaw-deg", type=float, default=0.0)
    args = parser.parse_args()
    config = SimLoopConfig(interface="sim", enable_onscreen=False, enable_offscreen=False)
    wbc_config = config.load_wbc_yaml()
    assert wbc_config["INTERFACE"] == "lo"
    wbc_config["ENV_NAME"] = config.env_name
    sim = SimWrapper(
        instantiate_g1_robot_model(), config.env_name, wbc_config,
        onscreen=False, offscreen=False,
    ).sim
    env = sim.sim_env
    # Rotate only the simulated floating base before the first physics step.
    # This exercises the actual telemetry/reference conversion across +/-pi.
    half_yaw = math.radians(args.initial_yaw_deg) * 0.5
    env.mj_data.qpos[3:7] = [math.cos(half_yaw), 0.0, 0.0, math.sin(half_yaw)]
    mujoco.mj_forward(env.mj_model, env.mj_data)
    if args.initial_yaw_deg:
        # The temporary startup spring normally torques toward world yaw zero.
        # Align only its orientation reference with this test's initial heading;
        # otherwise a 179-degree spawn receives a large artificial yaw impulse.
        original_band_advance = env.elastic_band.Advance
        c, s = math.cos(half_yaw), math.sin(half_yaw)
        cy, sy = math.cos(2 * half_yaw), math.sin(2 * half_yaw)

        def aligned_band(pose):
            relative = pose.copy()
            w, x, y, z = pose[3:7]
            relative[3:7] = [c*w + s*z, c*x + s*y, c*y - s*x, c*z - s*w]
            vx, vy = pose[10:12]
            relative[10:12] = [cy*vx + sy*vy, -sy*vx + cy*vy]
            wrench = original_band_advance(relative)
            tx, ty = wrench[3:5]
            wrench[3:5] = [cy*tx - sy*ty, sy*tx + cy*ty]
            return wrench

        env.elastic_band.Advance = aligned_band
    original_step = env.sim_step
    release_file = args.output / "release-band"
    sample_at = 0.0
    with (args.output / "ground-truth.jsonl").open("w") as stream:
        def step():
            nonlocal sample_at
            if release_file.exists() and env.elastic_band.enable:
                env.elastic_band.enable = False
                print("Diagnostic: virtual support released", flush=True)
            original_step()
            now = time.monotonic()
            if now < sample_at:
                return
            sample_at = now + 0.1
            stream.write(json.dumps({
                "at": now,
                "sim_time": float(env.mj_data.time),
                "band_enabled": bool(env.elastic_band.enable),
                "floating_base_pose": env.obs["floating_base_pose"].tolist(),
                "body_q": env.obs["body_q"].tolist(),
                "left_hand_q": env.obs["left_hand_q"].tolist(),
                "right_hand_q": env.obs["right_hand_q"].tolist(),
                "body_target": [m.q for m in sim.unitree_bridge.low_cmd.motor_cmd[:29]],
                "left_hand_target": [m.q for m in sim.unitree_bridge.left_hand_cmd.motor_cmd],
                "right_hand_target": [m.q for m in sim.unitree_bridge.right_hand_cmd.motor_cmd],
            }) + "\n")
            stream.flush()
        env.sim_step = step
        sim.start()


if __name__ == "__main__":
    main()
