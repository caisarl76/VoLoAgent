"""Extract one recorded handover cycle as the loop simulation's stand-in policy.

Usage (native worktree Python): build_replay.py  -> replay-cycle.npz, replay-cycle.json, start-pose.json
From handover_bottle_261019 episode 6, cycle 3: the last open-hand frame within
0.25 rad of the median palm-in ready pose before the grasp (so the replay includes
the reach), through 0.6 s after the palm-up release. The reach lasts 2.4 s. Also records the palm
position at the grasp (level pelvis frame, plus pelvis height). The simulated
bottle stands where the replayed hand closes in simulation instead (run_loop_probe.py).
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from gear_sonic.data.robot_model.instantiation.g1 import instantiate_g1_robot_model
from gear_sonic.utils.inference import ready_clearance as rc
from gear_sonic.utils.teleop.xr_upperbody_bridge import G1_UPPER_BODY_JOINT_INDICES

ROOT = Path(__file__).parent
DATASET = Path("/mnt/data/jihun/datasets/G1_WBT_GR00T/handover_bottle_261019")
EPISODE, CYCLE, FPS = 6, 2, 50  # Zero-based cycle index: the third cycle.
# Episodes whose initial standing frames are still (5 has none; 7 starts moving).
START_EPISODES = (2, 3, 4, 6)
# Median palm-in ready pose between handover cycles of episodes 2-7 (the profile's ready pose).
READY = np.array([-0.2287, -0.3292, 0.0495, -0.3238, 0.4396, 0.1176, -0.0919])


def main():
    df = pd.read_parquet(DATASET / f"data/chunk-000/episode_{EPISODE:06d}.parquet")
    state = np.stack(df["observation.state"].to_numpy())
    arm, closed = state[:, 29:36], np.abs(state[:, 36:43]).mean(1) > 0.3
    start = int(np.argmax(df["teleop.stream_mode"].to_numpy().ravel() == 1))
    edges = np.flatnonzero(np.diff(closed.astype(int)))
    grasps = [e + 1 for e in edges if closed[e + 1] and e + 1 > start]
    releases = [e + 1 for e in edges if not closed[e + 1] and e + 1 > start]
    grasp = grasps[CYCLE]
    release = next(r for r in releases if r > grasp)
    near = np.abs(arm - READY).max(1) < 0.25
    previous = releases[CYCLE - 1] if CYCLE else start
    first = [i for i in range(previous, grasp) if near[i] and not closed[i]][-1]
    last = release + int(0.6 * FPS)
    rows = slice(first, last + 1)
    tokens = np.stack(df["action.motion_token"].to_numpy())[rows]
    left = np.stack(df["teleop.left_hand_joints"].to_numpy())[rows]
    right = np.stack(df["teleop.right_hand_joints"].to_numpy())[rows]
    model = instantiate_g1_robot_model(waist_location="lower_and_upper_body")
    quat = np.stack(df["observation.root_orientation"].to_numpy())[grasp]
    tilt = rc.level_tilt(quat)
    pelvis = rc.pelvis_height_estimator(model)(np.r_[state[grasp, :22], state[grasp, 29:36]], tilt)
    q = model.get_configuration_from_actuated_joints(
        body_actuated_joint_values=np.r_[state[grasp, :22], state[grasp, 29:36]],
        right_hand_actuated_joint_values=state[grasp, 36:43])
    model.cache_forward_kinematics(q, auto_clip=False)
    palm = tilt @ model.frame_placement("right_hand_palm_link").translation
    np.savez_compressed(ROOT / "replay-cycle.npz", motion_token=tokens.astype(np.float32),
                        left_hand_joints=left.astype(np.float32), right_hand_joints=right.astype(np.float32))
    meta = dict(dataset=str(DATASET), episode=EPISODE, cycle=CYCLE + 1, fps=FPS,
                first_frame=first, last_frame=last, frames=last - first + 1,
                grasp_offset_s=(grasp - first) / FPS, release_offset_s=(release - first) / FPS,
                start_arm=arm[first].round(4).tolist(), release_wrist_roll=float(arm[release, 4]),
                pelvis_height_m=pelvis, palm_at_grasp_pelvis_frame_m=palm.round(4).tolist(),
                palm_at_grasp_floor_height_m=float(palm[2] + pelvis))
    starts = []
    for episode in START_EPISODES:
        frames = pd.read_parquet(DATASET / f"data/chunk-000/episode_{episode:06d}.parquet")
        states = np.stack(frames["observation.state"].to_numpy())
        starts.append(np.median(states[frames["teleop.stream_mode"].to_numpy().ravel() == 2], 0))
    first_state = np.median(np.stack(starts), 0)
    body29 = np.r_[first_state[:22], first_state[29:36]]  # Legs, waist, left arm, right arm.
    (ROOT / "start-pose.json").write_text(json.dumps(dict(
        source="handover_bottle_261019 initial frames (stream_mode 2), median of episodes "
               + ", ".join(map(str, START_EPISODES)),
        body_q_motor_order=body29.round(4).tolist(),
        upper_body_planner_order=body29[G1_UPPER_BODY_JOINT_INDICES].round(4).tolist(),
        left_hand=first_state[22:29].round(4).tolist(), right_hand=first_state[36:43].round(4).tolist(),
        between_episode_spread_rad=float(np.ptp(np.stack(starts), 0)[12:43].max())), indent=2) + "\n")
    (ROOT / "replay-cycle.json").write_text(json.dumps(meta, indent=2, default=lambda v: v.item()) + "\n")
    print(json.dumps(meta, indent=2, default=lambda v: v.item()))


if __name__ == "__main__":
    main()
