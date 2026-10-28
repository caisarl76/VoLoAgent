"""Ready-pose candidates with the right gripper out of the head-camera view (2026-10-26).

Each candidate is the ep76 ready pose with right-arm joint offsets (chosen in MuJoCo so the gripper leaves the
d435_head image with a 0.1 rad margin); its token is the SONIC encoding of the new static joint pose (the same
encoder and adapter lerobot's make_ready_pose uses for the lowering). Hands, raise and lower paths are unchanged:
the loop's 1 s token blends join raise end -> ready and ready -> lower start, as they do today (0.50 rad).
"""
import hashlib, json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
from sonic_targets import SonicEncoder, joint_chunk_to_sonic, load_joint_limits
S = Path.home() / "work/GR00T-WholeBodyControl/gear_sonic_deploy"
enc = SonicEncoder(S / "policy/sonic_v1_1/model_encoder.onnx", S / "policy/sonic_v1_1/observation_config.yaml")
lim = load_joint_limits(S / "g1/g1_29dof_with_hand.xml")
SRC = Path("/home/jihun/work/VoLoAgent/.worktrees/g1-loop-native/gear_sonic/data/handover/ready_pose_261020_ep76.npz")
base = dict(np.load(SRC))
RIGHT = {"shoulder_pitch": 7, "shoulder_roll": 8, "shoulder_yaw": 9, "elbow": 10, "wrist_roll": 11, "wrist_pitch": 12, "wrist_yaw": 13}
CANDS = {"A": {"shoulder_yaw": -0.40}, "B": {"shoulder_roll": -0.15, "shoulder_yaw": -0.40},
         "C": {"shoulder_pitch": -0.20, "wrist_yaw": -0.50}}
out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
base_static = joint_chunk_to_sonic(np.tile(base["state"], (60, 1)), lim, enc, fps=50)[0, :64]
for name, d in CANDS.items():
    state = base["state"].copy()
    for j, v in d.items(): state[RIGHT[j]] += v
    clipped = np.clip(state[:14], lim[:14, 0], lim[:14, 1])
    assert np.array_equal(clipped, state[:14]), f"{name} outside joint limits"
    token = joint_chunk_to_sonic(np.tile(state, (60, 1)), lim, enc, fps=50)[0, :64].astype(np.float32)
    assert np.abs(token).max() <= 1.25
    meta = json.loads(str(base["meta"])) | {"candidate": name, "right_arm_offsets_rad": d,
            "token": "static SONIC encoding of the offset state (sonic_targets.joint_chunk_to_sonic, 60 x 50 Hz hold)",
            "token_dist_from_ep76_static_encoding": float(np.abs(token - base_static).max())}
    pose = base | {"token": token, "state": state.astype(np.float32), "meta": json.dumps(meta)}
    p = out / f"ready_pose_261026_cand{name}.npz"
    np.savez(p, **pose)
    print(name, d, "token max|diff| vs ep76", round(float(np.abs(token - base["token"]).max()), 3), p.name, hashlib.sha256(p.read_bytes()).hexdigest()[:12])
