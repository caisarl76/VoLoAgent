"""Screen raised waypoints by the maximum shoulder-yaw gravity sag along the reset path.

SONIC barely offsets right-shoulder-yaw gravity sag, so the arm stalls wherever
static gravity / Kp on that joint exceeds the usable reference lead (0.15 rad
minus ~0.014 frictionloss and ~0.02 policy give-back, about 0.115 rad). The
stall happens mid-swing, so the screen bounds the maximum sag over the ideal
production-reset path (StandingReset, perfect tracking) until yaw is within
0.05 rad of the waypoint, not the endpoint value.

Candidates: raised frames from user-confirmed handover episodes 15 and 30, and a
grid around the current waypoint whose per-joint move from entry is no larger
than the current one (+0.05 rad). Raised = whole hand >=3 cm above the assumed
85 cm desk while behind its 40 cm near edge, and not already the ready pose.
Second constraint: the preserved left shoulder yaw. Its gravity load is tiny, but
SONIC holds it wherever the right-arm swing disturbs it rather than restoring the
reference; drift by 3 s grew with right-hand forward reach in every tested pose
(0.306 m: 0.035 rad, 0.327 m: 0.053, 0.353 m: 0.092, 0.359 m: 0.112). Reach is
limited to 0.32 m, below the 0.05 rad tolerance crossing.

Kept: path-max yaw sag <= 0.10 rad, hand forward reach <= 0.32 m, positive
ideal-route conservative desk gap, not already rejected in SONIC. Chosen: largest
desk gap. Also reports the sag/reach Pareto frontier. No robot commands.
"""
import json
from pathlib import Path
import sys

import mujoco
import numpy as np
import yaml

ROOT = Path(__file__).parent
ART = ROOT.parent
NATIVE = Path("/home/jihun/work/VoLoAgent/.worktrees/g1-harness-native")
sys.path[:0] = [str(NATIVE), str(ART / "g1_handover_20261012_acceptance"), str(ART / "g1_handover_20261013_lift")]
from gear_sonic.utils.inference.standing_reset import StandingReset  # noqa: E402
from gear_sonic.utils.teleop.xr_upperbody_bridge import G1_UPPER_BODY_JOINT_INDICES  # noqa: E402
from geometry import closest_bounds_gap, geom_aabb  # noqa: E402

FIXTURE = ART / "g1_handover_20261012_acceptance/ready-040-fixture"
ARM = ["right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint", "right_elbow_joint",
       "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint"]
# Native kps (policy_parameters.hpp): STIFFNESS_5020 for 5 joints, STIFFNESS_4010 for wrist pitch/yaw.
KP = np.array([.003609725] * 5 + [.00425] * 2) * (10 * 2 * np.pi) ** 2
CURRENT = "source30_f60"
PATH_SAG_LIMIT = 0.10
# ponytail: reach is a 4-pose empirical proxy for policy-driven left-yaw drift, not a model;
# replace with a per-candidate SONIC drift measurement if a pose lands near the limit.
REACH_LIMIT_M = 0.32
# Endpoint-sag screen choice; 0/5 in SONIC (lead runaway, preserved left shoulder yaw).
REJECTED_IN_SIM = {"source30_f103"}
GRID = np.stack(np.meshgrid(np.arange(-.3, .31, .1), np.arange(0, .51, .05), np.arange(-.2, .61, .1),
                            np.arange(-.3, .31, .1), indexing="ij"), -1).reshape(-1, 4)  # pitch, roll, yaw, elbow


def main():
    z = np.load(ART / "g1_handover_20261009_followup/palm-diagnostics/measured-extract.npz")
    model = mujoco.MjModel.from_xml_path(str(FIXTURE / "scene.xml"))
    data = mujoco.MjData(model)
    names = z["names"].tolist()
    assert names[29:36] == ARM
    addresses = np.array([model.joint(n).qposadr[0] for n in names])
    body_addresses = np.concatenate([addresses[:22], addresses[29:36]])
    hand_order = ("thumb_0", "thumb_1", "thumb_2", "index_0", "index_1", "middle_0", "middle_1")
    hand_addresses = {s: [model.joint(f"{s}_hand_{n}_joint").qposadr[0] for n in hand_order] for s in ("left", "right")}
    arm_q = [model.joint(n).qposadr[0] for n in ARM]
    arm_v = [model.joint(n).dofadr[0] for n in ARM]
    yaw_v = arm_v[2]
    hand_geoms = [g for g in range(model.ngeom) if model.geom_contype[g] and (
        (model.body(int(model.geom_bodyid[g])).name or "").startswith("right_hand")
        or model.body(int(model.geom_bodyid[g])).name == "right_wrist_yaw_link")]

    events = [json.loads(s) for s in (FIXTURE / "events.jsonl").read_text().splitlines()]
    begin = next(r["at"] for r in events if r["event"] == "ready_begin")
    initial = next(r for r in map(json.loads, (FIXTURE / "ground-truth.jsonl").read_text().splitlines())
                   if r["received_monotonic"] >= begin)
    entry_qpos = np.array(initial["qpos"])
    entry = dict(body_q=np.array(initial["body_q"]), left_hand_q=np.array(initial["left_hand_q"]),
                 right_hand_q=np.array(initial["right_hand_q"]), base_quat=entry_qpos[3:7].copy())
    final = list(yaml.safe_load((ART.parents[1] / "configs/g1/handover.example.yaml").read_text())
                 ["handover"]["ready_right_arm_joints"].values())

    def static(arm):
        """Sag and hand geometry with the entry body and the given right arm."""
        data.qpos[:] = entry_qpos
        data.qpos[arm_q] = arm
        data.qvel[:] = 0
        mujoco.mj_forward(model, data)
        boxes = [geom_aabb(model, data, g) for g in hand_geoms]
        return (np.abs(data.qfrc_bias[arm_v]) / KP, min(b[0][2] for b in boxes), max(b[1][0] for b in boxes))

    def route(waypoint):
        """Ideal tracking of the production reset: desk gap and per-stage max yaw sag."""
        state = {k: v.copy() for k, v in entry.items()}
        gap, path_sag = np.inf, [0.0, 0.0]
        for stage, target in enumerate((waypoint, final)):
            hand = entry["right_hand_q"] if stage == 0 else np.zeros(7)
            reset = StandingReset(state, state["left_hand_q"], hand, right_arm_target=target)
            for _ in range(1000):
                command = reset.advance(state, .02)
                state["body_q"][G1_UPPER_BODY_JOINT_INDICES] = command.upper_body_position
                state["left_hand_q"] = np.array(command.left_hand_position)
                state["right_hand_q"] = np.array(command.right_hand_position)
                data.qpos[:] = entry_qpos
                data.qpos[body_addresses] = state["body_q"]
                for side in ("left", "right"):
                    data.qpos[hand_addresses[side]] = state[f"{side}_hand_q"]
                mujoco.mj_forward(model, data)
                gap = min(gap, closest_bounds_gap(model, data, ["desk_top"])["gap_m"])
                if abs(state["body_q"][24] - target[2]) > .05:
                    path_sag[stage] = max(path_sag[stage], abs(data.qfrc_bias[yaw_v]) / KP[2])
                if reset.is_settled(state, 1e-5, .01):
                    break
            else:
                raise RuntimeError("Ideal reference did not settle")
        return dict(ideal_route_min_desk_gap_m=gap, raise_path_max_yaw_sag_rad=path_sag[0],
                    final_path_max_yaw_sag_rad=path_sag[1])

    def candidate(label, arm, **extra):
        sag, lowest_z, farthest_x = static(arm)
        return dict(label=label, right_arm_rad=np.asarray(arm).tolist(), sag_rad=sag.tolist(),
                    hand_lowest_z_m=float(lowest_z), hand_farthest_x_m=float(farthest_x), **extra)

    candidates = [candidate(f"source{ep}_f{frame}", row[29:36], source_episode=ep, source_frame=frame)
                  for ep in (15, 30) for frame, row in enumerate(z[f"{ep}/observation.state"])]
    current = next(c for c in candidates if c["label"] == CURRENT)
    entry_arm = entry["body_q"][22:29]
    move_cap = np.abs(np.subtract(current["right_arm_rad"], entry_arm)) + .05
    for delta in GRID:
        arm = np.array(current["right_arm_rad"])
        arm[:4] += delta
        if np.all(np.abs(arm - entry_arm) <= move_cap):
            label = "grid_f60_" + "_".join(f"{k}{v:+.2f}" for k, v in zip(("p", "r", "y", "e"), delta))
            candidates.append(candidate(label, arm, synthesized_from=CURRENT, delta_pitch_roll_yaw_elbow=delta.tolist()))
    # Endpoint sag bounds the path max from below near the end of the swing; skip
    # routing poses that cannot pass.
    raised = [c for c in candidates if c["hand_lowest_z_m"] >= .88 and c["hand_farthest_x_m"] < .40
              and np.max(np.abs(np.subtract(c["right_arm_rad"], final))) > .1
              and c["label"] not in REJECTED_IN_SIM
              and (c["sag_rad"][2] <= PATH_SAG_LIMIT + .01 or c["hand_farthest_x_m"] <= REACH_LIMIT_M)]
    screened = [dict(c, **route(c["right_arm_rad"])) for c in [current] + raised]
    clear = [c for c in screened[1:] if c["ideal_route_min_desk_gap_m"] > 0]
    kept = [c for c in clear if c["raise_path_max_yaw_sag_rad"] <= PATH_SAG_LIMIT
            and c["hand_farthest_x_m"] <= REACH_LIMIT_M]
    frontier = [c for c in clear if not any(
        o["raise_path_max_yaw_sag_rad"] <= c["raise_path_max_yaw_sag_rad"] and o["hand_farthest_x_m"] <= c["hand_farthest_x_m"]
        and (o["raise_path_max_yaw_sag_rad"], o["hand_farthest_x_m"]) != (c["raise_path_max_yaw_sag_rad"], c["hand_farthest_x_m"])
        for o in clear)]
    frontier.sort(key=lambda c: c["hand_farthest_x_m"])
    chosen = max(kept, key=lambda c: c["ideal_route_min_desk_gap_m"]) if kept else None
    report = dict(
        scope="Static gravity/Kp sag along an ideal-tracking production-reset path and provisional-scene "
              "geometry, entry body from ready-040-fixture (grounded standing, not a post-handover pose). "
              "Not dynamics, swept clearance or hardware qualification.",
        desk_height_assumed_m=.85, desk_front_assumed_m=.40, kp=KP.tolist(), joint_names=ARM,
        path_sag_limit_rad=PATH_SAG_LIMIT, reach_limit_m=REACH_LIMIT_M, rejected_in_sim=sorted(REJECTED_IN_SIM),
        candidates_total=len(candidates), routed=len(raised), kept=len(kept),
        current=screened[0], screened=screened, frontier=frontier, chosen=chosen, hardware_acceptance=False)
    (ROOT / "screen.json").write_text(json.dumps(report, indent=2) + "\n")
    print("sag/reach frontier (current first):")
    for c in [screened[0]] + frontier + sorted(kept, key=lambda c: -c["ideal_route_min_desk_gap_m"])[:10]:
        print(f"{c['label']:>34} path_yaw_sag={c['raise_path_max_yaw_sag_rad']:.3f} end={c['sag_rad'][2]:.3f} "
              f"final_path={c['final_path_max_yaw_sag_rad']:.3f} z={c['hand_lowest_z_m']:.3f} "
              f"x={c['hand_farthest_x_m']:.3f} gap={c['ideal_route_min_desk_gap_m'] * 1000:.1f}mm")
    print("candidates", len(candidates), "routed", len(raised), "kept", len(kept),
          "chosen", chosen and chosen["label"])

if __name__ == "__main__":
    main()
