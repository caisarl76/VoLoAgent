"""Acceptance sim worker plus a delivered-PD/applied-torque recorder.

G1_DIAG_GRAVITY_FF_MOTORS (comma-separated motor indices) adds the model's
static gravity torque as feedforward on those motors. SIMULATION DIAGNOSTIC
ONLY: it changes the physics the policy sees and is not a deployment model.
"""
import json
import os
from pathlib import Path
import sys

import mujoco
import numpy as np

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT.parent / "g1_handover_20261012_acceptance"))
from gear_sonic.utils.mujoco_sim import base_sim  # noqa: E402
import sim_worker  # noqa: E402

FF_MOTORS = [int(x) for x in os.environ.get("G1_DIAG_GRAVITY_FF_MOTORS", "").split(",") if x]
EVERY = 5  # 500 Hz physics -> 100 Hz trace.
ORIGINAL_TORQUES = base_sim.DefaultEnv.compute_body_torques
ORIGINAL_STEP = base_sim.DefaultEnv.sim_step
STATE = dict(step=0, scratch=None, stream=None, pending=None)


def _static_gravity(env):
    """Joint torque that holds the current pose against gravity (zero velocity)."""
    if STATE["scratch"] is None:
        STATE["scratch"] = mujoco.MjData(env.mj_model)
    scratch = STATE["scratch"]
    scratch.qpos[:] = env.mj_data.qpos
    scratch.qvel[:] = 0
    mujoco.mj_kinematics(env.mj_model, scratch)
    mujoco.mj_comPos(env.mj_model, scratch)
    bias = np.zeros(env.mj_model.nv)
    mujoco.mj_rne(env.mj_model, scratch, 0, bias)
    return bias[env.body_joint_index + env.qvel_offset - 1]


def compute_body_torques(self):
    torques = ORIGINAL_TORQUES(self)
    gravity = _static_gravity(self)
    feedforward = np.zeros_like(torques)
    feedforward[FF_MOTORS] = gravity[FF_MOTORS]
    if STATE["step"] % EVERY == 0 and self.unitree_bridge is not None and self.unitree_bridge.low_cmd:
        cmd = self.unitree_bridge.low_cmd.motor_cmd
        n = self.unitree_bridge.num_body_motor
        STATE["pending"] = dict(
            sim_time=float(self.mj_data.time), env_class=type(self).__name__,
            cmd_q=[cmd[i].q for i in range(n)], cmd_dq=[cmd[i].dq for i in range(n)],
            kp=[cmd[i].kp for i in range(n)], kd=[cmd[i].kd for i in range(n)],
            tau_ff=[cmd[i].tau for i in range(n)],
            q=self.mj_data.qpos[self.body_joint_index + self.qpos_offset - 1].tolist(),
            dq=self.mj_data.qvel[self.body_joint_index + self.qvel_offset - 1].tolist(),
            pd_torque=torques.tolist(), static_gravity=gravity.tolist(),
            diagnostic_feedforward=feedforward.tolist(),
        )
    return torques + feedforward


def sim_step(self):
    ORIGINAL_STEP(self)
    pending, STATE["pending"] = STATE["pending"], None
    if pending is not None:
        actuators = self.body_joint_index - 1
        dofs = self.body_joint_index + self.qvel_offset - 1
        pending.update(
            clipped_torque=self.torques[actuators].tolist(),
            actuator_force=self.mj_data.actuator_force[actuators].tolist(),
            qfrc_constraint=self.mj_data.qfrc_constraint[dofs].tolist(),
            qfrc_passive=self.mj_data.qfrc_passive[dofs].tolist(),
        )
        if STATE["stream"] is None:
            output = Path(sys.argv[sys.argv.index("--output") + 1])
            STATE["stream"] = (output / "pd-trace.jsonl").open("x")
        STATE["stream"].write(json.dumps(pending, allow_nan=False) + "\n")
    STATE["step"] += 1


if __name__ == "__main__":
    base_sim.DefaultEnv.compute_body_torques = compute_body_torques
    base_sim.DefaultEnv.sim_step = sim_step
    try:
        sim_worker.main()
    finally:
        if STATE["stream"] is not None:
            STATE["stream"].close()
