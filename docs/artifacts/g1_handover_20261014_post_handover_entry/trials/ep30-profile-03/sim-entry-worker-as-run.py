"""Acceptance sim worker that keeps the desk away while the fixture pre-poses the arm.

SIMULATION FIXTURE ONLY. After grounded startup the desk geoms and the bottle
move +3 m in x, so reaching the recorded offer pose cannot hit the desk. When
the probe creates `desk-restore`, both move back, but only if the conservative
right-hand/forearm to desk bounds gap is positive; the outcome goes to
desk-restore.json. The return under test always runs with the desk in place.
"""
import json
from pathlib import Path
import sys

import mujoco
import numpy as np

ROOT = Path(__file__).parent
sys.path[:0] = [str(ROOT.parent / "g1_handover_20261012_acceptance"), str(ROOT.parent / "g1_handover_20261013_lift")]
from gear_sonic.utils.mujoco_sim import base_sim  # noqa: E402
import sim_worker  # noqa: E402
from geometry import closest_bounds_gap  # noqa: E402

SHIFT_M = 3.0
DESK = ["desk_top", *(f"desk_leg_{i}" for i in range(4))]
ORIGINAL_STARTUP, ORIGINAL_STEP = sim_worker.apply_grounded_startup, base_sim.DefaultEnv.sim_step
STATE = dict(env=None, away=False, done=False, steps=0)


def shift(env, dx):
    model, data = env.mj_model, env.mj_data
    for name in DESK:
        model.geom_pos[model.geom(name).id][0] += dx
    joint = model.joint("task_bottle_free")
    data.qpos[joint.qposadr[0]] += dx
    data.qvel[joint.dofadr[0]:joint.dofadr[0] + 6] = 0
    mujoco.mj_forward(model, data)


def startup(sim):
    receipt = ORIGINAL_STARTUP(sim)
    STATE["env"] = sim.sim_env
    shift(sim.sim_env, SHIFT_M)
    STATE["away"] = True
    return dict(receipt, desk_moved_away_m=SHIFT_M)


def sim_step(self):
    ORIGINAL_STEP(self)
    STATE["steps"] += 1
    if not STATE["away"] or STATE["done"] or STATE["steps"] % 25:
        return
    output = Path(sys.argv[sys.argv.index("--output") + 1])
    if not (output / "desk-restore").exists():
        return
    STATE["done"] = True
    shift(self, -SHIFT_M)
    gap = closest_bounds_gap(self.mj_model, self.mj_data, ["desk_top"])
    restored = gap["gap_m"] > 0
    if not restored:  # Never drop the desk into the arm.
        shift(self, SHIFT_M)
    STATE["away"] = not restored
    (output / "desk-restore.json").write_text(json.dumps(dict(
        restored=restored, sim_time=float(self.mj_data.time), desk_gap=gap,
        desk_top_xpos=self.mj_data.geom_xpos[self.mj_model.geom("desk_top").id].tolist()), indent=2) + "\n")


if __name__ == "__main__":
    sim_worker.apply_grounded_startup = startup
    base_sim.DefaultEnv.sim_step = sim_step
    sim_worker.main()
