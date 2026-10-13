"""Headless MuJoCo scene; controller communication is restricted to loopback."""
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

from scene import closest_clearance

LEGACY = Path('/home/jihun/work/.enpire-runtime/g1-bottle-sim-20261006')
sys.path.insert(0,str(LEGACY))
from grounded_startup import apply_grounded_startup
from scene_fixture import configure_hand_order


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--scene',type=Path,required=True)
    args = parser.parse_args()
    config = SimLoopConfig(interface='sim',enable_onscreen=False,enable_offscreen=False,
        enable_real_device=False,data_collection=False)
    wbc = config.load_wbc_yaml()
    assert wbc['INTERFACE']=='lo' and wbc['DOMAIN_ID']==0
    wbc['ROBOT_SCENE'], wbc['ENV_NAME'] = str(args.scene),config.env_name
    sim = SimWrapper(instantiate_g1_robot_model(),config.env_name,wbc,onscreen=False,offscreen=False).sim
    env = sim.sim_env
    mapping = configure_hand_order(env)
    state = dict(automatic_resets=0,startup_resets=0,physics_steps=0)
    applying_startup, sample_at = False, 0.0
    original_step, original_reset = env.sim_step,env.reset
    deadline = time.monotonic()+100

    def stop(*_):
        sim._running = False

    signal.signal(signal.SIGTERM,stop)
    signal.signal(signal.SIGINT,stop)

    def reset():
        state['startup_resets' if applying_startup else 'automatic_resets'] += 1
        return original_reset()

    env.reset = reset
    with (args.output/'ground-truth.jsonl').open('x') as stream:
        def step():
            nonlocal applying_startup,sample_at
            if (args.output/'startup-request').exists() and state['startup_resets']==0:
                applying_startup = True
                try:
                    receipt = apply_grounded_startup(sim)
                finally:
                    applying_startup = False
                (args.output/'startup-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
            original_step()
            state['physics_steps'] += 1
            now = time.monotonic()
            if now >= sample_at:
                sample_at = now+0.05
                foot_contacts=[]
                for contact in env.mj_data.contact:
                    ids=[int(env.mj_model.geom_bodyid[g]) for g in (contact.geom1,contact.geom2)]
                    names=[env.mj_model.body(i).name or '' for i in ids]
                    if 0 in ids and any('ankle' in n or 'foot' in n for n in names):
                        foot_contacts.append(names)
                clearance=closest_clearance(env.mj_model,env.mj_data,
                    ['desk_top',*(f'desk_leg_{i}' for i in range(4)),'person_proxy_geom'])
                stream.write(json.dumps(dict(received_monotonic=now,sim_time=float(env.mj_data.time),
                    qpos=env.mj_data.qpos.tolist(),qvel=env.mj_data.qvel.tolist(),
                    body_q=env.obs['body_q'].tolist(),left_hand_q=env.obs['left_hand_q'].tolist(),
                    right_hand_q=env.obs['right_hand_q'].tolist(),
                    band_enabled=bool(env.elastic_band.enable),
                    support_force_norm=float(np.linalg.norm(env.mj_data.xfrc_applied[env.band_attached_link])),
                    foot_floor_contacts=foot_contacts,clearance=clearance,**state),allow_nan=False)+'\n')
                stream.flush()
            if now>=deadline or (args.output/'stop-simulator').exists():
                stop()
        env.sim_step=step
        (args.output/'simulator-ready.json').write_text(json.dumps(dict(interface='lo',domain_id=0,
            hand_mapping=mapping,headless=True,mujoco_version=mujoco.__version__),indent=2)+'\n')
        try:
            sim.start()
        finally:
            sim.close()
            (args.output/'simulator-result.json').write_text(json.dumps(state,indent=2)+'\n')


if __name__=='__main__':
    main()
