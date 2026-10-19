"""Check bottle measurements and truth-gate controls in the native scene."""

import argparse
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np

from placement_truth import placement_confirmed, sample_bottle_truth


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scene',type=Path,required=True)
    parser.add_argument('--original-scene',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    for name in ['verify_bottle_calibration.py', 'placement_truth.py']:
        (args.output / f'as-run-{name}').write_bytes(Path(__file__).with_name(name).read_bytes())
    model = mujoco.MjModel.from_xml_path(str(args.scene.resolve()))
    original = mujoco.MjModel.from_xml_path(str(args.original_scene.resolve()))
    geom, body = model.geom('task_bottle_geom').id, model.body('task_bottle').id
    radius, half_height = model.geom_size[geom,:2]
    mass = model.body_mass[body]
    np.testing.assert_allclose([2*radius,2*half_height,mass],[0.08,0.20,0.30],rtol=1e-9)
    old_geom,old_body=original.geom('task_bottle_geom').id,original.body('task_bottle').id
    stool = model.geom('right_green_stool').id
    destination = dict(geom='right_green_stool',xy=model.geom_pos[stool,:2].tolist(),
                       radius=float(model.geom_size[stool,0]),height=float(model.geom_pos[stool,2]+model.geom_size[stool,1]))
    qa,va = model.joint('task_bottle_free').qposadr[0],model.joint('task_bottle_free').dofadr[0]
    results = {}
    for name,xy in [('source',[0.45,0]),('forced_destination',destination['xy'])]:
        data=mujoco.MjData(model)
        robot_q=data.qpos[:qa].copy()
        robot_q[2]=0.786
        data.qpos[qa:qa+7]=[*xy,destination['height']+half_height+0.015,1,0,0,0]
        rows=[]
        step_count = round(2.0 / model.opt.timestep)
        sample_stride = max(1, round(0.1 / model.opt.timestep))
        sample_start = round(1.2 / model.opt.timestep)
        for step in range(step_count):
            data.qpos[:qa]=robot_q
            data.qvel[:va]=0
            mujoco.mj_step(model,data)
            if step>=sample_start and step%sample_stride==0:
                rows.append(sample_bottle_truth(model,data,now=float(data.time)))
        confirmed=placement_confirmed(rows,destination,now=rows[-1]['at'])
        assert confirmed==(name=='forced_destination'),(name,rows)
        (args.output/f'{name}.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows))
        results[name]=dict(placement_confirmed=confirmed,last_sample=rows[-1])
    result=dict(robot_actuated=False,policy_used=False,positive_control_forced_object_placement=True,
                freshness_clock='simulation_time_for_offline_controls',scene_fully_calibrated=False,
                model_timestep_s=float(model.opt.timestep),
                matched_user_bottle=dict(height_m=2*half_height,diameter_m=2*radius,mass_kg=mass),
                previous_sample_bottle=dict(height_m=float(2*original.geom_size[old_geom,1]),
                    diameter_m=float(2*original.geom_size[old_geom,0]),mass_kg=float(original.body_mass[old_body])),
                results=results,sources={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [args.scene,args.original_scene]})
    (args.output/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['results','sources']}))


if __name__=='__main__':
    main()
