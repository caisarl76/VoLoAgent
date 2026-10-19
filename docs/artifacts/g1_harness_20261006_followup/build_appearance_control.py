"""Render a bottle appearance comparison with identical collision physics."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import cv2
import mujoco
import numpy as np

from gear_sonic.data.robot_model.instantiation.g1 import instantiate_g1_robot_model
from gear_sonic.scripts.run_vla_inference import prepare_observation_from_sensors


def rows(path):
    return [json.loads(s) for s in path.read_text().splitlines()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scene', type=Path, required=True)
    parser.add_argument('--saved-live', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    events = rows(next((args.saved_live/'mission').glob('*/events.jsonl')))
    started = next(x['at'] for x in events if x['event'] == 'skill_start')
    truth_path = args.saved_live/'ground-truth.jsonl'
    source = min(rows(truth_path), key=lambda x: abs(x['at']-started))
    obj = min(rows(args.saved_live/'bottle-ground-truth.jsonl'), key=lambda x: abs(x['at']-source['at']))
    models, data, paths, images = [], [], [], {}
    for name in ['cyan', 'clear_green_cap']:
        folder = args.output/name
        folder.mkdir()
        shutil.copy2(args.scene.parent/'robot.xml', folder/'robot.xml')
        xml = ET.parse(args.scene)
        if name == 'clear_green_cap':
            body = xml.getroot().find("worldbody/body[@name='task_bottle']")
            body.find("geom[@name='task_bottle_geom']").set('rgba', '0.6 0.75 0.8 0.22')
            # Visual cap lies within the original cylinder bounds and adds no mass/contact.
            ET.SubElement(body, 'geom', name='task_bottle_cap_visual', type='cylinder',
                          size='0.028 0.005', pos='0 0 0.095', rgba='0.35 0.7 0.08 1',
                          contype='0', conaffinity='0', density='0')
        path = folder/'scene.xml'
        xml.write(path)
        model = mujoco.MjModel.from_xml_path(str(path.resolve()))
        state = mujoco.MjData(model)
        state.qpos[:7] = source['floating_base_pose']
        values = np.r_[source['body_q'], source['left_hand_q'], source['right_hand_q']]
        state.qpos[model.jnt_qposadr[model.actuator_trnid[:, 0]]] = values
        joint = model.joint('task_bottle_free').qposadr[0]
        state.qpos[joint:joint+7] = [*obj['bottle_position'], 1, 0, 0, 0]
        mujoco.mj_forward(model, state)
        with mujoco.Renderer(model, height=480, width=640) as renderer:
            renderer.update_scene(state, camera='head_camera')
            rgb = renderer.render().copy()
        cv2.imwrite(str(folder/'ego_view.png'), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        images[name] = rgb
        models.append(model)
        data.append(state)
        paths.append(path)
    invariant_fields = ['body_mass', 'body_inertia', 'body_ipos', 'body_iquat', 'jnt_qposadr',
                        'jnt_dofadr', 'dof_damping', 'actuator_trnid', 'actuator_gainprm',
                        'cam_pos', 'cam_quat', 'cam_fovy']
    for field in invariant_fields:
        np.testing.assert_array_equal(getattr(models[0], field), getattr(models[1], field))
    for geom in range(models[0].ngeom):
        name = models[0].geom(geom).name
        # All original geom IDs and collision parameters must remain identical.
        assert models[1].geom(geom).name == name
        for field in ['geom_type', 'geom_size', 'geom_pos', 'geom_quat', 'geom_contype',
                      'geom_conaffinity', 'geom_friction', 'geom_solref', 'geom_solimp']:
            np.testing.assert_array_equal(getattr(models[0], field)[geom], getattr(models[1], field)[geom])
    cap = models[1].geom('task_bottle_cap_visual').id
    assert models[1].geom_contype[cap] == models[1].geom_conaffinity[cap] == 0
    robot = instantiate_g1_robot_model(waist_location='lower_and_upper_body')
    state_msg = {key: np.asarray(source[key]) for key in ['body_q', 'left_hand_q', 'right_hand_q']}
    state_msg['base_quat'] = np.asarray(source['floating_base_pose'][3:7])
    arrays = {}
    for name, rgb in images.items():
        camera = SimpleNamespace(read=lambda rgb=rgb: dict(images={'ego_view': rgb}, timestamps={'ego_view': 1.0}))
        subscriber = SimpleNamespace(get_msg=lambda: state_msg)
        observation = prepare_observation_from_sensors(camera, subscriber, robot,
                         'pick drink bottle and place it on the right table')
        arrays.update({f'{name}__state__{key}': value for key, value in observation['state'].items()})
        arrays[f'{name}__image'] = observation['video']['ego_view']
    np.savez_compressed(args.output/'inputs.npz', **arrays)
    maximum_delta = 0.0
    frozen = data[0].qpos[:50].copy()
    for _ in range(round(1/models[0].opt.timestep)):
        for model, state in zip(models, data):
            state.qpos[:50] = frozen
            state.qvel[:49] = 0
            mujoco.mj_step(model, state)
        maximum_delta = max(maximum_delta, float(np.max(np.abs(data[0].qpos-data[1].qpos))))
    assert maximum_delta < 1e-12
    result = dict(robot_actuated=False,dds_started=False,physics_unchanged=True,
                  invariant_fields=invariant_fields,one_second_dynamics_max_qpos_delta=maximum_delta,
                  image_mean_absolute_difference=float(np.mean(np.abs(images['cyan'].astype(float)-images['clear_green_cap']))),
                  state_sample_at=source['at'],skill_start_at=started,state_skew_s=source['at']-started,
                  reconstructed_pose_is_approximate=True,whole_scene_calibrated=False,
                  sources={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [args.scene,truth_path]},
                  scenes={p.parent.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
    (args.output/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
