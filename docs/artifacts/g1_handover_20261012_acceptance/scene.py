"""Provisional 85 cm desk scene and geometric screening; no actuator publisher."""
import copy
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

NATIVE = Path('/home/jihun/work/VoLoAgent/.worktrees/g1-harness-native')
SOURCE_SCENE = NATIVE/'gear_sonic/data/robot_model/model_data/g1/scene_43dof.xml'


def vector(values):
    return ' '.join(str(float(v)) for v in values)


def build_scene(output, *, desk_front_m=0.40, desk_height_m=0.85):
    output = Path(output)
    if not np.isfinite(desk_front_m) or desk_front_m <= 0:
        raise ValueError('Desk front distance must be positive and finite')
    if not np.isfinite(desk_height_m) or not 0.5 <= desk_height_m <= 1.2:
        raise ValueError('Desk height must be finite, 0.5-1.2 m')
    root = ET.parse(SOURCE_SCENE).getroot()
    include = root.find('include')
    source_robot = SOURCE_SCENE.parent/include.attrib['file']
    robot = ET.parse(source_robot).getroot()
    root.remove(include)
    for element in reversed(list(robot)):
        node = copy.deepcopy(element)
        if node.tag == 'compiler':
            node.set('meshdir', str((source_robot.parent/node.get('meshdir','')).resolve()))
        root.insert(0, node)
    root.set('model', 'g1_provisional_handover_85cm')
    world = root.find('worldbody')
    width, depth, height, thickness = 1.20, 0.50, desk_height_m, 0.05
    center_x = desk_front_m+depth/2
    ET.SubElement(world, 'geom', name='desk_top', type='box',
        pos=vector([center_x,0,height-thickness/2]),
        size=vector([depth/2,width/2,thickness/2]), rgba='0.35 0.23 0.12 1')
    for i,(x,y) in enumerate((
        (desk_front_m+0.025,-width/2+0.025), (desk_front_m+0.025,width/2-0.025),
        (desk_front_m+depth-0.025,-width/2+0.025), (desk_front_m+depth-0.025,width/2-0.025))):
        ET.SubElement(world, 'geom', name=f'desk_leg_{i}', type='box',
            pos=vector([x,y,(height-thickness)/2]), size=vector([0.025,0.025,(height-thickness)/2]),
            rgba='0.23 0.23 0.23 1')
    person = ET.SubElement(world, 'body', name='person_proxy',
        pos=vector([desk_front_m+depth+0.30,0,0.875]))
    ET.SubElement(person, 'geom', name='person_proxy_geom', type='capsule',
        size='0.20 0.675', rgba='0.20 0.55 0.85 0.5')
    bottle_center = [desk_front_m+0.15,-0.10,height+0.10+0.001]
    bottle = ET.SubElement(world, 'body', name='task_bottle', pos=vector(bottle_center))
    ET.SubElement(bottle, 'freejoint', name='task_bottle_free')
    ET.SubElement(bottle, 'geom', name='task_bottle_geom', type='cylinder',
        size='0.04 0.10', mass='0.30', rgba='0.65 0.85 0.75 1',
        friction='0.7 0.005 0.001', condim='4')
    ET.SubElement(world, 'camera', name='handover_overview', pos='1.6 -2.0 1.5',
        mode='targetbody', target='pelvis', fovy='50')
    ET.SubElement(world, 'camera', name='handover_closeup', pos='0.85 -0.85 1.30',
        mode='targetbody', target='right_wrist_yaw_link', fovy='45')
    ET.indent(root)
    with output.open('x') as stream:
        stream.write(ET.tostring(root,encoding='unicode')+'\n')
    layout = dict(source_scene=str(SOURCE_SCENE), source_robot=str(source_robot),
        source_scene_sha256=hashlib.sha256(SOURCE_SCENE.read_bytes()).hexdigest(),
        source_robot_sha256=hashlib.sha256(source_robot.read_bytes()).hexdigest(),
        desk_height_m=height, desk_front_from_robot_origin_m=desk_front_m,
        desk_depth_m=depth, desk_width_m=width, bottle_center_xyz_m=bottle_center,
        bottle_height_m=0.20, bottle_diameter_m=0.08, bottle_mass_kg=0.30,
        person_proxy_center_xyz_m=[desk_front_m+depth+0.30,0,0.875],
        physical_scene_calibrated=False, palm_contact_calibrated=False,
        bottle_dimensions_confirmed_for_handover=False,
        assumption='Desk height and across-desk arrangement confirmed; other geometry provisional.',
        hand_model_warning='Source scene explicitly warns the hand model is unstable for simulation.',
        human_hand_and_forearm_clearance_modelled=False)
    output.with_suffix('.layout.json').write_text(json.dumps(layout,indent=2)+'\n')
    return layout


def closest_clearance(model, data, obstacle_names):
    """Signed geometric distance for right hand/wrist/elbow collision geoms.

    Meshes use MuJoCo's collision representation. This does not establish actual
    human clearance or stable contact. Positive distances are capped at 1 m.
    """
    nearest = dict(distance_m=1.0, actor=None, obstacle=None, distance_capped=True)
    actors = []
    for geom in range(model.ngeom):
        body = model.body(int(model.geom_bodyid[geom])).name or ''
        if body.startswith(('right_hand','right_wrist','right_elbow')) and model.geom_contype[geom]:
            actors.append((geom,body))
    if not actors:
        raise ValueError('No right hand/forearm collision geometry found')
    for name in obstacle_names:
        obstacle = model.geom(name).id
        for geom, body in actors:
            distance = float(mujoco.mj_geomDistance(model,data,geom,obstacle,1.0,np.zeros(6)))
            if distance < nearest['distance_m']:
                nearest = dict(distance_m=distance,actor=body,obstacle=name,distance_capped=False)
    return nearest
