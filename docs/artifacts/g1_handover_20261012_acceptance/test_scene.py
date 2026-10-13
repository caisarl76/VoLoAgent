"""Independent model and signed-distance checks for the provisional scene."""
from pathlib import Path
import sys

import mujoco
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))
from scene import SOURCE_SCENE, build_scene, closest_clearance


def test_scene_preserves_every_robot_joint_and_actuator(tmp_path):
    before = SOURCE_SCENE.read_bytes()
    original = mujoco.MjModel.from_xml_path(str(SOURCE_SCENE))
    path = tmp_path/'scene.xml'
    build_scene(path, desk_front_m=0.40)
    model = mujoco.MjModel.from_xml_path(str(path))
    assert SOURCE_SCENE.read_bytes() == before
    assert model.nu == original.nu == 43
    np.testing.assert_array_equal(model.actuator_trnid, original.actuator_trnid)
    for i in range(original.njnt):
        assert model.joint(i).name == original.joint(i).name
        assert model.jnt_qposadr[i] == original.jnt_qposadr[i]
        np.testing.assert_array_equal(model.jnt_range[i], original.jnt_range[i])


def test_confirmed_height_and_explicit_bottle_assumptions(tmp_path):
    path = tmp_path/'scene.xml'
    build_scene(path, desk_front_m=0.35)
    model = mujoco.MjModel.from_xml_path(str(path))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    desk = model.geom('desk_top').id
    assert data.geom_xpos[desk, 2]+model.geom_size[desk, 2] == pytest.approx(0.85)
    assert data.geom_xpos[desk, 0]-model.geom_size[desk, 0] == pytest.approx(0.35)
    np.testing.assert_allclose(model.geom_size[model.geom('task_bottle_geom').id, :2], [0.04, 0.10])
    assert model.body_mass[model.body('task_bottle').id] == pytest.approx(0.30)
    assert model.jnt_type[model.joint('task_bottle_free').id] == mujoco.mjtJoint.mjJNT_FREE
    person = data.xpos[model.body('person_proxy').id]
    assert person[0] > data.geom_xpos[desk, 0]+model.geom_size[desk, 0]
    assert person[1] == 0


def test_signed_clearance_detects_an_intersecting_hand_proxy():
    model = mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
      <geom name="desk_top" type="box" pos="0 0 0.825" size="0.3 0.3 0.025"/>
      <body name="right_hand_test" pos="0 0 1.00"><freejoint/>
        <geom name="hand" type="sphere" size="0.05" mass="0.1"/>
      </body></worldbody></mujoco>''')
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    assert closest_clearance(model, data, ['desk_top'])['distance_m'] == pytest.approx(0.10)
    data.qpos[2] = 0.88
    mujoco.mj_forward(model, data)
    result = closest_clearance(model, data, ['desk_top'])
    assert result['distance_m'] == pytest.approx(-0.02)
    assert result['obstacle'] == 'desk_top'


@pytest.mark.parametrize('front', [float('nan'), float('inf'), -0.1])
def test_bad_distance_rejected_before_scene_creation(tmp_path, front):
    path = tmp_path/'scene.xml'
    with pytest.raises(ValueError):
        build_scene(path, desk_front_m=front)
    assert not path.exists()
