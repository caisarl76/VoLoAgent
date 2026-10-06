"""Physical positive/negative controls for the simulation truth sampler."""

import mujoco

from placement_truth import placement_confirmed, sample_bottle_truth
from test_placement_truth import DESTINATION


XML = '''<mujoco><option timestep="0.005"/><worldbody>
<body name="pelvis" pos="3 3 3"><geom name="elbow" type="sphere" size="0.05"/></body>
<geom name="source_table" type="box" pos="0.5 0.2 0.4" size="0.237 0.3 0.4"/>
<geom name="right_green_stool" type="cylinder" pos="0.35 -0.6 0.4" size="0.25 0.4"/>
<body name="task_bottle" pos="0.35 -0.6 0.877"><freejoint/>
<geom name="task_bottle_geom" type="cylinder" size="0.03 0.075" density="100"/>
</body></worldbody></mujoco>'''


def collect(model, data):
    rows = []
    for step in range(240):
        mujoco.mj_step(model, data)
        if step >= 120 and step % 20 == 0:
            rows.append(sample_bottle_truth(model, data, now=float(data.time)))
    return rows


def test_real_physics_stool_support_passes_source_support_fails():
    model = mujoco.MjModel.from_xml_string(XML)
    data = mujoco.MjData(model)
    rows = collect(model, data)
    assert placement_confirmed(rows, DESTINATION, now=rows[-1]["at"])
    data = mujoco.MjData(model)
    data.qpos[:2] = [0.45, 0]
    rows = collect(model, data)
    assert not placement_confirmed(rows, DESTINATION, now=rows[-1]["at"])


def test_robot_contact_uses_body_ancestry_even_without_hand_in_name():
    xml = XML.replace('name="pelvis" pos="3 3 3"', 'name="pelvis" pos="0.35 -0.6 0.88"')
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    row = sample_bottle_truth(model, data, now=0)
    assert any(c["geom"] == "elbow" and c["is_robot"] for c in row["contacts"])
