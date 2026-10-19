"""Independent separation counterexamples for the provisional route screen."""
import mujoco
import numpy as np

from geometry import geom_aabb, aabb_gap


def test_separated_boxes_have_known_conservative_gap():
    m=mujoco.MjModel.from_xml_string('<mujoco><worldbody><geom name="a" type="box" size=".1 .1 .1"/><geom name="b" type="box" size=".1 .1 .1" pos=".5 0 0"/></worldbody></mujoco>')
    d=mujoco.MjData(m);mujoco.mj_forward(m,d)
    assert np.isclose(aabb_gap(geom_aabb(m,d,0),geom_aabb(m,d,1)),.3)


def test_overlapping_bounds_cannot_certify_separation():
    assert aabb_gap((np.array([0,0,0]),np.array([1,1,1])),
        (np.array([.5,.5,.5]),np.array([1.5,1.5,1.5])))==0


def test_rotated_translated_box_bounds_include_every_corner():
    m=mujoco.MjModel.from_xml_string('<mujoco><worldbody><geom type="box" size=".1 .2 .3" pos=".5 .4 .3" euler="0 0 45"/></worldbody></mujoco>')
    d=mujoco.MjData(m);mujoco.mj_forward(m,d)
    low,high=geom_aabb(m,d,0)
    half=np.array([.3/np.sqrt(2),.3/np.sqrt(2),.3])
    np.testing.assert_allclose(low,np.array([.5,.4,.3])-half,atol=1e-7)
    np.testing.assert_allclose(high,np.array([.5,.4,.3])+half,atol=1e-7)
