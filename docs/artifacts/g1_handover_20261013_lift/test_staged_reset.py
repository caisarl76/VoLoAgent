"""Simulation prototype gates; no hardware, sockets or controller process."""
import sys

import numpy as np
import pytest

sys.path.insert(0,'/home/jihun/work/VoLoAgent/.worktrees/g1-harness-native')
from gear_sonic.utils.teleop.xr_upperbody_bridge import G1_UPPER_BODY_JOINT_INDICES
from staged_reset import StagedReadyReset

FINAL=[-.1,-.08,.3,-.21,1.7,-.09,-.07]
LIFT=[.011,-.682,-.598,.164,1.086,-.157,-.490]


def state():
    body=np.zeros(29);body[22:29]=[-.04,-.259,.60,1.028,.216,-.039,-.26]
    return dict(index=0,body_q=body,left_hand_q=np.zeros(7),right_hand_q=np.ones(7),
        base_quat=[1,0,0,0])


def follow(s,command):
    s['body_q'][G1_UPPER_BODY_JOINT_INDICES]=command.upper_body_position
    s['left_hand_q']=np.array(command.left_hand_position)
    s['right_hand_q']=np.array(command.right_hand_position)
    s['index']+=1


def test_lift_stage_keeps_fingers_until_measured_raised_dwell():
    s=state();initial_hand=s['right_hand_q'].copy()
    reset=StagedReadyReset(s,s['left_hand_q'],np.zeros(7),right_arm_target=FINAL,waypoints=[LIFT])
    for _ in range(100):
        command=reset.advance(s,.02);follow(s,command)
        np.testing.assert_allclose(s['right_hand_q'],initial_hand)
        assert reset.stage==0
    for _ in range(60):follow(s,reset.advance(s,.02))
    assert reset.stage==1
    assert not reset.is_settled(s,.05,.1)


def test_repeated_feedback_cannot_finish_intermediate_dwell():
    s=state();s['body_q'][22:29]=LIFT
    reset=StagedReadyReset(s,s['left_hand_q'],np.zeros(7),right_arm_target=FINAL,waypoints=[LIFT])
    for _ in range(100):reset.advance(s,.05)
    assert reset.stage==0


def test_entire_ideal_route_keeps_native_rate_lead_and_held_targets():
    s=state();events=[]
    reset=StagedReadyReset(s,s['left_hand_q'],np.zeros(7),right_arm_target=FINAL,
        waypoints=[LIFT],on_stage=events.append)
    previous=np.r_[reset.command.upper_body_position,reset.command.left_hand_position,reset.command.right_hand_position]
    for _ in range(800):
        command=reset.advance(s,.02)
        current=np.r_[command.upper_body_position,command.left_hand_position,command.right_hand_position]
        assert np.max(np.abs(current-previous))<=.01000001
        measured=np.r_[s['body_q'][G1_UPPER_BODY_JOINT_INDICES],s['left_hand_q'],s['right_hand_q']]
        assert np.max(np.abs(current-measured))<=.15000001
        assert tuple(command.movement)==(0,0,0) and command.speed==0
        follow(s,command);previous=current
        np.testing.assert_allclose(s['body_q'][:22],0)
        if reset.is_settled(s,.05,.1):break
    else:pytest.fail('Staged route did not reach final ready target')
    assert reset.stage==1 and events==[1]
    assert s['body_q'][22:29]==pytest.approx(FINAL,abs=.05)


def test_invalid_feedback_holds_and_clears_intermediate_dwell():
    s=state();s['body_q'][22:29]=LIFT
    reset=StagedReadyReset(s,s['left_hand_q'],np.zeros(7),right_arm_target=FINAL,waypoints=[LIFT])
    for _ in range(20):s['index']+=1;reset.advance(s,.02)
    old=reset.command
    assert reset.advance(None,.05)==old
    for _ in range(10):s['index']+=1;reset.advance(s,.02)
    assert reset.stage==0


def test_raised_arm_gate_does_not_require_closed_finger_tracking_to_be_exact():
    s=state();s['body_q'][22:29]=LIFT
    reset=StagedReadyReset(s,s['left_hand_q'],np.zeros(7),right_arm_target=FINAL,waypoints=[LIFT])
    s['right_hand_q'][:]=1.08
    for _ in range(30):s['index']+=1;reset.advance(s,.02)
    assert reset.stage==1
    assert not reset.is_settled(s,.05,.1)


@pytest.mark.parametrize('waypoint',[[0]*6,[float('nan')]*7,[9]*7])
def test_bad_waypoint_rejected_before_advancing(waypoint):
    s=state()
    with pytest.raises(ValueError):
        StagedReadyReset(s,s['left_hand_q'],np.zeros(7),right_arm_target=FINAL,waypoints=[waypoint])
