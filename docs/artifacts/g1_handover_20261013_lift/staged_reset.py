"""Simulation-only lift-before-reach prototype built on native reset limits."""
import numbers

import numpy as np

from gear_sonic.utils.inference.standing_reset import StandingReset, _measured_joints
from gear_sonic.utils.teleop.xr_upperbody_bridge import feedback_payload_heading_yaw


class StagedReadyReset:
    """Raise with current fingers, settle, then open and approach final ready.

    This fixture does not know real desk/person geometry and cannot be installed
    as a hardware collision guard. The native harness retains its total deadline
    and final settling checks; intermediate dwell requires distinct feedback.
    """

    def __init__(self,state,left_hand_target,right_hand_target,*,right_arm_target,
                 waypoints,on_stage=None,**kwargs):
        final=StandingReset(state,left_hand_target,right_hand_target,
            right_arm_target=right_arm_target,**kwargs)
        self._final_target=final.target
        if not waypoints:raise ValueError('A raised waypoint is required')
        held_hand=_measured_joints(state)[24:]
        self._targets=[StandingReset(state,left_hand_target,held_hand,
            right_arm_target=t,**kwargs).target for t in waypoints]+[final.target]
        self._reset=StandingReset(state,left_hand_target,held_hand,
            right_arm_target=waypoints[0],**kwargs)
        self.stage=0
        self._dwell=0.
        self._last_index=None
        self._tolerance=kwargs.get('joint_tolerance_rad',.05)
        self._on_stage=on_stage

    @property
    def target(self):return self._final_target.copy()

    @property
    def command(self):return self._reset.command

    def is_settled(self,state,joint_tolerance,yaw_tolerance_rad):
        return self.stage==len(self._targets)-1 and self._reset.is_settled(state,joint_tolerance,yaw_tolerance_rad)

    def _raised_arm_settled(self,state):
        try:
            measured=_measured_joints(state)
            yaw=feedback_payload_heading_yaw(state)
            if yaw is None:return False
            error=np.arctan2(np.sin(yaw-self._reset._yaw),np.cos(yaw-self._reset._yaw))
            # Fingers stay at their entry targets during lifting. Their gravity
            # tracking must not block arm progress; full settling is required at final ready.
            return bool(np.max(np.abs(measured[:17]-self._reset.target[:17]))<=self._tolerance
                and abs(error)<=np.radians(5))
        except (TypeError,ValueError):return False

    def advance(self,state,dt):
        if not isinstance(state,dict) or not np.isfinite(dt) or dt<=0:
            self._dwell=0.;return self.command
        index=state.get('index')
        if isinstance(index,bool) or not isinstance(index,numbers.Integral) or index<0:
            self._dwell=0.;return self.command
        if self._last_index is not None and index<self._last_index:
            self._dwell=0.;return self.command
        command=self._reset.advance(state,dt)
        if self.stage==len(self._targets)-1:return command
        if index!=self._last_index:
            self._last_index=index
            if self._raised_arm_settled(state):
                self._dwell+=float(np.clip(dt,0,.05))
            else:self._dwell=0.
            if self._dwell>=.5:
                self.stage+=1
                self._reset._target=self._targets[self.stage].copy()
                self._reset._tracking_trim[:]=0
                self._reset._nominal_reached=False
                self._dwell=0.
                if self._on_stage:self._on_stage(self.stage)
        return command
