"""Paired-trace native worker with a pre-pose to a recorded post-handover entry.

SIMULATION FIXTURE ONLY. The first reset_ready drives the waist and both arms to
the recorded end-of-offer pose (G1_DIAG_ENTRY, e.g. source30_f420) at the same
0.5 rad/s rate and tracking trim, but without the lead guard: moving 15 joints at
once from standing made a wrist swing briefly away and trip it. The desk is kept
away meanwhile (sim_entry_worker.py). Left fingers keep their measured targets and
the right hand opens, as on the offered palm. It settles within 0.12 rad (SONIC
leaves the recorded left arm ~0.1 rad off). None of these allowances applies to the
return under test. The second reset_ready
is the production return: G1_DIAG_RETURN=profile keeps the profile waypoint;
=direct passes the measured right arm as the waypoint, so stage 0 only dwells
before the direct approach to the ready pose.
"""
import os
from pathlib import Path
import sys

import numpy as np

ART = Path(__file__).parents[1]
sys.path.insert(0, str(ART / "g1_handover_20261014_tracking"))
import native_worker as traced  # noqa: E402
from gear_sonic.scripts import run_vla_inference as loop  # noqa: E402
from gear_sonic.utils.inference import harness_control  # noqa: E402
from gear_sonic.utils.inference.standing_reset import _RIGHT_ARM_PLANNER_INDICES, StandingReset, _measured_joints  # noqa: E402
from gear_sonic.utils.teleop.xr_upperbody_bridge import G1_UPPER_BODY_JOINT_INDICES  # noqa: E402

EPISODE, FRAME = map(int, os.environ["G1_DIAG_ENTRY"].removeprefix("source").split("_f"))
RETURN = os.environ["G1_DIAG_RETURN"]
assert RETURN in ("profile", "direct")
ROW = np.load(ART / "g1_handover_20261009_followup/palm-diagnostics/measured-extract.npz")[
    f"{EPISODE}/observation.state"][FRAME]
ENTRY_BODY = np.r_[ROW[:22], ROW[29:36]]  # 29 motors: legs, waist, left arm, right arm
PRE_POSE_TOLERANCE_RAD = 0.12


class PrePose(StandingReset):
    """Fixture-only approach: rate-limited reference plus tracking trim, no lead guard."""

    def advance(self, state, dt):
        # Same input guards as StandingReset.advance: missing feedback holds the command.
        if (not isinstance(state, dict) or isinstance(dt, bool) or not isinstance(dt, (int, float))
                or not np.isfinite(dt) or dt <= 0):
            return self.command
        try:
            measured = _measured_joints(state)
        except (TypeError, ValueError):
            return self.command
        dt = min(dt, 0.05)
        reference = self._target.copy()
        self._nominal_reached |= bool(np.max(np.abs(self._position[:17] - self._target[:17])) < 1e-6)
        if self._nominal_reached:
            error = self._target[:17] - measured[:17]
            error = np.where(np.abs(error) > self._tracking_deadband, error, 0.0)
            self._tracking_trim = np.clip(self._tracking_trim + dt * error, -0.15, 0.15)
            reference[:17] += self._tracking_trim
        reference[_RIGHT_ARM_PLANNER_INDICES] = np.clip(reference[_RIGHT_ARM_PLANNER_INDICES], *self._right_arm_bounds)
        self._position = self._position + np.clip(reference - self._position, -0.5 * dt, 0.5 * dt)
        return self.command

    def is_settled(self, state, joint_tolerance, yaw_tolerance_rad):
        # SONIC leaves the recorded left arm 0.08-0.10 rad off; the return under test
        # preserves whatever the arm actually reached, with the production 0.05 rad.
        return super().is_settled(state, max(joint_tolerance, PRE_POSE_TOLERANCE_RAD), yaw_tolerance_rad)


class EntryControl(traced.PairedTraceControl):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        begin, calls = self.hooks.begin_ready_reset, []

        def begin_ready_reset(feedback, target, waypoint):
            calls.append(None)
            if len(calls) == 1:  # Pre-pose: whole recorded upper body, no waypoint.
                reset = begin(feedback, ENTRY_BODY[22:29], None)
                # The production hook may still choose a raise from this standing pose; the
                # pre-pose is not under test, so force its single stage and full target.
                reset.stage = 1
                reset._target[:17] = ENTRY_BODY[G1_UPPER_BODY_JOINT_INDICES]
                reset._target[24:] = 0.0
                reset._final_target = reset._target.copy()
                reset.__class__ = PrePose
                return reset
            if RETURN == "direct":
                waypoint = tuple(np.asarray(feedback["body_q_measured_motor"], dtype=float)[22:29])
            return begin(feedback, target, waypoint)

        self.hooks.begin_ready_reset = begin_ready_reset


if __name__ == "__main__":
    harness_control.HarnessControl = EntryControl
    loop.planner_command_in_reference_frame = traced.paired_convert
    sys.path.insert(0, str(ART / "g1_harness_20261006_followup"))
    from native_locomotion_worker import main
    main()
