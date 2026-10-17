# Handoff: investigate the G1 handover ready-return failure

Prepared 14 October 2026. Start with the saved evidence, then instrument one
isolated SONIC/MuJoCo trial to locate the remaining tracking error.

## Task and current failure

Determine why `reset_ready` cannot reliably settle at its raised waypoint, and
produce a supported diagnosis or a verified fix. Keep the existing motion
limits and measured completion checks.

The intended application is a stationary G1 across an 85 cm desk from a person:
detect a bottle, pick it, offer it on a stable upward-facing open right palm,
observe removal, return to a handshake-ready pose, then repeat. The return first
raises the hand to avoid the desk, retains entry finger positions during that
stage, and then opens the hand while moving to the final ready pose.

The latest trial starts from grounded standing and tests that return component.
It uses the real native SONIC controller with scripted camera/policy prewarm.
It publishes no VLA latent actions. The bottle is a scene prop. The full
connected VLA cycle and actual post-handover entry poses remain untested here.

The final `decoder-trace` trial stays in stage 0 (raising). It never satisfies
the intermediate dwell, then interrupts after about 15 seconds with
`reset_timeout`. Thus opening/final-ready motion is never reached.

## Starting checkouts

| Checkout | Branch | Published starting commit |
| --- | --- | --- |
| `/home/jihun/work/VoLoAgent/.worktrees/g1-harness` | `work/g1-harness-20261001` | `7bff810` |
| `/home/jihun/work/VoLoAgent/.worktrees/g1-harness-native` | `work/volo-g1-harness-20261001` | `5deabf4` |

The native tracking fix is `0c44006`; `5deabf4` adds its environment-check log.
`7bff810` contains the five recorded trials and their analysis. Check current
Git status before edits. The original
`/home/jihun/work/GR00T-WholeBodyControl` checkout contains unrelated user work;
use its existing assets/environments as the fixture does and preserve that work.

Use the worktrees' separate Python environments: VoLoAgent `.venv` is Python
3.13; native `.venv` is Python 3.10 with MuJoCo and matplotlib; the original
GR00T checkout's `.venv_sim` has MuJoCo/cv2 for rendering. VoLoAgent's environment
does not contain MuJoCo.

## First: inspect the saved failure

Read the [tracking report](README.md), view the
[comparison plot](tracking-comparison.png) and [video](ready-return.mp4), then
inspect these files under `decoder-trace/`:

- [result.json](decoder-trace/result.json): terminal phase/reason, source and
  binary hashes, component scope and process cleanup.
- [paired-reference-feedback.jsonl](decoder-trace/paired-reference-feedback.jsonl):
  exact native reference/feedback pairs, loop intervals, stage, targets, trim,
  `controller_last_action` and `controller_motion_target`.
- [ground-truth.jsonl](decoder-trace/ground-truth.jsonl): sampled physics state,
  qpos/qvel, support and clearance evidence.
- [simulation-profile.yaml](decoder-trace/simulation-profile.yaml) and
  [scene.xml](decoder-trace/scene.xml): the actual trial configuration.
- [trace-worker-as-run.py](decoder-trace/trace-worker-as-run.py): the observer
  source used for that trial, matching the recorded hash.

The last 40 native samples give this right-shoulder-yaw comparison. Motor index
is 24; its upper-body planner index is 8. These are averages, not one instant.

| Value | Radians | Degrees |
| --- | ---: | ---: |
| Raised waypoint | -0.59844 | -34.29 |
| Bounded Python reference | -0.37694 | -21.60 |
| Decoded motor target | -0.40761 | -23.35 |
| Measured joint | -0.22694 | -13.00 |

The decoded/measured gap is 0.18067 rad (10.35 degrees). Native feedback agrees
with nearby saved physics: the maximum upper-body difference in that final
window is 0.00812 rad. All final traces remain at stage 0; the final original-pose
trace has zero recorded lift dwell. Its tracking trim reaches the existing
0.15 rad cap, so bias compensation is already active.

Completion of this step: verify the failure and angle gaps from the saved data,
and distinguish the desired waypoint, bounded reference and decoded target.

## What is already fixed or checked

The old reset could freeze its reference when an arm moved back, producing
0.20855 rad lead against a 0.15 rad limit. The fix intersects measured lead and
right-arm joint-limit intervals, retracts at the existing rate, and interrupts
an impossible update. Invalid right-arm measurements cannot construct a hold.
Both-sign joint-limit counterexamples from a fresh review are covered by tests.

The five trials are summarized in [README.md](README.md). The old baseline
completed once; its repeat failed. The corrected trials all timed out. The final
original waypoint and isolated episode-15 candidate both respected the reference
rate/lead limits on all 31 commanded joints. The candidate's right arm tracked
better, but its preserved left shoulder yaw missed by up to 0.382 rad. It was
not adopted into the hardware template.

All 43 scene actuators matched the simulator's `joint_id - 1` indexing convention.
Sampled hand/forearm desk contact counts were zero. In the final 40 matched
physics states, robot contacts with non-floor geometries were zero. The 200
non-floor contact points were between scene props. Distinguish floor by geometry
ID: desk geometry also belongs to world body 0.

The current gravity calculation gives about -2.226 Nm at those poses. With the
native shoulder Kp of 14.2506 Nm/rad, the decoded/measured gap gives an estimated
proportional torque of -2.575 Nm. This supports investigating arm dynamics under
gravity. It does not establish incorrect gains or masses: velocity, damping,
friction, acceleration and the delivered actuator torque still need checking.

## Next investigation, in order

1. Capture the effective encoder targets in one loopback trial. Record
   `has_upper_body_data_`, `upper_body_joint_positions_buffer_`, observation
   mode, joint ordering and the final position/velocity features consumed by
   the encoder, with a common tick identifier. `body_q_target` describes the
   planner motion; the encoder can overlay the 17 external upper-body targets.
   Completion: show whether the effective features match the bounded Python
   request, including preserved joints and the applicable observation window.
2. Capture the delivered motor commands and applied simulation torques, especially
   motors 24 and 17. Record q/dq targets, measured q/dq, kp/kd, feedforward,
   torque clipping and `mj_data.ctrl` alongside those encoder inputs. Compare
   model masses/inertias and gain/effort conventions against the SONIC assets.
   Completion: identify where the remaining error enters and account for the
   actual PD/torque behavior; the existing gravity estimate alone is insufficient.
3. Test the identified mechanism with a controlled comparison, changing one
   diagnostic variable at a time and retaining the unchanged baseline. Any
   experimental gains or gravity changes belong in isolated simulation fixtures
   until justified against the intended deployment model. Completion: save
   evidence that supports or rejects the proposed cause without changing the
   acceptance thresholds.
4. If a production fix is justified, add a meaningful regression, run affected
   checks and repeat independently started simulations. Rebuild and record the
   binary hash if C++ changes. Completion: document each trial's measured dwell,
   stage transition, final settling, posture preservation and terminal result.
   Report pass counts; one successful trial does not establish repeatability.

Relevant native sources, relative to the native checkout:

| Area | File / entry point |
| --- | --- |
| Bounded reference and intermediate dwell | `gear_sonic/utils/inference/standing_reset.py`, `StandingReset.advance` |
| Reset timeout, final dwell and measured hold | `gear_sonic/utils/inference/harness_control.py` |
| Native loop and interruption | `gear_sonic/scripts/run_vla_inference.py` |
| Effective encoder features / motor command | `gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/src/g1_deploy_onnx_ref.cpp`, `GatherMotionJointPositionsMultiFrame`, `CreatePolicyCommand` |
| Gains, scales, default angles and ordering | `gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/include/policy_parameters.hpp` |
| Telemetry scaling | `gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/include/output_interface/zmq_output_handler.hpp` |
| Applied PD and clipping | `gear_sonic/utils/mujoco_sim/base_sim.py`, `compute_body_torques`, `sim_step` |

`last_action` in ZMQ telemetry is already converted to absolute motor-order
angles with scale and default offset. Use it directly; double scaling corrupts
the comparison. Also capture the actual delivered command rather than assuming
this telemetry establishes every subsequent output transformation.

## Commands and fixture limits

From the VoLoAgent worktree, recompute the saved analysis using native Python.
This updates derived analysis/plot files and sends no robot commands:

```bash
cd /home/jihun/work/VoLoAgent/.worktrees/g1-harness
/home/jihun/work/VoLoAgent/.worktrees/g1-harness-native/.venv/bin/python docs/artifacts/g1_handover_20261014_tracking/analyze.py
```

For a fresh simulation, check available disk/GPU space and existing owned
processes. The prior trials used local GPU 1, with a 2048 MiB free-memory
preflight; GPU allocation is shared and must be checked again. Use a new output
directory. The probe checks loopback ports and refuses an existing controller
or simulator. An unsuccessful measured return deliberately exits with code 1.

```bash
cd /home/jihun/work/VoLoAgent/.worktrees/g1-harness
env PYTHONPATH=. .venv/bin/python docs/artifacts/g1_handover_20261014_tracking/run_probe.py --gpu 1 --desk-front-m 0.40 --output /tmp/volo-tracking-next-session-01
```

The probe uses `ready_probe.py` in `g1_handover_20261012_acceptance`, the
production-provenance wrapper in `g1_handover_20261014_native_lift`, and this
folder's `native_worker.py`. Its simulator also reads existing helpers under
`/home/jihun/work/.enpire-runtime/g1-bottle-sim-20261006`. Record any dependency
changes. `analyze.py` currently selects the five saved trial folders and
`render.py` selects `decoder-trace`; adapt these explicitly for new evidence.
Preserve the original raw streams and their as-run sources.

Keep experiments on loopback with zero-speed planner packets, grounded physics,
virtual support disabled and only the native controller publishing actuation.
Keep `checkpoint_verified` and `ready_pose_reviewed` false in actual profiles.
The fixture sets both true only in its isolated generated simulation profile.

Retain the existing 0.5 rad/s reference rate, 0.15 rad reference lead,
right-arm joint bounds, 0.05 rad joint tolerance, 5-degree heading tolerance,
0.5-second dwells and 15-second reset deadline. Raising checks the preserved
upper body and heading; final settling also checks the fingers. Sampled
clearance in the provisional scene does not qualify continuous paths, camera
alignment or human-hand interaction.

## Verification and later work

The published source passed 674 native tests, 745 VoLoAgent tests with 19 skips,
and 13 explicit memory-only cross-process tests. Logs are in this folder.
After native Python changes, use these checks from the indicated checkouts:

```bash
cd /home/jihun/work/VoLoAgent/.worktrees/g1-harness-native
env PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest gear_sonic/tests -q --tb=short
cd /home/jihun/work/VoLoAgent/.worktrees/g1-harness
env PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest -q --tb=short
env PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest tests/test_g1_harness_e2e.py --g1-native-checkout=/home/jihun/work/VoLoAgent/.worktrees/g1-harness-native --g1-native-python=/home/jihun/work/VoLoAgent/.worktrees/g1-harness-native/.venv/bin/python --require-g1-e2e -q --tb=short
```

The broad environment preflight passed CUDA/TensorRT but failed on training
setup, an unused decoupled-WBC LFS balance asset and its 10 GiB disk
recommendation (9.6 GiB remaining at the recorded check). See the native
`docs/artifacts/g1_handover_tracking_guard_20261014/environment-preflight.log`.
These results describe the existing inference/simulation environments.

After reliable ready settling, return to the
[handover implementation plan](../../superpowers/plans/2026-10-09-g1-bottle-handover-autopilot.md)
for actual post-handover entry poses, swept desk/person clearance and the full
VLA cycle. Dataset: `/home/jihun/work/GR00T-WholeBodyControl/outputs/handover_bottle_260930`.
The user confirmed episodes 15 and 30 as intended handovers and episode 16 as a
bottle rolling off the open palm due to hand angle. Exact existing VLA prompt:
`pick drink bottle from the table and handover`.

The rule-based measured-hand classifier is implemented; recorded Qwen offer
recognition remains a separate bottleneck, with Genon as the vision baseline.
Dashboard implementation belongs to another session. Finish this investigation
with a source-linked diagnosis, saved comparison evidence, verification results
and an explicit statement of which motion/full-cycle acceptance remains open.
Sign off every published commit with the author's valid name and email.
