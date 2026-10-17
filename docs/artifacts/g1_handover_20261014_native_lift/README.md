# Native lift-first ready return, 14 October 2026

The handover ready return is now integrated into the production native controller.
**Motion acceptance is still failing.** The final SONIC/MuJoCo run did not reach
the raised waypoint and interrupted at the unchanged 15-second deadline.
Both hardware review flags remain false.

Native implementation: `f2fafc5` on `work/volo-g1-harness-20261001`.

Watch the [saved simulation video](ready-return.mp4) or
[preview](ready-return-preview.jpg). These show the actual measured simulation
states from the failed final run, rendered after all controller processes stopped.
The bottle is a scene prop; no VLA pickup or human handover was executed.

## Implemented behavior

The native `StandingReset` first raises the right arm while retaining the entry
finger targets. It requires distinct valid upper-body and heading measurements
within tolerance for at least 0.5 seconds. It then approaches the final ready
target and opens the fingers. The harness still requires a separate final
0.5-second dwell with all 31 measured arm/finger values settled.

The existing 0.5 rad/s reference rate, 0.15 rad reference-advance lead logic,
0.05 rad settling tolerance and total 15-second deadline were retained. The
other arm, waist, heading and left finger targets are preserved. Ordinary
`reset_standing` keeps its original target behavior.

Handover profiles now require both seven-joint mappings,
`raised_right_arm_joints` and `ready_right_arm_joints`. Missing or malformed
waypoints fail profile loading. Older handover profiles must be updated; the
placement profile is unchanged. `ready_pose_reviewed` applies to the entire
path, including the lift. The disabled example uses the episode-30 frame-60
candidate from the [earlier lift investigation](../g1_handover_20261013_lift/README.md).

## Verification and observed failure

| Check | Result |
| --- | --- |
| VoLoAgent project suite | 745 passed, 19 skipped, 46 warnings |
| Native `gear_sonic/tests` suite after review fix | 664 passed, 123 warnings |
| Explicit memory-only cross-process acceptance | 13 passed |
| Final native return simulation | INTERRUPTED: `reset_timeout` |
| Time to interruption | 15.067 s |
| Saved hand/forearm desk-contact states | 0 / 311 |
| Minimum conservative sampled desk gap | 37.01 mm |
| Foot/floor contact states | 311 / 311; physical balance unqualified |
| VLA latent-action publications | 0 |

One fresh `gpt-6-astra`, high-reasoning review found that the first qualifying
lift sample credited the preceding interval. This could shorten the dwell by
one sample. The [regression](review-dwell-red.log) failed before the fix and
passed afterwards with the full native suite. No other confirmed code defect
was reported. The reviewer independently ran 68 focused tests before the fix.

The initial production run completed in 10.132 seconds before that dwell fix;
its evidence remains in [native-return](native-return/result.json). The final
run used the corrected code, retained in [native-return-final](native-return-final/result.json).
The old fixture's `completed` field means the probe reached a terminal state;
it does not imply a successful return. `measured_ready_settling_passed` and
[analysis.acceptance_passed](analysis.json) are false for the final run. The
updated launcher returns a failing exit status for an unsuccessful return.

In the final second before interruption, the saved right shoulder yaw averaged
−0.399 rad against the raised target −0.598 rad: about 0.200 rad error, four
times the settling tolerance. No saved right-arm sample in that window met
tolerance. The [tracking diagnosis](tracking-diagnosis.json) also finds about
0.300 rad between planner reference and saved physical angle. These are nearest
preceding roughly 20 Hz physics samples, not synchronized native-feedback
measurements. They do not certify the reference lead limit. The next diagnostic
must capture the controller's exact motor feedback beside its reference, check
the lead behavior when tracking falls back, and distinguish SONIC tracking
from feedback/ordering or timing errors. No gains, offsets, tolerances or
deadlines were changed to make this run pass.

All owned processes stopped. The fixture used the production Python inference
loop and C++ SONIC controller with loopback addresses, scripted prewarm/camera
and a provisional 85 cm desk whose front is 0.40 m from the scene origin. The
ready operation used no artifact motion wrapper. The [source hashes](native-return-final/result.json)
identify the exact native code tested.

The video reconstructs saved `qpos` with `mj_forward`; it is not a new physics
trial. The [render metadata](render.json) records 318 frames at 20 fps, 15.9 s,
and maximum saved-sample age 50.36 ms. Positive conservative bounds separation
and zero sampled contacts do not prove swept clearance between samples. The
person proxy has no hands/forearms, and actual post-handover entry poses remain
untested. MuJoCo hand-asset stability warnings remain in the raw logs.

The restricted VoLoAgent suite aborted when the existing mock GR00T server
created a ZeroMQ socket. The unrestricted rerun passed; both logs are retained.
The read-only native deployment preflight also remains red: the filesystem had
only about 0.3 GB free and the unused decoupled-WBC balance model is still an
LFS pointer. CUDA and TensorRT passed outside the sandbox. The existing SONIC
assets used for this simulation were available. No files were deleted, model
assets downloaded, physical robot commands sent or hardware gates enabled.

## Remaining jobs, in order

1. Diagnose the raised-pose tracking failure with paired native reference and
   measured-feedback traces. Achieve repeatable measured settling within the
   existing limits before treating the ready return as accepted.
2. Free workstation disk space before another build, dataset transfer or model
   download; select cleanup targets without deleting recordings or checkpoints.
3. Test returns from actual post-handover poses and validate desk-edge, hand,
   forearm and person clearance, including continuous paths and camera alignment.
4. Run the existing 20,000-step handover VLA through a complete connected
   pick → stable offer → observed taking → empty hand → return → repeat simulation.
   Restore H100 access if needed. Disconnected inference has passed; this cycle has not.
5. Validate longer stable offering pauses and genuinely unseen recording
   sessions. Start with 10–15-second offers; actual fresh monitor confirmations
   govern completion. Decide on further training from those results.
6. Perform supervised hardware acceptance: empty-hand return, one bottle cycle,
   then repeats. Only afterwards enable reviewed checkpoint/ready flags.

The current recorded vision baseline remains Genon `openai/gpt-5.6-sol`.
Qwen with measured hand cues still rejected all four offer frames; the finger
classifier correctly identified those hands as open. Palm support, taking,
occlusion and slipping remain vision judgments. Dashboard work stays deferred.

## Reproduce

From the VoLoAgent harness worktree, with the previously authorized GPU and
existing environments/assets:

```bash
env PYTHONPATH=. .venv/bin/python docs/artifacts/g1_handover_20261014_native_lift/run_probe.py --gpu 1 --desk-front-m 0.40 --output /tmp/volo-native-lift-new-run
```

Analysis/render helpers use the saved `native-return-final` folder:

```bash
/home/jihun/work/VoLoAgent/.worktrees/g1-harness-native/.venv/bin/python docs/artifacts/g1_handover_20261014_native_lift/analyze.py
/home/jihun/work/VoLoAgent/.worktrees/g1-harness-native/.venv/bin/python docs/artifacts/g1_handover_20261014_native_lift/diagnose_tracking.py
env MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=1 /home/jihun/work/GR00T-WholeBodyControl/.venv_sim/bin/python docs/artifacts/g1_handover_20261014_native_lift/render.py
```

The analyzer returns exit status 1 for the retained failed trial. It writes the
evidence before exiting, so a failure is reviewable rather than hidden.
