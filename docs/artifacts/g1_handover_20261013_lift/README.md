# Raising the hand before reaching over the desk

Episodes 15 and 30 show the useful approach: raise the hand while it stays back,
then bring it forward and lower it toward the bottle. Watch the
[recorded approaches side by side](recorded-lift.mp4), slowed to half speed.
The original recordings are unchanged.

The recorded pose around source 30 frame 60 provides a raised intermediate pose.
In the provisional model, the hand clears an 85 cm tabletop before its forward
extent crosses the assumed 40 cm near edge. Source 15 shows the same ordering.
The [frame sheets](source-30-raising.jpg) and [geometry plot](lift-comparison.png)
show how height changes before reach. Absolute heights depend on the assumed
78.7 cm pelvis height and neutral root orientation; the ego video supports the
motion ordering, not those exact scene measurements. Source 16 is an approach
comparison only; its later user-confirmed roll/rescue remains a failed episode.

The [new SONIC video](ready-return.mp4) tests this empty-hand sequence:

1. Keep the current finger pose while raising the arm to the demonstrated
   intermediate pose, nearer the robot and above the desk.
2. Wait for measured arm/heading settling on distinct feedback frames for
   0.5 seconds.
3. Open the empty hand while approaching the original palm-up ready target.
   Require the existing full final settling check before completing.

This is a simulation-only wrapper around the existing native reset. It keeps
the 0.5 rad/s rate cap, 0.15 rad measured-lead cap, 0.05 rad tolerance and total
15-second deadline. It preserves the entry targets for the other arm, waist,
left hand and heading. The underlying controller can compensate tracking bias
within its existing limits. Intermediate settling checks the upper body and
heading while fingers retain their entry target; final settling still checks
the hands too. It is not a runtime desk/person collision guard.

| Measured simulation result | Previous direct return | Lift-first prototype |
|---|---:|---:|
| Time to COMPLETED | 5.339 s | 8.942 s |
| Largest final pre-completion 0.5 s right-arm error | 0.04860 rad | 0.03182 rad |
| Minimum conservative sampled hand/forearm-to-desk gap | 0 mm; bounds overlap | 53.21 mm |
| Saved snapshots with simulated desk contact | 7 / 121 | 0 / 189 |
| Foot/floor contact snapshots | 121 / 121 | 189 / 189 |

These are checks of roughly 20 Hz saved physics states, including a short
post-completion hold. They do not certify uninterrupted swept clearance between
samples. All owned processes stopped; the G1 remained in zero-speed planner
mode, with no VLA latent-action publication, virtual support or automatic reset.
The [measured comparison](native-lift-comparison.png) and
[raw statistics](native-comparison.json) distinguish this trial from ideal
tracking and recorded demonstrations.

The first prototype tried opening the fingers while the arm hung low. Finger
tracking was slow in that posture, and the native deadline interrupted it before
lifting. Its [failure evidence](staged-native/result.json) is preserved. The
corrected trial lifts first; it does not increase the timeout or relax final
settling. Regression tests cover that sequence, repeated/invalid feedback,
intermediate finger tracking, final settling, limits and preserved targets.
The fixture/scene suite passed 17 tests; the project suite passed 709 with
19 skips and 46 warnings.

Signed-distance queries sometimes gave zero for separated shapes in this
inspection. A positive separation between conservative geometry bounds provides
an independent separation check; overlapping bounds provide no clearance
certificate. The comparison also recomputes MuJoCo contacts from the saved
states. The implementation of the raw query is documented in
[MuJoCo 3.8.1 source](https://github.com/google-deepmind/mujoco/blob/3.8.1/src/engine/engine_support.c#L488).
The preview/video displays the conservative desk gap.

No production native reset or hardware profile was changed. The helper defaults
for the earlier direct probe are preserved; this trial explicitly substitutes
the fixture worker. The hand assets warn of simulation instability, the person
proxy has no hands/forearms, and desk position and bottle geometry remain
provisional. Physical balance, stable bottle support and the full VLA handover
cycle remain unqualified.

Use this demonstrated lift-before-reach ordering for the next native ready-path
implementation, with the intermediate pose configured and reviewed for the real
desk. Once that path is accepted, run the VLA from the raised ready state using
the existing handover prompt. Test that new starting state against recordings
and unseen cycles before enabling hardware. This native preparation motion does
not require inventing another VLA prompt.

Reproduction from the VoLoAgent worktree:

```bash
/home/jihun/work/VoLoAgent/.worktrees/g1-harness-native/.venv/bin/python docs/artifacts/g1_handover_20261013_lift/inspect_lift.py
env PYTHONPATH=. .venv/bin/python docs/artifacts/g1_handover_20261013_lift/run_staged_probe.py --output /tmp/volo-lift-new-run --desk-front-m 0.40 --gpu 1
/home/jihun/work/VoLoAgent/.worktrees/g1-harness-native/.venv/bin/python docs/artifacts/g1_handover_20261013_lift/analyze_native.py
env MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=1 /home/jihun/work/GR00T-WholeBodyControl/.venv_sim/bin/python docs/artifacts/g1_handover_20261013_lift/render_staged.py
```

The analysis/render commands inspect the saved fixed trial. A new control trial
requires a fresh output directory, available GPU memory and the recorded native
assets/environments on this workstation. [WORK.md](WORK.md) records scope and
decisions; [verification.json](verification.json) records hashes and checks.
