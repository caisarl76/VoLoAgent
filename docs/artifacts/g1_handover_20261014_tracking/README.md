# Paired native tracking diagnosis, 14 October 2026

The reference guard is fixed. **The ready return is still not accepted.**
Corrected simulations held the high-level references within the existing
0.15 rad lead and 0.5 rad/s rate bounds, but the measured arms did not settle
before the 15-second deadline. No hardware commands ran or review flags changed.
The workstation had about 26 GB free at preflight and about 10 GiB at the final
check, so disk space did not block these trials.

See the [comparison plot](tracking-comparison.png),
[corrected-run video](ready-return.mp4) and [raw analysis](analysis.json).
The video renders saved physics; it does not replay VLA actions or demonstrate
a bottle handover. The bottle is a scene prop.

The signed-off native implementation is
[`0c44006`](https://github.com/caisarl76/GR00T-WholeBodyControl/commit/0c44006).
The required environment-check log is retained in
[`5deabf4`](https://github.com/caisarl76/GR00T-WholeBodyControl/commit/5deabf4).

For the next session investigating the remaining ready-return failure, read
the [investigation handoff](HANDOFF.md).

## Confirmed defect and fix

The unchanged production controller completed the first paired trial in 8.991 s.
Its repeat timed out. In that repeat, the exact native feedback and reference
showed up to **0.20855 rad lead**. The simulated arm moved back while the old
reference stayed fixed: the old update limited further movement toward the goal
but did not retract a reference that had become too far ahead.

`StandingReset.advance` now intersects the measured ±0.15 rad interval with the
right-arm joint limits, then approaches that interval at the existing rate. If
both limits cannot be met, it rejects the update without changing the command.
The native loop invalidates policy actions and interrupts an owned reset with
`reset_reference_lead_exceeded`, using its existing measured-hold path. Operator
resets also pause and request a measured hold. Out-of-model right-arm feedback
is rejected when constructing a reset or hold, so recovery does not publish the
invalid angle. The target behavior of ordinary standing resets is unchanged.

One fresh `gpt-6-astra`, high-reasoning review found that the initial lead
projection could undo joint-bound clipping when feedback drifted outward past
a wrist limit. The [counterexamples](review-bounds-red.log) failed before the
intersection and measured-hold validation fixes. Both signs, valid small
retractions, impossible jumps, owned interruption and operator recovery are
covered by regression tests. The [original lead regression](lead-regression-red.log)
and [runtime regression](runtime-regression-red.log) also failed before their fixes.

Final software checks: **674 native tests passed**, **745 VoLoAgent tests passed
with 19 skips**, and **13 explicit cross-process tests passed**. The software
tests use synthetic tracking and memory-only publishers; they do not qualify
the physical return.

The repository's broad environment preflight was also run outside the sandbox.
CUDA and TensorRT passed. The all-purpose check failed on training-only Python
and dependency requirements, the unused decoupled-WBC LFS balance asset, and its
10 GiB disk recommendation (9.6 GiB at that later check). These trials used the
existing native inference/simulation environments; no training setup was added.
The log is saved in the native checkout's
`docs/artifacts/g1_handover_tracking_guard_20261014/environment-preflight.log`.

## Five recorded simulation trials

| Trial | Implementation / waypoint | Terminal | Maximum exact reference lead | Sampled desk gap |
| --- | --- | --- | ---: | ---: |
| `paired-baseline` | Original / source 30 frame 60 | COMPLETED, 8.991 s | 0.15000 rad, 17 upper joints checked | 61.93 mm |
| `paired-repeat` | Original / source 30 frame 60 | `reset_timeout`, 15.052 s | **0.20855 rad**, 17 upper joints checked | 68.86 mm |
| `corrected-first` | Lead fix before joint-bound review / source 30 frame 60 | `reset_timeout`, 15.051 s | 0.15000 rad, 17 upper joints checked | 20.75 mm |
| `candidate-15-first` | Final guard / source 15 frame 70 | `reset_timeout`, 15.051 s | 0.15000 rad, all 31 joints checked | 55.04 mm |
| `decoder-trace` | Final guard / source 30 frame 60 | `reset_timeout`, 15.051 s | 0.15000 rad, all 31 joints checked | 40.03 mm |

The final two traces also contain the exact Python loop interval used by the
ramp. Neither exceeded the per-step rate bound. All five trials had zero sampled
hand/forearm desk contacts, retained foot/floor contact, used zero-speed planner
packets, published no VLA latent actions and stopped all owned processes.
These are sampled component checks in the provisional 85 cm desk scene, not
continuous swept-clearance or physical balance certificates.

The episode-15 candidate was screened from the prior recorded approach. Its
right shoulder yaw was close to target, but the preserved **left** shoulder yaw
was up to 0.382 rad away. It was not adopted into the hardware template. This
shows why checking only the right arm would hide a failure to preserve posture.
The intermediate dwell and final settling still check the specified joints.

## Where the remaining error appears

In the final decoder trace, the last 40 native samples gave this right-shoulder
yaw comparison:

| Value | Mean angle |
| --- | ---: |
| Raised physical target | −0.59844 rad |
| Bounded Python reference | −0.37694 rad |
| SONIC decoder motor target | −0.40761 rad |
| Native measured angle | −0.22694 rad |

The decoder motor target was about 0.03067 rad from the bounded reference, while
the motor-target/measurement gap was about **0.18067 rad**. Native feedback and
nearest saved physics also agreed closely; their maximum difference across all
upper joints in this final window was 0.00812 rad. The exact native pairs, not
that approximate physics association, establish the corrected lead bound.

At the recorded poses, zero-velocity forward dynamics estimated mean gravity
load −2.226 Nm on that joint. The native proportional gain and decoded/measured
angle difference give an estimated proportional torque of −2.575 Nm. No
robot contacts with non-floor geometries were found in the matched final-window
states. Static desk geometry is distinguished from the floor by geometry ID,
not by its world-body ID. The 200 non-floor contact points counted across those
40 saved states were between scene props. This supports
investigating arm dynamics under gravity. It does **not** establish a wrong
gain or mass, and excludes damping, friction, acceleration and measured motor
torque. The arm was still moving. No gains, masses, feedforward torques, settling
tolerances or deadlines were changed to force a successful result.

`last_action` is already converted to absolute motor-order angles by the native
ZMQ output handler; the analysis does not scale it twice. `body_q_target` is
also captured, but that is the planner motion's target. The encoder can overlay
external upper-body targets while gathering observations, so this field alone
does not prove the effective encoder target. Capture that overlay/flag if
further diagnosis needs to separate input composition from tracking dynamics.
The scene's actuator-to-joint mapping was checked independently: all 43
actuators matched the simulator's `joint_id - 1` indexing convention.

Each trial retains its actual observer source, native source hashes, simulation
profile and raw streams. Tracing changes no target or motion algorithm. The
initial streams check 17 upper joints; the final observer also records both hand
references and verifies all 31 joints. Profile review flags are true only in the
isolated test fixtures with scripted prewarm/camera and loopback controller
addresses. The actual handover example and placement profile are unchanged.

## Next work

1. Verify effective encoder upper-body targets and the model's mass/gain/actuator
   conventions against the SONIC assets. Qualify a return that settles while
   preserving the other arm, waist and heading, within the existing limits.
2. Validate actual post-handover entry poses, desk/person clearance and continuous
   paths. Camera/scene alignment and human hands are not qualified by this scene.
3. Run the existing handover VLA through a complete connected cycle and repeats.
4. Validate longer stable offers and recordings from unseen sessions before
   deciding whether further training is needed. Genon remains the vision baseline;
   Qwen still misses the recorded valid offers.
5. Perform supervised hardware acceptance before enabling either review flag.

Dashboard work remains in another session. Disk-space cleanup is complete;
the unrelated decoupled-WBC LFS balance model remains a separate asset issue.

## Reproduce

From this VoLoAgent worktree, using the previously authorized GPU:

```bash
env PYTHONPATH=. .venv/bin/python docs/artifacts/g1_handover_20261014_tracking/run_probe.py --gpu 1 --desk-front-m 0.40 --output /tmp/volo-tracking-new-run
```

The optional `--candidate-profile docs/artifacts/g1_handover_20261014_tracking/candidate-source15-f70.yaml`
selects the isolated episode-15 candidate, not the hardware configuration.
The launcher's failed exit status on an unsuccessful measured return is intentional.

Analyze or render the retained evidence without running new physics:

```bash
/home/jihun/work/VoLoAgent/.worktrees/g1-harness-native/.venv/bin/python docs/artifacts/g1_handover_20261014_tracking/analyze.py
env MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=1 /home/jihun/work/GR00T-WholeBodyControl/.venv_sim/bin/python docs/artifacts/g1_handover_20261014_tracking/render.py
```
