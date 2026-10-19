# G1 harness follow-up validation — 2026-10-02

The real checkpoint, native VLA loop, SONIC, live simulator camera and Genon
monitor now run together. The first live trial exposed a false success: the
bottle stayed on the source table. Naming the separate green stool in the
completion criterion fixed that observed case. A fresh trial produced 44
in-progress decisions, then interrupted at the 120-second deadline with planner
hold confirmed. The bottle never left the source table, so pick-and-place remains
unvalidated. No physical robot was used.

Two actual VLA-loop fault checks pass, and an editable sample scene passes
mapping/contact/camera checks. Negative wrapped turning remains unaccepted.

The earlier [validation report](../g1_harness_20261001/README.md) retains software,
checkpoint-serving, recorded-vision and controller results. This report adds
follow-up evidence on the same native runtime. The dashboard remains deferred.

## Live checkpoint and destination correction

Both live trials use the recorded `checkpoint-20000`, exact trained prompt
`pick drink bottle and place it on the right table`, unchanged native main loop,
real C++ controller and sample scene on DDS `lo`. Native inference remains the
sole actuator publisher. The policy runs on GPU 1 and SONIC/camera on GPU 0;
the workstation has two RTX 3060 GPUs. The coordinator uses
`openai/gpt-5.6-sol` through `https://api.genon.ai/v1`.

| Check | Result | Limit |
| --- | --- | --- |
| [Original live criterion](live-original-criterion/result.json) | Coordinator falsely completed; independent placement failed | Source table appeared on the image's right |
| [Three saved negative frames](live-negative-replay/result.json) | Three valid in-progress decisions, zero false-completes | First saved frame substitutes for the unsaved initial reference |
| [Recorded bottle regression](recorded-green-stool-vision/result.json) | Ten valid judgments; eight negatives, zero false-completes; one two-frame release confirmation | Small, session-labeled acceptance set |
| [Fresh live trial](live-green-stool-criterion/result.json) | 44 in-progress decisions; task interruption after 120.159 s; advancing planner feedback confirms hold | No successful grasp or placement |

[The three original frames](live-false-complete-frames.png) show the bottle
remaining on the source table. Two raw model replies called it complete;
the second satisfied the monitor's two-frame requirement. Independent simulator
positions and contacts show that the bottle stayed on `source_table`. The
coordinator paused and completed a measured standing reset after the false
judgment; that successful transition does not establish manipulation success.
The historical raw result still records coordinator completion and a failed
placement check. It has not been rewritten to conceal the error.

The profile now identifies the destination as the **separate green stool**,
requires movement from the source and release onto that stool, and rejects
source-table, held, occluded or unseen-destination evidence. Camera left/right
does not identify the destination. This changes the monitor criterion only;
the checkpoint's trained prompt is unchanged. Current profile SHA-256:
`2cb9b06e6283a552eedf1d6f2c3659229908d9309b6e3b5e0270be732c896154`.
Earlier fault and turning trials used the historical profile digest in
[provenance](provenance.json).

The fresh live trial retained fresh camera frames and real policy chunks.
All 44 monitor decisions remained in progress while the bottle stayed on the
source. Deadline interruption invalidated the action epoch and confirmed
PLANNER hold; it did not automatically reset or retry. These observations
close this observed false-success case, not general live-monitor reliability.
Successful live placement and fully hidden carried-bottle footage remain open.

Two earlier launch failures are retained: the inference environment lacked
PyYAML, and starting native prewarm before the checkpoint server was ready
exhausted its policy deadline. `gear_sonic[inference]` now declares PyYAML.
PyYAML 6.0.2 was installed into the existing workstation inference environment,
and its harness/real-inference imports pass. No other package changed. The
driver now waits for server readiness before starting native inference; the
10-second prewarm/policy deadline is unchanged. Model loading is a separate
startup prerequisite. The final policy process uses one OpenMP/MKL thread.

Independent review found that the prototype's stool-contact check alone could
accept side contact or support by a different robot link. Both saved live
trials failed that check, so their source-table conclusions are unaffected.
The current probe labels it `destination_contact_only` and keeps placement
acceptance closed. A future positive must establish fresh advancing samples,
stable pose/velocity, support on the destination top and no robot contact.
[As-run sources](as-run-live-probe/bottle_mission_probe.py) are retained separately
from this reporting correction. Earlier failed launch prototypes were superseded;
their exact source revisions were not preserved.

Reproduce a diagnostic live trial from the VoLoAgent worktree with the local
services stopped and a fresh output directory. This runs simulation only and
returns nonzero while physical placement acceptance is unavailable:

```bash
.venv/bin/python docs/artifacts/g1_harness_20261002/bottle_mission_probe.py \
  --native /home/jihun/work/VoLoAgent/.worktrees/g1-harness-native \
  --profile /home/jihun/work/VoLoAgent/.worktrees/g1-harness/configs/g1/workstation.yaml \
  --scene /home/jihun/work/VoLoAgent/.worktrees/g1-harness/docs/artifacts/g1_harness_20261002/bottle-scene/scene.xml \
  --output /tmp/g1-bottle-live-NEW \
  --env-file /home/jihun/work/VoLoAgent/.env
```

## Actual VLA-loop faults

These checks run the unchanged `run_vla_inference.main`, its worker, observation
preparation, LoopHooks, IPC and sole action publisher against the real C++
controller and MuJoCo on DDS `lo`. The policy and camera inputs are scripted;
the selected checkpoint is not used in these fault cases.

| Case | Result | Scope |
| --- | --- | --- |
| [Policy stall](actual-loop-policy-stall/result.json) | Passed | Interruption after about 2.008 s; advancing PLANNER feedback confirms hold |
| [Fresh late result during reset](actual-loop-late-reset/result.json) | Passed | Valid chunk consumed at 92.5 ms while RESETTING; measured reset completes |
| Scripted latent publication | Zero attempts, zero `pose:v4` messages | Separate queue/pack instrumentation and action-wire observations |

The late-result case increments the inference epoch from 3 to 5, but the result
is consumed by the **paused queue drain**. It demonstrates discard during reset;
it does not isolate the active loop's epoch comparison. The existing nine
cross-process checks cover old-epoch rejection with synthetic inputs. Independent
artifact review caught this distinction, and the assertion wording was corrected.

The publication guard logs an attempted pack before raising. The probe separately
requires zero such attempts, so the guard cannot turn an attempted publication
into a pass. The clean cases exited normally and stopped their owned processes.

Three preliminary driver failures are retained: startup was sent before C++
initialization, a new skill was requested from INTERRUPTED, and reset was
requested from INTERRUPTED. Those requests violate readiness/lifecycle rules.
The final driver runs each fault case in a fresh executor and follows the
permitted pause/reset path; it does not change product lifecycle behavior.

Reproduce from the VoLoAgent worktree, using a fresh output path for each case:

```bash
.venv/bin/python docs/artifacts/g1_harness_20261002/native_loop_fault_probe.py \
  --native /home/jihun/work/VoLoAgent/.worktrees/g1-harness-native \
  --profile /home/jihun/work/VoLoAgent/.worktrees/g1-harness/configs/g1/workstation.yaml \
  --output /tmp/g1-loop-fault-NEW --scenario policy-stall
# Use --scenario late-result-reset for the second independent case.
```

## Turning diagnosis

All trials retain the 5-degree measured lead, 10-degree/s command rate,
3-degree physical tolerance, 0.5-second dwell and 10-second deadline.
No runtime control change was made for these trials. Walking still means
duration completion plus stop acknowledgement; it does not measure distance.

| Trial | Result | Physical error | Actual wrap crossing |
| --- | --- | --- | --- |
| [Positive 15 degrees](shadow-positive/result.json) | Passed in 3.652 s | 2.153 degrees | No |
| [Negative 15 degrees](shadow-negative/result.json) | Passed in 4.326 s | 1.334 degrees | No |
| [Negative, near pi](shadow-near-pi-negative/result.json) | Deadline interruption at 10.015 s | 9.884 degrees | No |
| [Negative, crossing pi](shadow-cross-negative/result.json) | Lead/rate conflict interruption at 4.750 s | 6.885 degrees | Yes |
| [Positive, crossing pi](shadow-cross-positive/result.json) | Passed in 2.349 s | 2.746 degrees | Yes |

Initial spawn yaw does not establish a wrap crossing: startup and the preceding
walk change measured heading. The near-pi negative trial entered the turn at
+176.917 degrees despite spawning at -179 degrees, so it did not cross the
boundary. The next negative trial entered at -168.057 degrees and did cross.

During every saturated sample, an immediate maximum-trim shadow calculation
requested the same heading as the actual bounded controller. Increasing outer
trim cannot provide a stronger legal command in those samples. The shadow is
a single-step calculation, not an alternative controller rollout.

Earlier positive/negative failures remain recorded at 3.314/3.761 degrees.
The new passes establish that the behavior varies with state. Failure to
respond consistently lies downstream of the bounded facing command; planner,
decoder and contact contributions still need separation. Simulation speed was
about 0.983–0.984 times wall time in the new trials and 0.984 in the earlier
failed signed trials, which does not explain their different outcomes.
[Computed diagnostics](turn-diagnosis.json) retain values and raw trace links.

## Editable sample scene

[Layout JSON](scene_layout.json), [scene XML](bottle-scene/scene.xml),
[overview](bottle-scene/overview.png), [ego view](bottle-scene/ego_view.png), and
[smoke result](bottle-scene/result.json) are available for inspection.

The user supplied an 80 cm source table, approximately 1 m in front of the G1,
and a 70–90 cm green stool about 50 cm to the table's left. They also offered to
move the physical station to a sample layout. This sample uses 80 cm for both
surfaces and places the source closer to the robot. It is not a measured
reconstruction of the current station. The stool is on the robot's right,
which appears on the viewer's left in the overview, matching the registered
right-table prompt. Check that coordinate convention when matching the station.

Coordinates are meters from the initial pelvis XY projection onto the floor;
X points forward, Y to the robot's left, and Z up.

| Item | XY center | Surface / size |
| --- | --- | --- |
| Source table | (0.50, 0.20) | Height 0.80; tabletop 0.474 × 0.60 |
| Green stool | (0.35, -0.60) | Height 0.80; diameter 0.50 |
| Bottle on source | (0.45, 0.00) | Height 0.15; diameter 0.06 |

The bottle is a primitive cylinder with assumed dimensions and density;
its mass is about 42 g. It does not model the real clear bottle, yellow cap,
contents, friction or appearance. The inherited SONIC hand model itself warns
of simulation instability. Do not treat this scene as calibrated grasp evidence.

The compiled scene preserves all 43 actuator names and joint associations,
29 body and 7 joints per hand, robot state addresses, and the mounted
`head_camera` pose/FOV. The free bottle adds 7 position and 6 velocity entries.
The native simulator's body/hand lookup remains correct.

Both passive two-second support checks pass. Deliberately overlapping the
bottle with each fingertip produces contact, proving collision filtering is
enabled; this does not show a stable grasp. Three timestamped JPEG messages
cross the actual native image subprocess and decode through the camera schema.
These repeated copies of one rendered image test transport, not live acquisition
freshness. The [initial scene](bottle-scene-initial/result.json) is retained;
it preceded the user-supplied heights and editable layout.

Rebuild and check an adjusted layout from the VoLoAgent worktree:

```bash
env MUJOCO_GL=egl \
  PYTHONPATH=/home/jihun/work/VoLoAgent/.worktrees/g1-harness-native \
  /home/jihun/work/GR00T-WholeBodyControl/.venv_sim/bin/python \
  docs/artifacts/g1_harness_20261002/bottle_scene_probe.py \
  --native /home/jihun/work/VoLoAgent/.worktrees/g1-harness-native \
  --layout docs/artifacts/g1_harness_20261002/scene_layout.json \
  --output /tmp/g1-bottle-scene-NEW
```

## Training-review queue

[Episode list](training-review/episodes.jsonl) records all 179 source episodes,
their 174 retained mappings, durations, prompt and outcome-label provenance.
Source episode 5 is the user-confirmed unrecovered failure; source episode 18
has session-inspected placement onto the user-confirmed destination. The other
172 retained episodes remain unreviewed. No data, labels or checkpoint changed.

Source episode 14 becomes prepared episode 13. Its 27,244 frames / 544.88 s are
9.934% of retained frames. [Twenty-second-spaced inspection](training-review/episode-14-contact-sheet.png)
shows many stationary floor/table views; prioritize it for idle-segment review.
Frame share is not a measured optimizer sampling probability or proof of an
effect on the checkpoint. [Summary](training-review/summary.json) lists the next
longest recordings for review. The 37-frame source episode remains present,
although the experiment README states it yields no 40-step training window.

Inspect success, unrecovered failure, recovery and unclear outcomes before
choosing segments for another fine-tune. New subgoal prompts still require
demonstrations and held-out evaluation before registration as executable skills.

## Remaining acceptance work

- Separate planner, decoder and contact effects in negative/wrapped turning;
  require repeatable signed and wrapped turns at the existing limits.
- Validate fully hidden carried-bottle footage. Additional coarse inspection of
  source episodes 1, 2, 3, 14, 17 and 18 did not supply that labeled case.
  The corrected recorded Genon set has ten valid judgments, eight negatives
  with zero false-completes and one two-frame release confirmation.
- Match bottle appearance, mass, friction and scene geometry to the station,
  then demonstrate real-checkpoint grasp/placement and successful live monitor
  confirmation. Independently require stable released top support. Current
  live trials establish source-table failure and corrected negative monitoring.
- Review the training outcomes/segments using the prepared queue before another
  training run. The finite prompt pool currently has one trained bottle prompt.
- After relevant gates pass, perform supervised hardware observe-only,
  pause/reset, bottle/reset and locomotion trials. The
  [deploy skill](/home/jihun/.codex/skills/deploy/SKILL.md:19) requires:
  "Never start `real` deployment without explicit user confirmation that hardware,
  safety zone, and E-stop operator are ready."

Standing reset preserves the heading at reset entry. Returning to a saved floor
location needs localization and remains outside this harness. Dashboard work
is deferred to another session.

## Software verification and retained evidence

The required G1 test run passes **50 tests, zero skips, in 29.14 s**, including
the cross-process cases. It uses the current profile and explicit native
checkout/interpreter paths. The preceding sandbox attempt failed eight IPC
tests because socket binding was denied; the rerun outside the sandbox passes.
Native harness/launcher tests pass **23 tests in 0.20 s** after declaring PyYAML.
That dependency fix is committed as `73560de` in the isolated native branch.
The earlier full-suite and C++ results remain in the previous report; they
were not rerun or relabeled as fresh full-suite results.
Commands and captured counts are saved in [software checks](software-checks.json).

```bash
env PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest \
  tests/test_g1*.py \
  --g1-native-checkout=/home/jihun/work/VoLoAgent/.worktrees/g1-harness-native \
  --g1-native-python=/home/jihun/work/VoLoAgent/.worktrees/g1-harness-native/.venv/bin/python \
  --require-g1-e2e -q
```

All owned simulator, inference, policy and controller processes were stopped.
The simulation ports are released and no GPU compute job remains from these
trials. JSON/JSONL results retain their original temporary paths. Asset/source
hashes and output-directory mappings are recorded in [provenance](provenance.json).
The original dirty native checkout's source files were preserved.
Captured logs and model XML retain their original trailing whitespace and hashes.
This artifact directory disables Git whitespace checks for those two file types;
authored Python, configuration and prose still receive the normal checks.
