# G1 harness validation — updated 2026-10-02

Software checks pass. SONIC/MuJoCo standing reset, walking and tested interruption
paths pass. Turning remains unaccepted: the latest positive and negative
15-degree turns stop short and interrupt at the deadline. Recorded bottle
monitoring has ten valid judgments, eight negatives without false completion,
and one two-frame release confirmation. Full carried-bottle occlusion, live
manipulation and hardware acceptance remain open. No physical robot was connected
or actuated. The dashboard is deferred.

## Implementation

VoLoAgent: `work/g1-harness-20261001`, code/test commit `d5b4aa7`, base
`bff8266`, checkout `/home/jihun/work/VoLoAgent/.worktrees/g1-harness`.

Native executor: `work/volo-g1-harness-20261001`, commit `d513de5`, base
`7c6d710`, checkout `/home/jihun/work/VoLoAgent/.worktrees/g1-harness-native`.
The original native checkout's tracked edits and untracked files were preserved.

The shared profile SHA-256 is
`17e415bf4ebd63e771e3befe9c0c95ce360ad36ec03c94f1f2b535a89406b71f`.
Native and agent data-only contracts, profile validation, and wire fixtures are
checked for parity. Harness mode is opt-in; locomotion has a second opt-in on both
the coordinator and executor.

## Results

| Gate | Result | Practical limit |
| --- | --- | --- |
| VoLoAgent full suite (2026-10-01) | 604 passed, 6 existing skips | G1 integration cases were required and not skipped |
| Native full suite | 626 passed in 7.18 s | Includes existing MuJoCo hand-contact test; hardware gates are separate |
| Dedicated cross-process G1 gate | 9 passed, zero skips, 27.23 s | Synthetic policy/measurements; memory-only publisher |
| Ruff on modified Python files | Passed | Mirrors deliberately share the same source formatting |
| Rebuilt C++ controller tests | 11 passed | Motor mapping, measured hand validity, timeout retention, mode boundaries, stop/E-stop, VR filtering |
| Installed entry point | skills and help passed | No lease or robot connection |
| Native imports and G1 model construction | Passed | Policy client import and Pinocchio model only |
| Recorded checkpoint queries | Passed: 4.772 s cold, 0.128 s warm | Model serving, shape and finite values only |
| Fresh tracking review | Two important findings corrected with regressions | Profile tolerance and fresh probe output; main agent checked final diffs |
| SONIC cancellation/lease loss/stale feedback | Passed | Native control modules through a probe adapter; no VLA worker |
| SONIC measured standing reset | Repeated passes below 0.05 rad | 0.5-s fresh dwell; no long stationary or grasp trial |
| SONIC backward walk | Passed: 1 s | Duration/stop acknowledgement; no measured-distance claim |
| SONIC turn, current lead enforcement | Open: latest errors 3.314 / 3.761 degrees | Both exceed the unchanged 3-degree limit and interrupt at 10 s |
| SONIC measured hand stability on input loss | Passed: 0.013382 rad drift over 2.5 s | No object held; does not establish whole-body stability or grasp retention |
| SONIC 179-degree startup / wraparound | Prepared positive wrap passed before latest lead fix; negative wrap missed | Signed physical repeatability remains open |
| Completion-monitor recorded frames | Partial: 9/10 valid replies, 0 false-completes in 6 valid negatives | 2 positive-label disagreements, 1 timeout; provisional apple/box labels |
| Bottle recorded-frame monitor | 10/10 valid; zero false-completes in eight negatives; one two-frame confirmation | User-confirmed destination and failed episode; full carried-bottle occlusion missing |
| Actual bottle manipulation simulation | Compatible scene not configured | Legacy bottle scene needs adaptation; placement success unvalidated |
| Physical bottle/reset/locomotion | Pending | Robot readiness confirmation and supervised trials required |

The full tests use disabled third-party pytest auto-loading. Native tests use a
separate Python 3.10 environment importing the existing teleoperation dependencies,
plus cached safetensors 0.7.0. VoLoAgent uses its editable dev install in Python
3.13. Native serving uses the existing Isaac-GR00T environment.

The worktree initially contained Git LFS pointers in mesh files. The initial
native full suite reported 604 passes and one mesh-decoder failure. Sixty deploy
meshes and 65 robot-model meshes were hydrated from local files after matching
their SHA-256 against the pointer OIDs. No mesh content change was committed.
The full native suite then passed, and the actual G1 model constructor succeeded.
Three SDK library blobs were also hydrated from locally verified originals for
the C++ rebuild; their LFS pointer content is unchanged.

Full software commands:

```bash
# VoLoAgent checkout
env PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest tests \
  --g1-native-checkout=/home/jihun/work/VoLoAgent/.worktrees/g1-harness-native \
  --g1-native-python=/home/jihun/work/VoLoAgent/.worktrees/g1-harness-native/.venv/bin/python \
  --require-g1-e2e -q

# Current dedicated integration gate, VoLoAgent checkout
env PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest \
  tests/test_g1_harness_e2e.py \
  --g1-native-checkout=/home/jihun/work/VoLoAgent/.worktrees/g1-harness-native \
  --g1-native-python=/home/jihun/work/VoLoAgent/.worktrees/g1-harness-native/.venv/bin/python \
  --require-g1-e2e -q

# Native checkout
env PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  PYTHONPATH=/home/jihun/work/VoLoAgent/.worktrees/g1-harness-native \
  .venv/bin/python -m pytest gear_sonic/tests -q
```

The nine cross-process checks cover fixture parity, bottle completion/reset, late
old-epoch result rejection, frozen camera data, missing planner acknowledgement,
coordinator loss during manipulation, bottle/reset/walk/turn sequencing, owner
loss mid-walk, and operator takeover. They use actual RPC/control/coordinator
modules. The sample mission below retains the synthetic camera images and
scripted vision responses; it is not evidence of physical placement.

- [Sample result](sample-mission/result.json)
- [Coordinator events](sample-mission/events.jsonl)
- [Native memory-sink events](sample-mission/native.jsonl)

These files were copied without rewriting their recorded temporary source paths.
Associated image files are copied into the same sample-mission directory.

## Checkpoint serving evidence

Checkpoint:
`/mnt/data/jihun/models/pnp_bottle_260916_n17_raw_gpu7/checkpoint-20000`.
Prompt: `pick drink bottle and place it on the right table`.

Port 15558 was initially stopped. With approval, a temporary server loaded this
checkpoint on physical GPU 1 and bound only to `127.0.0.1:15558`. Both queries
used the experiment's original `smoke_inference.py`, recorded episode 18 and
native GR00T parsing. Assertions required finite arrays of:

- `motion_token: [1,40,64]`
- `left_hand_joints: [1,40,7]`
- `right_hand_joints: [1,40,7]`

The temporary derived dataset was absent. The experiment's `prepare_dataset.py`
reconstructed 174 retained episodes and 274,241 frames from the immutable source,
omitting source IDs 13, 94, 99, 106 and 109. The first loader check found missing
`meta/stats.json`; the official `gr00t.data.stats.generate_stats` generated it
before the successful queries. No training run or policy update was performed.

The user's unrecovered failed episode 5 is retained in this prepared corpus with
the same task prompt. Its video matches the raw training source by SHA-256.
The [training-data audit](bottle-training-data-audit.md) records the curation
job; this finding does not establish that the checkpoint learned the failure.

Logs are [checkpoint-server.log](checkpoint-server.log),
[checkpoint-smoke-cold.log](checkpoint-smoke-cold.log),
[checkpoint-smoke-warm.log](checkpoint-smoke-warm.log), and
[dataset-stats.log](dataset-stats.log). The server's log records the selected
model path and three loaded shards. This is launch evidence, not an RPC identity
attestation from an arbitrary external server. The temporary server was stopped
after both queries; port 15558 has no listener left by this validation.

## SONIC controller and measured behavior

The bounded [sonic_sim_probe.py](sonic_sim_probe.py) starts the headless MuJoCo
simulator and the rebuilt controller using DDS interface `lo`, GPU 0, action
port 11556 and state port 11557. It runs native HarnessControl, StandingReset
and BoundedPlanner through a probe Hooks adapter and a sole planner publisher.
It does not run the inference main loop's LoopHooks, VLA policy worker or camera,
and has no physical connection. Combined real-controller/VLA-worker fault
validation remains open.
Both owned process groups are stopped on exit.

Build uses `BUILD_ROS2=OFF`, `BUILD_DEPLOY_TESTS=ON`, cached GoogleTest,
TensorRT 10.13 and existing SONIC decoder/encoder/planner assets. The controller
SHA-256 in both final results is
`2d42a14cb0ef9c1fef4d00d44da0b7a34b1593b3af53d9df6dbb45daccac4e0b`.
Final corrected runs use SONIC v1.1, as used for bottle data collection. Each
result records the encoder, decoder and observation-config hashes. GPU 1 has
an unrelated job, which was left untouched.

The controller now emits explicit absolute motor-order body measurements and
real measured hand positions. Original hand DDS receipt times survive logging;
absent, disabled or older-than-500-ms hands produce empty measured arrays. The
harness requires these measurements and effective opt-in timeout capability
before granting authority, and revokes authority if either disappears.

`--harness-planner-hold` retains established commanded arm/hand positions on a
one-second planner-input timeout, zeros upper-body velocity and sets locomotion
to IDLE. Default behavior still clears overrides. Stop, emergency stop and mode
changes clear targets; fresh primed planner entry remains supported. Physical
stability and object retention require separate acceptance checks.

- [Corrected fault result](sonic-sim-final-faults/result.json): cancellation,
  coordinator lease expiry and stale body feedback pass. After 3 seconds of
  hand-target preparation and 2.5 seconds without planner input, measured drift
  is 0.013382 rad, below the unchanged 0.05-rad limit. A prior corrected run
  measured 0.013496 rad.
- [Earlier corrected transition result](sonic-sim-final-transitions/result.json),
  native `e1485e0`: reset
  completes in 1.401 seconds at 0.041892-rad maximum error, followed by a
  one-second backward walk, a measured 15-degree turn, and lease-loss stopping
  during another walk. Turn completes in 2.549 seconds at 2.755-degree error.
  A prior corrected transition run also passed. These positive passes precede
  the latest lead-bound fix and do not establish current turn acceptance.
- Current native `d513de5` [positive](sonic-sim-lead-positive/result.json) and
  [negative](sonic-sim-lead-negative/result.json) turns miss at 3.314 and
  3.761 degrees respectively and interrupt at 10 seconds. Reset passes at
  0.028170 / 0.025301 rad; one-second walking passes in both runs.
  [Slower negative turning](sonic-sim-slow-negative/result.json) at 5 degrees/s
  also misses, at about 6.242 degrees. Physical turning remains unaccepted.
- Earlier [fault](sonic-sim-receipt-faults/result.json) and
  [transition](sonic-sim-receipt-transitions/result.json) results retain their
  failures: 0.086657-rad hand drift and 0.530868-rad reset error, respectively.
  They used the previous simulation preparation; they are not overwritten.

The standalone standing baseline also missed the target. Headless MuJoCo kept
its elastic support enabled, lifting the pelvis to about 0.965 m. Releasing the
support after startup restores floor contact and reduces reset error to about
0.069 rad with release assets; v1.1 still has about 0.077 rad steady error.
Those [diagnostic results](tracking-diagnosis) led to bounded tracking trim.
The acceptance limit is unchanged.

Harness reset now trims upper-body references only after the nominal standing
ramp reaches its target. Trim and measured lead remain capped at 0.15 rad, slew
at 0.5 rad/s, and physical settling at the profile tolerance for 0.5 seconds.
The 15-second deadline and hand targets are unchanged. Turn trim similarly
preserves the 5-degree lead, 10-degrees/s rate, original physical heading goal,
3-degree tolerance, 0.5-second dwell, and 10-second deadline. Signed offset
plant regressions, including +/-pi, fail before these corrections and pass after.

A negative-turn trace exposed backward body drift allowing a held reference to
lead by 5.332 degrees. Commit `d513de5` retracts toward the measured lead interval.
If a feedback jump makes rate and lead bounds incompatible, it interrupts
before assigning the reference and requests a measured hold. Four regressions
cover reference retraction, incompatible bounds and lifecycle interruption.
[Sampled lead analysis](turn-tracking-analysis.json) shows current commands stay
within 5 degrees to numerical precision. Reliable physical tracking within that
bound remains the bottleneck; no tolerance or deadline was relaxed.

In the passing transition runs, reset immediately precedes walking and turn
immediately precedes another walk. These establish completion dwell and subsequent
handoff, not sustained stationary accuracy against an earlier skill's target.
The hand-input-loss test contains no grasped object and checks hand drift only.

The 179-degree spawn exposed the temporary support spring's yaw-zero reference.
After alignment, releasing from an arbitrary supported pose still fails reset
at 1.608 rad. A no-lease baseline stands; a bounded operator standing ramp before
lowering support resolves that preparation issue. The prepared positive wrap
passes on `e1485e0` at 2.628-degree error, crossing pi. The prepared negative wrap
misses by 10.970 degrees; negative turning also misses at ordinary headings.
A gait-mode diagnostic failed and was not adopted. Native turns still request
zero movement/speed in IDLE planner mode. Signed turn repeatability remains open.

An earlier measured-hand run showed 0.012148-rad drift, but the final run did
not repeat that result. Earlier binaries also had ambiguous body ordering and
reported commanded hands as measurements; those runs cannot close physical
acceptance. Exploratory attempts are preserved under
`/tmp/volo-g1-harness-investigation-20261002/`.

```bash
# Native checkout; local ZMQ tests require normal socket access.
gear_sonic_deploy/target/release/run_tests \
  --gtest_filter=HarnessTelemetry.*:HarnessTimeout.*:Vr3PtSafetyFilter.*

env PYTHONPATH=/home/jihun/work/VoLoAgent/.worktrees/g1-harness-native \
  .venv/bin/python \
  /home/jihun/work/VoLoAgent/.worktrees/g1-harness/docs/artifacts/g1_harness_20261001/sonic_sim_probe.py \
  --native /home/jihun/work/VoLoAgent/.worktrees/g1-harness-native \
  --profile /home/jihun/work/VoLoAgent/.worktrees/g1-harness/configs/g1/workstation.yaml \
  --output /tmp/g1-sonic-check --scenario faults \
  --gpu 0 --release-band --sonic-policy sonic_v1_1
```

Omit `--scenario faults` for the reset/walk/turn transition gate. Both require
the simulator workflow's existing readiness/launch authorization.
Output directories must be new; the probe rejects existing evidence before
opening sockets or launching processes. The worker saves 10-Hz MuJoCo ground
truth alongside the probe's measurements and command/status events.
Current transition reproduction adds `--prepare-standing`; signed and wrapped
probes use `--initial-yaw-deg 0|179|-179`, `--turn-angle-deg 15|-15`, and
`--turn-rate-deg-s 10` (or 5 for the slower diagnostic). The optional
`--diagnostic-turn-mode` is an experimental probe override, not native acceptance.
[Simulation provenance](simulation-provenance.json) maps saved runs to native
commits, parameters and result hashes. Scripts evolved during the investigation;
their final hashes do not attest earlier versions.

## Recorded visual evaluation

The [Genon report](vision-genon/README.md) records the two supplied manipulation
episodes, ten frame labels and raw replies. Evaluation uses
`openai/gpt-5.6-sol` through `https://api.genon.ai/v1`. Median valid decision
latency is 2.688 seconds; one held-apple frame exceeds the 10-second deadline.
That frame passes a single recorded retry in 3.260 seconds; the initial timeout
remains part of the acceptance record.
The model keeps the missed grasp and transport frames incomplete. It recognizes
one box placement frame, but declines two other positive labels because release
is unclear. The wide-view negative does not cover full object occlusion.

These apple/box checks remain provisional and do not extend the trained prompt
registry. The [bottle success report](vision-bottle-genon/README.md) uses actual
training episode 18 with the exact bottle prompt. All six judgments match its
provisional annotations, with no false-completes among four negative frames.
At 37 and 38 seconds, distinct released-bottle images produce streaks 1 and 2;
only the second completes. All six replies arrive in 2.667–3.775 seconds.
The user confirmed that the green stool is the intended right-table destination
on 2026-10-02; the recorded replies and original provenance remain unchanged.
The [failed episode report](vision-bottle-failure-genon/README.md) evaluates the
user's episode 5 at 12, 17, 18.8 and 19.4 seconds. All four replies remain
`in_progress`, with zero completion candidates or confirmations. This checks
false success on these samples; immediate failure classification is unvalidated.
Target visibility loss after a grasp attempt is covered, while full occlusion
of an established carried bottle is still missing. Persistent target identity
and live camera/mission behavior are unvalidated.

[Combined bottle results](bottle-monitor-validation.json): ten valid replies,
eight negatives, zero false-completes, zero positive-label disagreements,
one two-frame confirmation, and latency 2.667–4.967 seconds. Direct session
inspection supplies frame annotations; the user confirmed the destination and
failed episode. This small set does not establish general reliability.
Historical timeouts and unclear apple/box judgments remain visible.

The [bottle-scene inspection](bottle-scene-gap.md) records the simulation gap:
the SONIC default has no task objects; the legacy bottle XML has one front
table, a world-fixed camera, a different right-hand actuator order, and no
separate right-side destination. Both XML models load, but neither currently
validates the actual trained manipulation mission.

## Review and remaining gates

Review found seven issues: pause takeover resuming VLA, reset advancing after
planner acknowledgement timeout, worker camera receipt rebasing, telemetry
restart recovery, silent active inference retries, lost-start-reply cleanup,
and internal reset IDs passing whole-plan validation. Regression tests failed
before the fixes and passed afterward.

Follow-up review caught two camera timing races. Inference now shares an atomic
raw frame and local receipt time with the continuously sampled camera cache.
Freshness tolerates a concurrent local receipt update after the caller sampled
its clock. The final review approved runtime commits `49043dc` and `49b87c5`
with no Critical or Important findings outstanding on 2026-10-01. Resumed review
of the controller changes found stale targets across mode boundaries and missing
or stale hands masquerading as measured-open. Both are fixed and covered by
regressions; fresh final review approves `c77badc` for software correctness.

The fresh tracking review found a hardcoded reset deadband that prevented
tighter profiles from settling, and probe output reuse that could overwrite
evidence or release support early. Both regressions were demonstrated RED,
then corrected GREEN. The native full suite and required integration gate pass
on `e1485e0`; the artifact guard also passes. The later `d513de5` feedback-reversal
fix has four regressions, a 626-test native suite and main-agent diff review;
it has not had another independent review. See [review scope](tracking-review.md).
One minor probe-cleanup race remains deferred; all owned processes exited
normally in these completed checks.

Task 8 remains open for the following:

1. Resolve physical SONIC turn tracking without relaxing limits; require
   repeatable positive, negative and wrapped turns. Reset, walking and tested
   fault paths pass; current signed turns still miss the physical tolerance.
2. Combine scripted policy stalls and delayed old-epoch results with the actual
   SONIC control loop. These lifecycle guards pass cross-process tests; the
   current real-controller probe has no VLA worker.
3. Add fully hidden carried-bottle footage and validate the monitor on live
   frames. The failed episode and intended destination are now user-confirmed.
   Require zero false-completes in the chosen set; ten recorded samples are
   not general reliability.
4. Adapt a compatible bottle/right-table scene or retain the documented gap
   before explicitly supervised physical evaluation. Actual placement is unvalidated.
   Audit failed training episodes before further fine-tuning; no dataset was
   modified here.
5. After the relevant acceptance gates pass, use the deployment workflow to
   confirm hardware, safety zone and E-stop operator readiness, then perform
   observe-only, pause/reset, bottle/reset, and locomotion trials.

The dashboard is deferred to another session in the implementation plan.

Standing reset retains the heading at reset entry. Returning to a saved floor
location requires localization and is outside this implementation.
