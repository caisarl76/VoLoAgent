# G1 harness validation — updated 2026-10-02

Software checks pass. SONIC/MuJoCo control ran on loopback; measured reset and
hand stability gates failed. Recorded visual evaluation is partial. Hardware
acceptance is pending, and no physical robot was connected or actuated.

## Implementation

VoLoAgent: `work/g1-harness-20261001`, code/test commit `d5b4aa7`, base
`bff8266`, checkout `/home/jihun/work/VoLoAgent/.worktrees/g1-harness`.

Native executor: `work/volo-g1-harness-20261001`, commit `c77badc`, base
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
| Native full suite | 616 passed | Includes existing MuJoCo hand-contact test; physical harness gates are separate |
| Dedicated cross-process G1 gate | 9 passed, zero skips | Synthetic policy/measurements; memory-only publisher |
| Ruff on modified Python files | Passed | Mirrors deliberately share the same source formatting |
| Rebuilt C++ controller tests | 11 passed | Motor mapping, measured hand validity, timeout retention, mode boundaries, stop/E-stop, VR filtering |
| Installed entry point | skills and help passed | No lease or robot connection |
| Native imports and G1 model construction | Passed | Policy client import and Pinocchio model only |
| Recorded checkpoint queries | Passed: 4.772 s cold, 0.128 s warm | Model serving, shape and finite values only |
| Fresh independent code review | Approved for current software | Physical behavior and visual accuracy excluded |
| SONIC cancellation/lease loss/stale feedback | Passed | Actual native hooks with loopback controller; no VLA manipulation |
| SONIC measured standing reset | Failed: 0.531 rad maximum error at timeout | Required tolerance 0.05 rad; walk/turn sequence never starts |
| SONIC measured hand stability on input loss | Failed: 0.087 rad drift over 2.5 s | Command targets retain; physical tolerance 0.05 rad not established |
| Completion-monitor recorded frames | Partial: 9/10 valid replies, 0 false-completes in 6 valid negatives | 2 positive-label disagreements, 1 timeout; provisional apple/box labels |
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

Logs are [checkpoint-server.log](checkpoint-server.log),
[checkpoint-smoke-cold.log](checkpoint-smoke-cold.log),
[checkpoint-smoke-warm.log](checkpoint-smoke-warm.log), and
[dataset-stats.log](dataset-stats.log). The server's log records the selected
model path and three loaded shards. This is launch evidence, not an RPC identity
attestation from an arbitrary external server. The temporary server was stopped
after both queries; port 15558 has no listener left by this validation.

## SONIC controller and measured behavior

The bounded [sonic_sim_probe.py](sonic_sim_probe.py) starts the headless MuJoCo
simulator and the rebuilt controller using DDS interface `lo`, GPU 1, action
port 11556 and state port 11557. It uses actual native control/reset hooks and
a sole planner publisher. It has no VLA policy worker or physical connection.
Both owned process groups are stopped on exit.

Build uses `BUILD_ROS2=OFF`, `BUILD_DEPLOY_TESTS=ON`, cached GoogleTest,
TensorRT 10.13 and existing SONIC decoder/encoder/planner assets. The controller
SHA-256 in both final results is
`2d42a14cb0ef9c1fef4d00d44da0b7a34b1593b3af53d9df6dbb45daccac4e0b`.

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

- [Final fault result](sonic-sim-receipt-faults/result.json): cancellation,
  coordinator lease expiry and stale body feedback pass. After 3 seconds of
  target preparation and 2.5 seconds without planner input, measured hand drift
  is 0.086657 rad, exceeding the 0.05-rad limit.
- [Final transition result](sonic-sim-receipt-transitions/result.json): fresh
  PLANNER acknowledgement arrives, but standing reset times out at 15 seconds
  with maximum joint error 0.530868 rad. The harness interrupts and does not
  proceed to walking or turning.

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
  --output /tmp/g1-sonic-check --scenario faults
```

Omit `--scenario faults` for the reset/walk/turn transition gate. Both require
the simulator workflow's existing readiness/launch authorization.

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

These are provisional apple/box frame checks. They do not extend the checkpoint's
trained prompt registry or validate bottle placement and two-frame completion.

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

Task 8 remains open for the following:

1. Diagnose measured SONIC standing and hand tracking against a standalone
   stationary-planner baseline. Keep the 0.05-rad acceptance limit. Rerun reset
   and input-loss stability after correcting the underlying tracking issue.
2. Once reset settles, run bounded walking, turning and coordinator-loss checks
   in the actual simulator. Their synthetic integration checks already pass.
3. Review frame labels and add clear post-release tails, bottle placement,
   full occlusion and unrecovered failure examples. Resolve the Genon deadline
   failure and positive disagreements, then require zero false-completes in
   the stated acceptance set and verify two-frame confirmation.
4. Validate the actual bottle/table scene in compatible simulation or document
   the gap before supervised physical evaluation.
5. Use the existing deployment workflow to confirm robot readiness and then
   perform observe-only, pause/reset, bottle/reset, and locomotion trials.

Standing reset retains the heading at reset entry. Returning to a saved floor
location requires localization and is outside this implementation.
