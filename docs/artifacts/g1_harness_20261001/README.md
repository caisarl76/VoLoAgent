# G1 harness software validation — 2026-10-01

The software implementation is ready for SONIC simulation. Hardware acceptance
is pending. No robot command or SONIC control loop was launched in this session.

## Implementation

VoLoAgent: `work/g1-harness-20261001`, code/test commit `d5b4aa7`, base
`bff8266`, checkout `/home/jihun/work/VoLoAgent/.worktrees/g1-harness`.

Native executor: `work/volo-g1-harness-20261001`, commit `00dc529`, base
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
| VoLoAgent full suite | 604 passed, 6 existing skips | G1 integration cases were required and not skipped |
| Native full suite | 607 passed | Includes existing MuJoCo hand-contact test; not SONIC harness physics validation |
| Dedicated cross-process G1 gate | 9 passed, zero skips | Synthetic policy/measurements; memory-only publisher |
| Ruff on modified Python files | Passed | Mirrors deliberately share the same source formatting |
| Installed entry point | skills and help passed | No lease or robot connection |
| Native imports and G1 model construction | Passed | Policy client import and Pinocchio model only |
| Recorded checkpoint queries | Passed: 4.772 s cold, 0.128 s warm | Model serving, shape and finite values only |
| Fresh independent code review | Approved | Physical behavior and visual accuracy excluded |
| SONIC simulation transitions/lost input | Pending | No SONIC simulation control loop run |
| Completion-monitor labeled clips | Pending | Acceptance labels and chosen vision endpoint not supplied |
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
with no Critical or Important findings outstanding. Later changes add only the
takeover fixture/test, environment ignore entry and these artifacts.

Task 8 remains open for the following:

1. Run bounded, headless SONIC simulation on loopback with separate action/state
   ports, validating measured reset and turn settling, walking deadlines,
   cancellation, stale telemetry, worker stalls and process loss.
2. Verify the deployed C++ manager's lost-input behavior. Source
   `zmq_manager.hpp` switches to IDLE after a 1-second planner-input timeout and
   clears upper-body and hand overrides. A preserved grasp across executor
   death has not been established. Python lease-expiry tests keep the executor
   alive; they cannot prove this separate process-death behavior.
3. Supply labeled success, failure, held-bottle and occlusion clips and a chosen
   vision endpoint. Record sample counts and require zero false-completes in
   that acceptance set. Scripted monitor tests measure control logic.
4. Validate the actual bottle/table scene in compatible simulation or document
   the gap before supervised physical evaluation.
5. Use the existing deployment workflow to confirm robot readiness and then
   perform observe-only, pause/reset, bottle/reset, and locomotion trials.

Standing reset retains the heading at reset entry. Returning to a saved floor
location requires localization and is outside this implementation.
