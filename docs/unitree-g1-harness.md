# Unitree G1 agent harness

Software checks and checkpoint serving pass. The real checkpoint, native loop,
SONIC, simulator camera and Genon monitor now run together. A live false-success
case was corrected by identifying the destination as the separate green stool.
The fresh live trial kept the bottle on the source table and interrupted with
confirmed planner hold at its deadline. Successful pick-and-place, repeatable
turning, full occlusion acceptance and supervised hardware remain open.
See the [current validation report](artifacts/g1_harness_20261002/README.md) for
the scene, raw evidence, software checks and remaining gates.

The agent chooses a registered skill, checks camera evidence, and asks the native
executor to pause and reset. The existing native inference loop publishes every
robot command. RPC, vision monitoring, and the agent have no actuator publisher.

The first manipulation skill is `bottle_to_right_table`. Its exact trained prompt
is `pick drink bottle and place it on the right table`; the expected checkpoint
is `/mnt/data/jihun/models/pnp_bottle_260916_n17_raw_gpu7/checkpoint-20000`.
Adding a name to the registry does not teach the policy a new behavior.

## Checkouts and environments

Implementation lives in two isolated branches:

| Checkout | Branch | Base |
| --- | --- | --- |
| `/home/jihun/work/VoLoAgent/.worktrees/g1-harness` | `work/g1-harness-20261001` | `bff8266` |
| `/home/jihun/work/VoLoAgent/.worktrees/g1-harness-native` | `work/volo-g1-harness-20261001` | `7c6d710` |

Use separate Python environments. VoLoAgent's editable install provides
`volo-g1-harness`; `python -m vlm_orchestrator.harness.g1.cli` also works.
The native runner needs its documented Torch, Pinocchio, ZMQ, camera, and GR00T
client dependencies. For this workstation, the existing policy server uses
`/home/jihun/work/Isaac-GR00T/.venv/bin/python`. The test-only native environment
in the worktree imports the existing teleoperation dependencies. It is not a
replacement for the deployment setup.

The inference extra now declares PyYAML, which the harness profile loader needs.
The existing workstation `.venv_inference` was missing it; PyYAML 6.0.2 is now
installed there and its native harness/inference module imports pass. The live
trial's native worktree environment also imports the actual GR00T client.

The shared profile is `configs/g1/workstation.yaml`. Its SHA-256 covers the exact
file bytes. Pass the same absolute path to both processes. Status reports
`checkpoint_expected`; it cannot attest which model an external server loaded.
Verify the serving log and run a separate non-actuating inference check.
The current profile digest is
`2cb9b06e6283a552eedf1d6f2c3659229908d9309b6e3b5e0270be732c896154`.
Historical trials before the destination correction use a different digest.

## Prepare and inspect

Keep SONIC deployment, controller startup, and emergency stop in the existing
deployment workflow. Use its verified remote/local state and action hosts.
Camera defaults to port 5555, state to 5557, and action to 5556. The policy profile
selects `127.0.0.1:15558`, horizon 40, and 50 Hz. Disable competing XR, teleop,
or another VLA sender using the same controller input.

Rebuild the C++ controller from the native harness checkout and append
`--harness-planner-hold` to its `--input-type zmq_manager` launch. The older
workstation binary lacks the required telemetry contract. The opt-in flag retains
the current planner session's commanded arm/hand position targets when planner
input disappears; lower-body movement becomes IDLE and arm velocity becomes zero.
Stop and mode changes clear those targets. This does not establish physical
pose stability or preservation of a grasp.

Harness authority requires `body_q_measured_motor` (29 absolute motor-order
angles), `harness_planner_hold_enabled=[1]`, and two finite seven-joint measured
hand arrays. C++ measured hand fields use sensor readings, with their original
DDS receipt times. Absent, disabled, or older-than-500-ms hand data produces empty
arrays and invalidates agent ownership. A running body controller alone is
insufficient.

Append these options to the existing native inference launch:

```bash
--harness-endpoint ipc:///tmp/volo-g1-harness.sock \
--harness-profile /home/jihun/work/VoLoAgent/.worktrees/g1-harness/configs/g1/workstation.yaml
```

The native tmux launcher forwards the same options. It retains its existing
deployment and keyboard behavior. The harness never starts motors through RPC.
An operator must start the controller and leave the policy paused. Prewarm runs
asynchronously while paused and discards its output. Expected readiness is a
running controller, fresh advancing telemetry, fresh ego camera frames,
`policy_ready=true`, no owner, and a paused/idle execution state.

From the VoLoAgent checkout:

```bash
.venv/bin/python -m vlm_orchestrator.harness.g1.cli skills
.venv/bin/python -m vlm_orchestrator.harness.g1.cli status
.venv/bin/python -m vlm_orchestrator.harness.g1.cli observe --output /tmp/g1-ego.jpg
```

These commands acquire no lease. IPC is local, private to the user, and protected
against a second executor. After a crashed process, an existing socket is refused.
Only remove a leftover socket after confirming no executor owns it.

## Supervised bottle mission

Configure a real vision model and endpoint; placeholder model names are rejected.
The endpoint must accept the existing OpenAI-compatible multimodal messages.

```bash
.venv/bin/python -m vlm_orchestrator.harness.g1.cli run \
  --skill bottle_to_right_table \
  --vlm-model YOUR_VISION_MODEL \
  --vlm-base-url YOUR_VISION_ENDPOINT \
  --evidence-dir /tmp/g1-missions
```

The monitor requires two distinct fresh frames showing the bottle moved from
the source and released on the separate green stool, the user-confirmed
right-table destination. A bottle on the source table or an unseen destination
does not establish completion. Camera left/right does not identify the stool.
Completion pauses VLA, invalidates its action epoch, then resets
to standing with open hands. Reset completion requires post-request planner
feedback plus measured joint/heading settling for 0.5 seconds. A requested mode
alone is insufficient. Standing reset preserves heading at reset entry; it does
not navigate to an earlier floor position.

Harness resets compensate a steady upper-body tracking offset after the nominal
standing ramp reaches its target. The trim is capped at 0.15 rad and uses half
the profile's joint tolerance as its deadband. The 0.5-rad/s slew, 0.15-rad lead,
physical target, measured dwell, and 15-second deadline still apply. Hands receive
no trim. Ordinary keyboard resets use their existing behavior.

Ctrl-C requests cancellation. Failure, stale data, cancellation, or lease expiry
invalidates VLA actions and requests a measured planner hold preserving the hand
targets. Missing telemetry means hold is unconfirmed. The ownership heartbeat
runs independently of vision calls. There is no automatic retry or resume.
Keyboard control changes revoke agent ownership; recording-only keys preserve it.

Each mission writes `events.jsonl`, the camera frames used for decisions, raw
monitor replies, request IDs, runtime/execution/epoch IDs, and `result.json`.
Review both coordinator evidence and the native serving/runtime logs.

## Bounded repositioning

Locomotion requires `--harness-locomotion` on the native runtime and
`--locomotion` on the coordinator. It is disabled by default. Runtime status
advertises the native setting so a mixed sequence is rejected before its first
skill if locomotion is unavailable.

```bash
.venv/bin/python -m vlm_orchestrator.harness.g1.cli execute \
  --plan-file configs/g1/bottle_then_reposition.json --locomotion \
  --vlm-model YOUR_VISION_MODEL --vlm-base-url YOUR_VISION_ENDPOINT
```

The example places the bottle, performs its automatic standing reset, walks
backward for 1 second at 0.2 m/s, then turns 15 degrees at 10 degrees/s. One
ownership lease covers the whole sequence. A failure ends the sequence.
Walking completion means the commanded duration ended; no measured distance
is claimed. Turns require measured yaw within 3 degrees for 0.5 seconds, followed
by fresh planner acknowledgement of the stopped command. Heading targets lead
measured yaw by at most 5 degrees. Arms and hands keep their measured positions.
After reaching the nominal turn reference, bounded heading trim can compensate
a steady offset; its magnitude is also capped at 5 degrees. Completion still
checks the original physical heading goal, tolerance, dwell, and deadline.
Backward heading drift retracts the reference to preserve the measured lead
bound. If a feedback jump makes the rate and lead limits incompatible, the
executor interrupts and requests a measured hold.
Floor-home navigation requires localization and remains future work.

## Software checks

The cross-process fixture uses the real native lifecycle, IPC server, snapshot
codec, coordinator, and completion monitor. Its policy and measurements are
synthetic and its single publisher is an in-memory sink. It has no robot-action
socket. Run the dedicated gate with explicit checkout/environment paths:

```bash
env PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest \
  tests/test_g1_harness_e2e.py \
  --g1-native-checkout=/home/jihun/work/VoLoAgent/.worktrees/g1-harness-native \
  --g1-native-python=/home/jihun/work/VoLoAgent/.worktrees/g1-harness-native/.venv/bin/python \
  --require-g1-e2e -q
```

Without explicit paths, ordinary tests skip these integration checks with a
reason. The dedicated gate fails for missing paths. Scripted vision responses
verify sequencing, not vision-model accuracy or physical task success.

SONIC v1.1/MuJoCo checks pass measured standing reset, one-second backward
walking, cancellation, lease expiry, stale body feedback and hand stability on
planner-input loss. The earlier probe uses a native control-module adapter;
the new fault probes run the actual VLA main loop and confirm policy-stall hold
and discard of a fresh late result during reset. That late chunk is discarded
by the paused queue drain; it does not isolate the active epoch comparison.
The headless simulator needs a prepared standing pose and released virtual
support. All controller trials use DDS loopback, with no physical robot.

Signed 15-degree turns pass in some states and interrupt in others. The latest
four controls passed both wrapped turns and interrupted both turns from a
zero-heading setup on the lead/rate guard. Earlier wrapped failures are retained.
Standing reset and one-second backward commands passed in all four controls.
The failed-turn probe exits before a later hold acknowledgement, so that path
still needs confirmation. Planner, decoder and contact effects need separation;
turning remains unaccepted. See the [October 6 report](artifacts/g1_harness_20261006/README.md)
for the limits, raw traces and adapter/main-loop distinction.

The corrected Genon criterion passes ten recorded judgments: eight negatives
without false completion and one two-frame release confirmation. Three saved
live negative frames also pass; a fresh live mission produces 44 in-progress
decisions and deadline interruption. These small sets do not establish general
monitor reliability. Fully hidden carried-bottle footage and successful live
placement remain pending. The original false-success frames/results are saved.

The [updated sample scene](artifacts/g1_harness_20261006/scene_layout_bottle_measured.json)
preserves native actuator/state/camera mapping and passes passive contacts and
camera transport. The source table and green stool are both 80 cm high. Bottle
dimensions and mass now match the user measurements: 20 cm tall, 8 cm diameter,
300 g. Full geometry, camera, appearance, friction and mass distribution remain
provisional. The checkpoint still never grasped it. An independent simulation
check now requires stable released top support for 0.5 seconds in both wall
and physics time, fresh samples and no robot contact. Forced-object controls
pass the intended stool and reject the source; they do not demonstrate a grasp.

Native observation preparation now retains independent left-finger measurements
as the training exporter does. All eight state groups match six recorded samples
exactly. Thirty checkpoint queries passed disconnected replay checks; the replay
uses a training episode and cannot establish generalization. Corrected live
trials still failed placement, including a 120-second measured-bottle trial with
scripted simulator-truth monitoring. That monitor is a diagnostic control;
Genon reliability is evaluated separately. All three fresh bottle runs end in
confirmed planner hold. The native component suite passes 627 tests, the normal
VoLoAgent suite passes 595 with 15 skips, and the required G1 integration command
passes 50 with no skips. These results do not establish hardware acceptance.

The [training-review queue](artifacts/g1_harness_20261002/training-review/summary.json)
maps all 179 source episodes to 174 retained episodes. Failed episode 5 remains
in the corpus; 172 retained episodes have no outcome review yet. The new
[numeric audit](artifacts/g1_harness_20261006/training-state-audit/summary.json)
checks all 274,241 retained frames without nonfinite/token-bound violations;
12 episodes contain stationary review candidates. Review outcomes and candidate
segments before another training run. Register new manipulation
prompts only after collecting demonstrations and evaluating the resulting policy.
Physical bottle placement and supervised hardware acceptance remain pending.
