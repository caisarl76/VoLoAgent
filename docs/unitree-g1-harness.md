# Unitree G1 agent harness

Software checks and recorded-checkpoint serving pass. SONIC loopback simulation
passes standing reset, bounded walking and tested interruption paths. Physical
turn repeatability, full visual acceptance and supervised hardware remain open.
See the [validation report](artifacts/g1_harness_20261001/README.md) for exact
results and remaining simulation, visual-accuracy, and hardware gates.

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

The shared profile is `configs/g1/workstation.yaml`. Its SHA-256 covers the exact
file bytes. Pass the same absolute path to both processes. Status reports
`checkpoint_expected`; it cannot attest which model an external server loaded.
Verify the serving log and run a separate non-actuating inference check.

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

The monitor requires two distinct fresh frames showing the bottle released on
the right table. Completion pauses VLA, invalidates its action epoch, then resets
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

The corrected SONIC v1.1/MuJoCo probe passes measured standing reset, one-second
backward walking, cancellation, lease expiry, stale body feedback and hand
stability on planner-input loss. Latest positive/negative turns interrupt after
10 seconds at 3.314 / 3.761-degree error, outside the 3-degree tolerance.
The headless simulator needs a prepared standing pose and released virtual
support. The probe runs native control modules through its own adapter and
records measurements/asset hashes; it has no VLA worker or camera.
See the validation report for trial values and the limits of these checks.

Genon evaluated ten bottle frames: eight negatives without false completion
and one two-frame release confirmation. The user confirmed the destination and
unrecovered failed episode; frame annotations come from session inspection.
Fully hidden carried-bottle footage and live-camera acceptance remain pending.
The failed episode is retained in the prepared training corpus, warranting a
data audit before further training. The default SONIC scene contains no task
objects, and the legacy bottle scene needs adaptation.
Physical bottle placement and supervised hardware acceptance remain pending.
