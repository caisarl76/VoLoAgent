# Unitree G1 harness implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Let an agent command the workstation's G1 to place the bottle, confirm completion, and reset to standing, then add bounded walking and turning.

**Architecture:** VoLoAgent runs a G1 coordinator and completion monitor. A local RPC interface in the existing G1 inference process accepts validated skill requests; that process keeps exclusive responsibility for publishing native VLA and SONIC commands. The two environments share a profile and wire fixtures without importing each other's packages.

**Tech Stack:** Python 3.10+, pyzmq, NumPy, PyYAML, existing VoLoAgent VLM/image helpers, native GR00T PolicyClient, and SONIC PLANNER control.

**Spec:** [Unitree G1 harness design](../specs/2026-09-30-unitree-g1-harness-design.md). Read both documents before implementation.

## Global constraints

- Preserve the dirty GR00T-WholeBodyControl workspace and implement in isolated checkouts. Verify the chosen native base includes the inspected reset/manual-planner behavior.
- Python 3.10+; existing dependencies; separate VoLoAgent and GR00T environments; no cross-repository package imports.
- One robot-control publisher in the existing G1 inference loop; RPC and monitoring never publish actuator messages.
- Profile version 1; exact prompt `pick drink bottle and place it on the right table`; native `unitree_g1_sonic`; horizon 40; publication 50 Hz; arrays `[1,40,64]`, `[1,40,7]`, `[1,40,7]`.
- Default RPC `ipc:///tmp/volo-g1-harness.sock`; local IPC only; no motor/deploy startup through agent RPC.
- Use all numeric limits in the design's proposed-limits table. They require simulation and supervised hardware validation.
- Pause/invalidate before reset; confirmation uses fresh telemetry and measured settling. Preserve hands on failure/cancel/lease expiry.
- Agent selects registered skill IDs. Reject free-form policy prompts and unsupported skills. Floor-home navigation and generic G1 grasp/place remain later work.
- Run non-actuating tests first. Hardware control requires the deployment workflow's explicit readiness confirmation.

## Review focus

1. A late policy result after reset/operator takeover must never regain control. Pin in task 2.
2. Repeated camera frames or a late VLM reply must never complete a different/current mission accidentally. Pin in tasks 3 and 4.
3. Duplicate RPC retries, rebooted executors, or a stalled VLM must not restart motion or keep a dead owner alive. Pin in tasks 3 and 5.
4. Failure while holding an object must preserve hand targets; stale telemetry must not create a false reset/hold acknowledgement. Pin in task 2.
5. A heading near +/- pi or a lost coordinator must not produce an unbounded walk/turn. Pin in task 7.

## Repositories and file map

`V` means the VoLoAgent checkout; `G` means the GR00T-WholeBodyControl checkout. Commands below run in the stated checkout. During execution, create worktrees with the worktree skill, record their absolute paths and base commits, and substitute those paths in integration commands. Do not stash/reset the existing native workspace or copy all of its edits indiscriminately.

| Files | Responsibility |
| --- | --- |
| `V/configs/g1/workstation.yaml` | Checkpoint, exact skill prompt, endpoints, limits |
| `V/docs/g1-harness-rpc-v1.md`; `V/tests/fixtures/g1_rpc_v1.json` | Wire contract and golden valid/invalid requests and responses |
| `V/vlm_orchestrator/harness/__init__.py`; `V/vlm_orchestrator/harness/g1/__init__.py` | Package markers |
| `V/vlm_orchestrator/harness/g1/registry.py`, `contract.py` | Profile/skill validation and typed RPC data |
| `V/vlm_orchestrator/harness/g1/client.py`, `monitor.py`, `runner.py`, `cli.py` | RPC client, completion adapter, mission lifecycle, agent-facing CLI |
| `V/pyproject.toml` | Add `volo-g1-harness` entry point |
| `V/tests/test_g1_registry_contract.py`, `test_g1_client.py`, `test_g1_monitor.py`, `test_g1_runner.py`, `test_g1_cli.py`, `test_g1_harness_e2e.py` | Tests owned by the matching tasks |
| `V/docs/unitree-g1-harness.md`; `V/configs/g1/bottle_then_reposition.json` | Runbook and validated sequence example |
| `G/gear_sonic/utils/inference/harness_control.py` | Lifecycle, action validity, ownership, watchdogs, status |
| `G/gear_sonic/utils/inference/harness_rpc.py`, `observation_snapshot.py` | IPC worker, wire validation, background camera snapshots |
| `G/gear_sonic/utils/inference/bounded_planner.py` | Deadline-bound translation and measured turns |
| `G/gear_sonic/utils/inference/standing_reset.py` | Read-only reset target and measured completion predicate |
| `G/gear_sonic/scripts/run_vla_inference.py`, `launch_inference.py` | Opt-in harness configuration and main-loop hooks |
| `G/gear_sonic/tests/test_harness_control.py`, `test_harness_rpc.py`, `test_observation_snapshot.py`, `test_bounded_planner.py` | New native tests |
| `G/gear_sonic/tests/test_run_vla_reset.py`, `test_launch_inference.py`, `test_vla_standing_reset.py` | Integration and existing-behavior regression coverage |
| `G/gear_sonic/tests/fixtures/g1_rpc_v1.json` | Exact copy of V's wire fixture; parity enforced by end-to-end tests |

Use `V/.venv/bin/python` for VoLoAgent tests after an editable dev install. Use the native inference environment configured for the chosen G checkout for native tests; the current workstation environment is `/home/jihun/work/GR00T-WholeBodyControl/.venv_inference/bin/python`. Run native tests with `PYTHONPATH=.` from G so an existing editable installation cannot silently load the wrong checkout. Install dependencies only during execution, following each repository's setup instructions.

## Task 1: Freeze the profile and wire contract

Files: create V's profile, contract document, golden fixture, package markers, `registry.py`, `contract.py`, and `test_g1_registry_contract.py`. Copy the fixture to G when task 3 begins.

Interfaces produced:

- `load_profile(path: Path) -> G1Profile` and `G1Profile.require_skill(skill_id: str) -> SkillDefinition` in `registry.py`.
- `validate_skill_call(profile: G1Profile, call: SkillCall, *, locomotion_enabled: bool = False) -> None` in `registry.py`. Bottle calls require empty params; standalone standing reset requires `open_hands: bool`; task 7 enables the design's walk/turn params. Built-in procedural skills have no VLA prompt and are validated separately from `profile.skills`.
- `G1Profile`: `registry_sha256`, `policy_host`, `policy_port`, `checkpoint`, `embodiment`, `camera_key`, `action_horizon`, `publish_rate`, `limits`, and `skills`. `limits` is a typed `HarnessLimits` with the design's settings; `SkillDefinition` contains `skill_id`, `prompt`, and `completion_criteria`.
- `SkillCall(skill_id: str, params: dict[str, object])`; `Request`/`Response` and `Status`/`Lease`/`Execution`/`ObservationSnapshot` dataclasses with the design's wire fields in `contract.py`. Phase is one of `IDLE`, `MANIPULATING`, `PAUSED`, `RESETTING`, `WALKING`, `TURNING`, `COMPLETED`, `INTERRUPTED`, `FAULT`.
- `decode_request(payload: bytes) -> Request`, `decode_response(payload: bytes) -> Response`, `encode_request(request: Request) -> bytes`, and `encode_response(response: Response) -> bytes`. JSON serialization rejects NaN/Infinity; schema validation rejects extra fields.

- [x] Write `test_profile_preserves_trained_prompt`, `test_registry_digest_matches_raw_bytes`, `test_profile_rejects_unsafe_limits`, and fixture-driven `test_rpc_schema_rejects_unknown_or_nonfinite_values`. Assert port 15558, horizon 40, exact prompt bytes, and rejection of `prompt`, unknown skill IDs, incorrect versions, NaN, and walk duration 5.01 s.
- [x] Run `V/.venv/bin/python -m pytest tests/test_g1_registry_contract.py -q` from V; confirm failures identify the missing implementation.
- [x] Implement `load_profile` and `require_skill` with strict YAML validation, raw-file SHA-256, and the single bottle skill. Define required/optional fields explicitly; do not import any native model code.
- [x] Implement the dataclasses and wire codecs; document every method, error code, retry rule, snapshot field, and status meaning. Include valid status, observe, claim, manipulation, pause, reset, and rejected requests in the fixture.
- [x] Re-run the task's tests; require all passing. Commit only this task's files with `feat: define G1 harness profile and RPC contract`.

## Task 2: Give the native loop explicit lifecycle hooks

Files: create G's `harness_control.py` and `test_harness_control.py`; modify `run_vla_inference.py`, `standing_reset.py`, `test_run_vla_reset.py`, and `test_vla_standing_reset.py`.

Interfaces consumed: the design's profile and RPC semantics. Implement `load_harness_profile(path: Path) -> dict` in `harness_control.py`; consume V's shared file, validate the same values, and compute the same digest without importing V.

Interfaces produced:

- `RuntimeFacts(controller_running: bool, policy_enabled: bool, policy_ready: bool, inference_busy: bool, mode_requested: str | None)` and `RuntimeHooks` protocol in `harness_control.py`: `runtime_facts() -> RuntimeFacts`, `invalidate_policy_actions() -> int`, `set_policy_prompt(prompt: str) -> None`, `set_policy_enabled(enabled: bool) -> None`, `request_planner_hold(feedback: dict, open_hands: bool) -> None`, and `begin_standing_reset(feedback: dict, open_hands: bool) -> StandingReset`. Implement adapters around the existing loop's helpers; calls occur only on the loop thread. The reset hook returns the active reset instance or raises on rejected initialization.
- `HarnessControl(profile_path: Path, hooks: RuntimeHooks, runtime_id: str)` with `dispatch(request: dict, now: float) -> dict`, `tick(now: float, feedback: dict | None, feedback_received_at: float | None) -> None`, `operator_override(reason: str) -> None`, `accept_policy_result(epoch: int, captured_at: float, action: dict[str, np.ndarray], now: float) -> bool`, and `status(now: float) -> dict`. `action` is the native processed dictionary with `motion_token`, `left_hand_joints`, and `right_hand_joints`; no flattening to an 8D action vector.
- `StandingReset.target -> np.ndarray` returns a copy; `StandingReset.is_settled(state: dict, joint_tolerance: float, yaw_tolerance_rad: float) -> bool` checks measured 31-joint targets and heading. The controller owns the fresh-feedback dwell timer and reset deadline.

- [x] Write `test_pause_rejects_late_epoch`, `test_expired_chunk_never_replays_last_frame`, `test_keyboard_override_revokes_lease`, `test_fault_hold_preserves_hands`, and `test_reset_requires_new_planner_feedback_and_measured_dwell`. Assert an old result publishes zero actions after pause; chunk age 0.801 s fails; a requested mode alone cannot complete reset; stale/unchanged telemetry index resets the 0.5 s settling dwell.
- [x] Run the new/modified targeted test modules from G: `PYTHONPATH=. <native-python> -m pytest gear_sonic/tests/test_harness_control.py gear_sonic/tests/test_run_vla_reset.py gear_sonic/tests/test_vla_standing_reset.py -q`; require the new assertions to fail first.
- [x] Implement the reset target/predicate and lifecycle controller. Validate exact native action shape/finite values and token bound 1.25 before acceptance; require a fresh advancing telemetry index for planner acknowledgement. Preserve the existing ramp and heading conversion.
- [x] Integrate hooks into the existing runner. Carry observation-capture time with the existing inference epoch into worker results. Validate camera/feedback freshness before every inference; add `test_stale_camera_blocks_policy_request` so a cached camera message cannot receive a newer capture time. Harness mode expires/exhausts cached actions instead of clamping to the last action forever, pauses on starvation, and cannot auto-resume. Retain ordinary keyboard operation when harness mode is disabled.
- [x] Add asynchronous non-actuating prewarm before accepting a manipulation start, using a valid recorded/live observation while policy output is paused. Record `policy_ready` in runtime facts; a 6.27 s warmup must not block the action loop/RPC or produce a publishable chunk. Active results must still satisfy the 0.8 s capture-age limit. Reject a busy/stuck worker before motion starts.
- [x] Add main-loop tests for pause/reset ordering, cancellation with invalid telemetry, prompt changes, and recording keys. Require hold-confirmed false when feedback/planner acknowledgement is unavailable; verify no hand-open command is emitted on failure. Recording-only `c`, `s`, and `f` must not revoke a lease when they do not change native control state.
- [x] Run the targeted suite above plus `test_manual_planner.py` and `test_planner_heading_frame.py`; require all passing. Commit the task files with `feat: add G1 harness lifecycle and stale-action guards`.

## Task 3: Expose local RPC, ownership, and fresh observations

Files: create G's `harness_rpc.py`, `observation_snapshot.py`, their tests, and native copy of `g1_rpc_v1.json`; modify `run_vla_inference.py` for opt-in configuration.

Interfaces consumed: `HarnessControl.dispatch/tick/status` and task 1's documented wire schema/fixtures.

Interfaces produced:

- `HarnessRPCServer(endpoint: str, snapshots: ObservationSnapshotCache)` with `start() -> None`, `drain(control: HarnessControl, now: float, max_requests: int = 1) -> None`, and `close() -> None`. A worker owns REP and queues validated requests; `drain` runs on the action loop and completes queued replies without network/image work.
- `ObservationSnapshotCache(camera_factory: Callable[[], object], camera_key: str)` with `start() -> None`, `latest(now: float) -> dict | None`, and `close() -> None`. The worker owns its camera subscriber and JPEG encoding. Cache encoded immutable snapshots.
- Native `InferenceConfig` adds `harness_endpoint: str | None = None` and `harness_profile: str | None = None`. Require both together; endpoint None keeps the current runner behavior. Expose a new runtime ID on every process start.

- [x] Write fixture-parity `test_native_rpc_contract`, `test_duplicate_start_is_idempotent`, `test_request_id_payload_conflict`, `test_old_runtime_id_rejected`, `test_only_one_owner`, and `test_lease_expires_without_motion_replay`. Assert duplicate heartbeats do not extend expiry; expired ownership interrupts at 2.0 s and cannot auto-claim.
- [x] Write `test_republished_camera_timestamp_ages`, `test_missing_ego_view_is_unavailable`, `test_socket_owned_by_one_thread`, and `test_second_server_does_not_unlink_live_socket`. A repeated timestamp must keep its frame ID/receipt age; a zero ROS timestamp must not invalidate advancing telemetry indices.
- [x] Run `PYTHONPATH=. <native-python> -m pytest gear_sonic/tests/test_harness_rpc.py gear_sonic/tests/test_observation_snapshot.py -q` from G; confirm the new tests fail.
- [x] Implement bounded request queuing and prompt replies. Claim requires operator-started controller, paused policy, matching profile digest, and no live owner. Lease/replay logic lives in `HarnessControl`; the RPC worker never mutates robot state. Reject control requests waiting past their lease/deadline.
- [x] Implement snapshot freshness from `timestamps["ego_view"]`. Drop missing/invalid timestamps; treat reconnect timestamp regression as a new camera stream generation, clear old snapshots, and require new advancing frames. Never reuse cached image data under a new frame ID without a new source timestamp.
- [x] Enforce IPC permissions, clean shutdown, finite RPC waits, and per-request error responses. Verify a large image reply or malformed request cannot block the action loop; keep serialization in the worker.
- [x] Run the task's tests and task 2's suite; require all passing. Commit with `feat: expose leased local RPC for the G1 executor`.

## Task 4: Add a G1 completion monitor

Files: create V's `monitor.py` and `test_g1_monitor.py`; reuse `failure_handlers/vlm.py`, `failure_handlers/base.py`, `strategies/base.py`, and `vlm/api.py` without changing legacy strategy behavior.

Interfaces consumed: `SkillDefinition`, `ObservationSnapshot`, existing `VLMFailureHandler`, and `HandlerResult`. Adapter state uses the existing `SessionState` fields expected by the handler.

Interfaces produced:

- `CompletionDecision(outcome: Literal["in_progress", "complete", "failure", "unavailable"], reason: str, frame_id: str, execution_id: str, inference_epoch: int, captured_at: float, decided_at: float)` in `monitor.py`.
- `G1CompletionMonitor(skill: SkillDefinition, vlm_call_fn: Callable, now_fn: Callable[[], float])` with `begin(execution: Execution, initial: ObservationSnapshot) -> None` and `check(snapshot: ObservationSnapshot) -> CompletionDecision`.
- The adapter schedules one check per second, uses the existing image encoding/chat helpers, captures attempted-check failures even if the handler returns None, and applies the design's two-frame confirmation rule.

- [x] Write `test_two_distinct_complete_frames_required`, `test_complete_next_pair_required`, `test_stale_or_previous_epoch_cannot_complete`, `test_parse_error_or_api_error_clears_streak`, `test_late_decision_discarded`, and `test_replan_instruction_never_reaches_policy`. Assert repeated frame IDs cannot build a streak; default confidence 1.0 cannot bypass validation; a decision after 10 s from capture is unavailable.
- [x] Run `V/.venv/bin/python -m pytest tests/test_g1_monitor.py -q` from V; confirm the new tests fail.
- [x] Implement the adapter with explicit `ego_view`, initial/current images, the exact task, and the registered completion criterion. Keep response parsing and freshness enforcement separate. Failure/replan requests become an interruption, and never a new VLA prompt.
- [x] Implement an injectable VLM call with a 10 s deadline and cancellation/late-result discard. Include success, held-bottle, wrong-table, occluded, malformed, and API-exception scripted responses in tests; these test logic, and are not a claim of visual accuracy.
- [x] Run the new monitor tests plus `tests/test_vlm_failure_handler.py`; require all passing. Commit with `feat: monitor G1 bottle placement with fresh-frame confirmation`.

## Task 5: Connect the coordinator and agent-facing CLI

Files: create V's `client.py`, `runner.py`, `cli.py`, `test_g1_client.py`, `test_g1_runner.py`, and `test_g1_cli.py`; modify `pyproject.toml`.

Interfaces consumed: task 1's profile/contract types, task 3's RPC, and task 4's `G1CompletionMonitor`.

Interfaces produced:

- `G1Client(endpoint: str, timeout_s: float = 0.5)` with `request(method: str, params: dict[str, object], *, session_id: str | None = None, lease_id: str | None = None, request_id: str | None = None) -> Response`, `get_status() -> Status`, `observe() -> ObservationSnapshot`, and `close() -> None`. Track runtime ID from initial status; callers supply the same request ID when retrying.
- `MissionResult(execution_id: str, outcome: Literal["completed", "interrupted", "fault"], reason: str, evidence_dir: Path)` and `HarnessRunner(profile: G1Profile, client_factory: Callable[[], G1Client], monitor: G1CompletionMonitor, evidence_dir: Path)` with `run(call: SkillCall) -> MissionResult`.
- `main() -> None` in `cli.py`, registered as `volo-g1-harness`. Commands: `skills`, `status`, `observe --output PATH`, and `run --skill bottle_to_right_table`; common `--profile` and `--endpoint`. Manipulation accepts `--vlm-model`, `--vlm-base-url`, and `--vlm-api-key`, using `VLM_API_KEY` then `OPENAI_API_KEY` as fallbacks. Require explicit non-placeholder model/endpoint values for manipulation; read-only commands need no VLM configuration. Emit structured JSON results and nonzero exit status for interrupted/fault outcomes.

- [x] Write `test_req_socket_recreated_after_timeout`, `test_runtime_reboot_interrupts_mission`, `test_vlm_delay_does_not_delay_heartbeat`, `test_success_pauses_then_resets`, `test_failure_preserves_hands_and_does_not_reset`, and `test_terminal_mission_does_not_recycle`. Assert the ordered calls are start, pause acknowledgement, reset, settled completion, release; no second start follows completion.
- [x] Write `test_cli_rejects_free_prompt_and_unknown_skill`, `test_status_and_observe_do_not_claim`, and `test_runtime_profile_mismatch_aborts`. Assert read-only commands never start policy/control and wrong digests cannot start a mission.
- [x] Run `V/.venv/bin/python -m pytest tests/test_g1_client.py tests/test_g1_runner.py tests/test_g1_cli.py -q` from V; confirm the new tests fail.
- [x] Implement RPC retries with stable request IDs and a dedicated heartbeat client/socket owned by its thread. VLM calls run separately, and heartbeat failure cancels the mission even if VLM is still running. Handle Ctrl-C and process teardown through cancel/hold; lease expiry covers abrupt process death.
- [x] Implement the mission lifecycle and completion/reset polling, using execution/epoch checks on every decision. Choose the registered prompt only in the executor. Respect 120 s task and 15 s reset deadlines; never interpret an RPC acceptance as finished motion.
- [x] Implement CLI and per-mission JSONL/images. Log source freshness, exact prompt, policy/monitor latency, mode evidence, rejected actions, outcomes, and profile digest. Block automatic retry after failure; require a new explicit run.
- [x] Run the task suite and task 1/4 tests; require all passing. Commit with `feat: run supervised G1 missions through VoLoAgent`.

## Task 6: Verify the complete first milestone and document launch

Files: create V's `test_g1_harness_e2e.py` and `docs/unitree-g1-harness.md`; modify G's `launch_inference.py` and `test_launch_inference.py`.

Interfaces consumed: tasks 1-5, the existing native launcher, and its current reset tests.

Interfaces produced: launcher forwards `--harness-endpoint` and an absolute `--harness-profile` to the native inference runner, preserving current policy/camera/state/action endpoints and defaults. The runbook records checkout commits, profile digest, policy identity evidence, and explicit expected readiness states.

- [x] Write an end-to-end subprocess test using a fake native policy, fake camera/telemetry, and scripted monitor. Launch the actual native harness RPC/control code from an explicitly supplied `G1_NATIVE_CHECKOUT` and `G1_NATIVE_PYTHON`; the test's publisher is an in-memory recording sink. Importing modules is allowed; connecting to real action port 5556 is forbidden in this fixture.
- [x] Add assertions for identical wire fixtures, exact prompt, a single publisher, fresh two-frame completion, pause/invalidation, planner acknowledgement, measured settling, and exactly one terminal completion. Add delayed old policy results, frozen camera timestamp, failed reset acknowledgement, and coordinator termination variants. Missing native paths skip with a clear reason in ordinary VoLoAgent tests; the dedicated milestone check requires them and treats skips as failure.
- [x] Run `G1_NATIVE_CHECKOUT=<G-worktree> G1_NATIVE_PYTHON=<native-python> V/.venv/bin/python -m pytest tests/test_g1_harness_e2e.py -q` from V; confirm failures precede implementation of the fixture/launcher forwarding.
- [x] Add launcher configuration/forwarding tests, including paths containing spaces and disabled harness mode. Preserve the native prompt and keyboard control behavior; derive supported launcher flags from the chosen checkout rather than an older runbook.
- [x] Write the runbook with observe-only preparation, recorded serving check, paused-worker prewarm, expected paused/started runtime, claim, supervised mission, cancellation, and evidence review. Policy is `127.0.0.1:15558`; camera/state/action defaults are 5555/5557/5556. Verify deploy consumes the selected native runner's commands and no competing XR/teleop sender controls the same input. Keep SONIC deploy and robot startup in the existing deployment workflow.
- [x] Run the end-to-end suite with explicit native paths and zero skips, native lifecycle/RPC/reset/launcher suites, and VoLoAgent's targeted G1/VLM suites. Run Ruff on new/modified Python files. Record commands and outcomes in the runbook; commit with `test: verify G1 bottle mission and document launch`.

Milestone 1 is ready for simulation when tasks 1-6 pass. It is ready for supervised hardware evaluation only after the relevant task 8 checks pass.

## Task 7: Add bounded locomotion and skill sequences

Files: create G's `bounded_planner.py` and `test_bounded_planner.py`; extend G's `harness_control.py`/tests. Extend V's registry, contract fixtures, runner, CLI/tests; create `configs/g1/bottle_then_reposition.json`.

Interfaces consumed: the same lease, control hooks, freshness, heading-frame conversion, and status contract as milestone 1. Walking/turning may start only from a confirmed stationary PLANNER state with no policy actions active.

Interfaces produced:

- `BoundedPlanner(start_feedback: dict, started_at: float)` with `begin_walk(direction: str, duration_s: float, speed_mps: float) -> None`, `begin_turn(angle_rad: float, rate_rps: float) -> None`, `advance(feedback: dict | None, now: float) -> PlannerCommand`, and `finished -> bool`. Import `PlannerCommand` from `gear_sonic.utils.teleop.bridge_planner_publisher`; preserve measured upper-body and hand positions. Extend `RuntimeHooks` with `set_planner_command(command: PlannerCommand) -> None`; the existing loop publishes that command after normal reference-frame conversion. The controller retains post-stop acknowledgement and lease checks.
- `HarnessRunner.run_sequence(calls: list[SkillCall]) -> MissionResult`; CLI `execute --plan-file PATH`. Validate the complete sequence before acquiring control; stop on the first interrupted/fault result. The heartbeat lease spans the sequence.
- Sequence JSON has `version: 1` and `skills: [{"skill_id": str, "params": object}]`. Advertised IDs add `walk_for` and `turn_by`; standing reset is allowed standalone. Manipulation still uses only registered exact prompts.

- [x] Write `test_walk_stops_at_deadline`, `test_turn_wraps_pi_and_uses_measured_yaw`, `test_stale_feedback_and_lease_loss_clear_motion`, `test_turn_rate_and_lead_are_bounded`, and `test_locomotion_preserves_hands`. Assert duration 5.01 s, speed 0.201 m/s, and angle >45 degrees fail; expiry yields zero movement; a 179-degree to -179-degree turn uses a 2-degree wrapped error.
- [x] Write `test_sequence_validated_before_claim`, `test_sequence_stops_on_first_failure`, and `test_locomotion_cannot_overlap_manipulation`. A bad later instruction must prevent the earlier bottle skill from starting; cancellation must not open the hands.
- [x] Run the new native planner and V runner/registry tests; confirm the new assertions fail.
- [x] Implement deadline-bound walking and rate-limited yaw targets using the existing planner frame converter. Follow the design's 0.2 m/s, 5 s, 45 degrees, 10 degrees/s, 5-degree lead, 3-degree/0.5 s settling, and 10 s turn deadline. Confirmation requires post-stop advancing planner-active feedback; report walk completion as duration completion, without claiming measured distance.
- [x] Implement sequence validation/dispatch and the example: bottle task (including its automatic standing reset), backward walk 1 s at 0.2 m/s, and turn 15 degrees at 10 degrees/s. Do not append a redundant standing reset between bottle completion and walking.
- [x] Extend golden fixtures on both sides and non-actuating end-to-end tests for sequencing, operator override, and coordinator death mid-walk. Run all milestone 1 tests plus task 7 tests; require no regression. Commit in the owning repositories with `feat: add bounded G1 walking and turning skills`.

## Task 8: Validate simulation and supervised hardware behavior

Files: update V's runbook; save results under `V/docs/artifacts/g1_harness_<validation-date>/` with exact commands, checkout commits, profile digest, checkpoint identity evidence, JSONL, and evidence media. This task runs only during implementation/validation, not while preparing this plan.

- [x] Verify the recorded derived dataset exists at `/tmp/pnp_bottle_260916_gr00t_raw_20260930` and the expected server is on port 15558. Run `<native-python> /home/jihun/work/GR00T-WholeBodyControl/docs/artifacts/pnp_bottle_260916_raw_training_20260930/smoke_inference.py`. Require finite native shapes and record latency. Keep all outputs disconnected from robot control; if the temporary dataset is missing, reconstruct it with the experiment's documented preparation before this check.
- [x] Run SONIC MuJoCo transitions with a scripted policy/monitor first: pause -> planner hold -> standing settle, then cancellation, lease expiry, telemetry loss, policy stall, and old-epoch results. Verify the C++ controller's lost-input behavior and whether a hold actually applies; unresolved behavior blocks agent-controlled hardware testing. Scope: adapter-driven controller interruptions plus actual VLA-main-loop policy stall and late-result discard during reset; the active epoch comparison remains covered by synthetic cross-process checks.
- [ ] Validate the completion monitor on recorded success/failure/held-bottle/occlusion clips. Save labels, decisions, and false-complete counts. Require zero false-completes in the chosen acceptance set before supervised trials; report the sample count so this is not presented as a general reliability guarantee.
- [x] Run an actual bottle/right-table manipulation scene if a compatible simulation environment is available. If it is missing, record that gap and keep manipulation success unvalidated; the generic SONIC control simulation does not close it. Build the scene or proceed only to explicitly supervised physical evaluation with the limitation documented. Two real-checkpoint/sample-scene missions were run; neither placed the bottle. Scene calibration and successful placement remain acceptance work.
- [ ] Test milestone 2 in simulation: 1 s backward walk, 15-degree turn, wraparound heading, fresh/stale telemetry, and killing the coordinator. Require zero continued movement requests after lease/deadline and measured turn/standing tolerances within the design limits.
- [ ] Use the deploy skill for the real G1. Obtain its explicit robot-readiness confirmation before actuation, then perform observe-only, supervised pause/reset, bottle placement/reset, and locomotion checks in that order. Record each result and intervention; fail the milestone if an old action reappears, a reset falsely completes, or an interrupted skill resumes automatically.
- [x] Report separate outcomes for contract/lifecycle tests, real-checkpoint serving, simulation control, monitor accuracy, physical placement, standing reset, and locomotion. Merge only validated code; document any hardware feature still disabled.

Validation on 2026-10-02 is recorded in
`docs/artifacts/g1_harness_20261002/README.md`; the earlier report retains the
full-suite/controller results. Fifty current G1 tests pass with no skips,
including required integration cases. Twenty-three native harness/launcher
tests pass after declaring PyYAML, and the workstation inference imports pass
after installing that missing dependency. Actual VLA-loop policy stall and
fresh late-result discard during reset now pass against SONIC/MuJoCo. Existing
reset, one-second walking, cancellation, lease expiry, stale body feedback and
lost-input hand-stability checks remain valid. Signed turns pass in some states,
but negative near/wrapped trials still fail at unchanged limits; repeatability
remains open.

An editable compatible sample scene now runs the real checkpoint, live camera,
SONIC and Genon together. The first trial falsely reported completion while
the bottle stayed on the source. Naming the separate green stool corrected
that observed case: three saved negatives and all ten recorded judgments pass,
and a fresh live trial reports 44 in-progress decisions before its 120-second
deadline and confirmed hold. Neither live trial placed the bottle. The
prototype contact check is labeled contact-only and cannot certify placement.

Scene calibration, successful grasp/placement, full carried-bottle occlusion,
turn repeatability and supervised hardware remain open. The training-review
queue maps 179 source/174 retained episodes; failed episode 5 remains retained,
and 172 retained outcomes remain unreviewed. No dataset or checkpoint changed.
Historical vision disagreements, false success and failed driver attempts are
preserved. Task 8 remains open despite completing its scene-run and interruption
activities. Dashboard implementation remains deferred below.

Validation on 2026-10-06 is recorded in
`docs/artifacts/g1_harness_20261006/README.md`. Native inference now preserves
the independently measured left fingers as training does. Six recorded samples
match all eight state groups; the regression also prevents cached-state mutation.
Thirty disconnected checkpoint queries pass, with replay explicitly limited to
training-fit evidence. Native component tests pass 627; normal VoLoAgent tests
pass 595 with 15 skips; the dedicated G1 check passes 50 with zero skips.

The sample bottle now matches the user's 20 cm height, 8 cm diameter and 300 g
mass. Forced-object controls pass stable stool support and reject the source.
The placement checker requires dwell in both wall and physics time and cannot
retain a completed outcome after evidence failure. Fresh Genon and scripted
truth-control runs still never grasp the bottle; all three confirm planner hold
on interruption. A measured-bottle trial reaches its 120-second deadline without
placement. Full scene/camera/appearance calibration and live success remain open.

Four more adapter-driven SONIC trials pass standing reset and backward-duration
checks. Both wrapped turns pass; both zero-heading turns interrupt on the
lead/rate guard. Limits are unchanged and repeatability is unaccepted. Those
failed-turn probes exit before a fresh hold acknowledgement; that path and
native-main-loop locomotion remain acceptance work. The full retained-frame
numeric audit is done; its stationary flags are review candidates, and 172
retained outcomes still lack visual review. No data, checkpoint or prompt pool
changed. These activities do not close task 8 or authorize physical deployment.

Follow-up validation on 2026-10-06 is recorded in
`docs/artifacts/g1_harness_20261006_followup/README.md`. Four actual-native-loop
trials pass measured standing reset and backward-duration checks. Two turns
complete; two reach their deadline and confirm fresh hold within 0.1 seconds,
with stationary requests and no resumption over the following second. Successful
cases also pass coordinator loss. Planner, decoded target and measured yaw are
plotted separately. Repeatability and historical lead/rate-guard stop coverage
remain open; limits are unchanged.

A deterministic regression reproduced fresh-query action-chunk exhaustion.
Harness query cadence now starts from capture time; late-result rejection and
legacy cadence remain intact. A real-checkpoint clear-cylinder trial at the new
cadence lasts 120 seconds, fails to grasp and confirms deadline hold. Appearance
controls preserve physics, but full calibration and successful placement remain
open. Native tests pass 629, VoLoAgent 595 with 15 skips, explicit required
cross-process tests nine with zero skips, and probe helper tests 12.

All 12 stationary-flagged recordings now have sampled visual screening, without
new accepted outcome labels. Source 155 and 159 expose incompatible sampled
content under the blanket prepared bottle prompt. Task boundaries, task labels
and 172 complete outcomes need review before a new derived training set. No
dataset, checkpoint or prompt pool changed. Task 8 remains open, and these
simulation checks do not authorize physical deployment.

Acceptance investigation on 2026-10-06 is recorded in
`docs/artifacts/g1_harness_20261006_acceptance/README.md`. Dense sampled review
inspects 316 frames on 23 sheets from six episodes; it creates no new accepted
episode labels. Source 52 remains visibly held in terminal samples. Source 155
mixes cup/box/other-object tasks with a bottle portion and needs segmentation.
All 174 retained episodes receive a control-mode audit and future split plan;
zero accepted training windows keep the standalone metadata checker blocked.
Source 5/18 are reserved only for a future checkpoint, since the current one
already trained on them. No dataset or checkpoint changes.

Eight additional Genon frame judgments match provisional labels, with zero
false completions on six negatives and one two-frame release confirmation. An
incorrect manifest expectation for the first positive outcome is corrected
without changing the monitor or rerunning API calls. Fully hidden held-bottle
coverage remains open. Offline reconstruction finds a roughly 20 cm minimum
sampled right-hand/bottle gap in the earlier clear-cylinder trial.

The user confirms D435i with the official G1 head mount. A frozen-pose URDF mount
proxy preserves physical parameters and state inputs, but RGB intrinsics and
optical-to-robot transform remain unmeasured. A read-only SDK helper is supplied;
the checked local native environment lacks pyrealsense2. Two fresh native-loop
turns fail under unchanged limits, with and without a one-second stationary
request first. Both receive fresh hold acknowledgement and no resumption over
one second. The guard case closes one missing lead/rate interruption-and-hold
check, while repeatable turning remains unaccepted. Fresh focused checks pass
27 with no skips. No production changes or physical actuation; Task 8 remains
open. Next: accepted segment review, measured camera calibration, controller
input/trajectory/frame instrumentation, and successful manipulation evaluation.

Review correction: the metadata checker now rejects list/object split values
without raising. Two new regression cases bring the focused checks to 29 passes;
the fresh full suite passes 595 with 15 skips outside the sandbox. The reviewed
commit receives a valid author sign-off. Original sources, review, test logs and
evidence are preserved; the correction does not change acceptance results.

## Deferred to another session: G1 monitoring dashboard

Deferred by the user on 2026-10-02. Dashboard implementation is outside the
remaining work in this session.

- [ ] Add a browser monitoring page for the G1 harness: camera, executor
  connection and telemetry freshness, active skill and mission progress,
  VLA/SONIC mode, completion-monitor decisions, faults, and saved evidence.

The existing subgoal HITL page does not consume G1 harness status. Start the
separate session from `docs/unitree-g1-harness.md` and the current CLI `status`
and `observe` interfaces. The first dashboard should monitor without acquiring
a control lease; show unavailable or stale data explicitly. Robot command
controls require their own design before implementation.

## Execution order and review gates

Tasks 1-6 deliver the first usable harness. Task 7 extends the same executor after that gate. Task 8 validates each milestone at the relevant level; hardware results cannot be replaced by unit-test results.

Keep architecture, cross-repository integration, native control-loop changes, and final review with the main agent. Bounded fixture/config/CLI work can be delegated in non-overlapping files under the workspace routing instructions; verify the selected worker model is actually available. Do not delegate repository exploration or concurrent edits to the shared controller/contract. Inspect every delegated diff and rerun the owning task's checks.

Before starting implementation, record the two clean worktree bases and explicitly account for the native behavior currently present only in dirty edits. This plan authorizes no cleanup of those edits. The present planning deliverables are the design and this task list; no robot process, dependency installation, or production control change is part of their creation.
