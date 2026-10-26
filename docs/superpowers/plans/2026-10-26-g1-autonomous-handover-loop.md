# G1 Autonomous Handover Loop Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run lerobot's validated pick-and-handover loop (recorded ready pose and paths, anchoring, one attempt per trial, hand-state handover detection) unattended on VoLoAgent's native harness. A local VLM triggers each trial (case empty, then a bottle upright with no hand at it). The operator keeps the startup gates, a freeze-first stop key, and non-blocking scoring.

**Architecture:**
- The native harness (GR00T-WholeBodyControl fork, on PC2 next to the deploy) gains a POSE-mode `PoseCommander`. It streams holds, blends, recorded paths and the policy at 50 Hz.
- A `LoopController` state machine is exposed as new `loop_*` RPC methods. Episode guards and a trial recorder use lerobot's file format.
- VoLoAgent (also on PC2: one clock) gains a `loop` command. It runs the operator gates, the stop key and a re-arm VLM monitor, and drives the native loop over RPC. A separate `verdict` console scores trials.
- A stdlib OpenAI-compatible server on workstation GPU 0 serves Qwen3-VL-2B.

**Tech Stack:** Python 3.10 (PC2 venv) and 3.13 (workstation), numpy, pyzmq, msgpack, pytest; transformers 5.5.4 (lerobot venv) for the VLM server; MuJoCo fixture for simulation.

**Spec:** `docs/superpowers/specs/2026-10-26-g1-autonomous-handover-loop-design.md` (VoLoAgent, commit 31325b2). Lerobot reference: caisarl76/lerobot commit **a6e267c6**, `examples/g1_dex3_training/`.
- Read it **only** via the pinned commit: `git -C ~/work/lerobot show a6e267c6:examples/g1_dex3_training/<file>`.
- Never use the moving `caisarl76/main` ref, and never use the working tree, which is on another branch.

**Repos and branches:**
- **Native:** `/home/jihun/work/VoLoAgent/.worktrees/g1-harness-native`. New branch `work/g1-autonomous-loop-20261026` from **`d0f29b3`** (the head of `work/g1-lerobot-policy-20261021`). It contains the LeRobot adapter (`ad20fd7`, `b50613c`) and the deploy's `--dds-domain` flag (`d0f29b3`).
  - Always `git add` explicit paths.
- **VoLoAgent:** `/home/jihun/work/VoLoAgent/.worktrees/g1-harness`. New branch `work/g1-autonomous-loop-20261026` from the corrected plan's commit (`1b33c3d` or its amended successor on `work/g1-s10-loop-sim-20261021`), which contains the spec `31325b2` and this plan.
  - `docs/artifacts/g1_handover_20261021_s10_loop_sim/` belongs to another session. Don't stage it.
- **Commits:** `git -c user.name="Jihun Kim" -c user.email="jihun.kim@genon.ai" commit -s`, ending with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- **Tests:**
  - Native: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest -q -p no:cacheprovider gear_sonic/tests/<file>`.
  - VoLoAgent: the same command with `tests/<file>`.

## Global Constraints

- **Settings (verbatim from the spec/lerobot):**
  - 50 Hz, chunk 40, replan 0.4 s;
  - anchoring 1.0 s on the 64 token dims, from the token actually sent;
  - hands cross-fade 0.3 s;
  - token slew 0.05 per tick, on the token only, in every phase;
  - hold-rest 1.5 s, hands open;
  - blend-in 1 s (50 ticks); blends to and from the ready pose 1 s (50 ticks);
  - raise and lowering 83 ticks, one row per tick;
  - trial cap 30 s;
  - HandoverDetector `open_s=1.0, min_hold_s=0.5, closed_above=0.19` on the mean of the 7 measured right Dex3 joints, timed with `time.monotonic()`.
- **Watchdogs that end a trial:**
  - arm speed > 6.0 rad/s, measured over 3 ticks (`ARM_SPEED_TICKS = 3`, wall clock);
  - arm outside `ARM_LO`/`ARM_HI` (verbatim from `sonic_policy_streamer.py:150-170`) by > 0.15 rad;
  - no valid chunk for 1.0 s.
- **Freeze conditions:**
  - robot state > 0.2 s old;
  - token echo mismatch (`TOKEN_ECHO_TOL = 1e-4`) for 0.5 s;
  - camera frame > 0.5 s old;
  - invalid policy result;
  - telemetry restart or incompatible telemetry;
  - lease expiry;
  - a stray native key.
- **Stop:** a stop freezes the robot (it streams the last sent token and hands). The planner hold is used only at the session-end hand-back.
- **Ready pose asset:** `ready_pose_261020_ep76.npz` from lerobot a6e267c6. Keys and shapes:
  - `token` (64), `hands` (14), `state` (28);
  - `raise_tokens` (83, 64), `raise_hands` (83, 14);
  - `lower_tokens` (83, 64), `lower_hands` (83, 14).
- **VLM:** Qwen3-VL-2B-Instruct bf16 on workstation GPU 0. Refuse to start if < 4.4 GB free. Two distinct frames confirm each re-arm step.
- **Contract mirrors:**
  - native `harness_contract.py` ↔ VoLoAgent `vlm_orchestrator/harness/g1/contract.py`;
  - native `harness_profile.py` ↔ VoLoAgent `registry.py`.

  They stay byte-identical except for the import line. Check with `diff <(sed 's/gear_sonic.utils.inference/X/g; s/from .harness_contract/from .contract/' NATIVE) <(sed 's/vlm_orchestrator.harness.g1/X/g' VOLO)`.
- **Records:** lerobot's `eval_session.write_trial` layout: `trial_NNN.npz` and `trial_NNN_frames.npz` per trial, written at trial end; the `results.jsonl` line is written when the verdict arrives.

## Review Focus

1. **A stop pressed during a blend, a path or the policy episode** must freeze at the exact last-sent token: no further token change until `loop_resume`. Test in Task 3 (`test_freeze_mid_blend_holds_last_sent_token`) and Task 6 (`test_stop_during_trial_freezes_and_ends_trial`).
2. **The orchestrator dying mid-loop:** the lease expires, the native side freezes (never a planner hold), and a new orchestrator claims from `LOOP_FROZEN` and resumes without take-over or a second raise. A freeze during the lowering completes the session instead. Tests: Task 6 (`test_lease_expiry_in_loop_freezes_and_allows_reclaim`, `test_resume_mid_lowering_finishes_to_planner_and_completes`) and Task 10 (`test_fresh_runner_reclaims_frozen_loop_without_takeover`, `test_reclaim_frozen_during_lowering_completes_without_restart`).
3. **A bottle still standing after a missed grasp** must not start the next trial until the case was seen empty. Test in Task 8 (`test_rearm_requires_empty_case_before_bottle`).
4. **A verdict typed while the robot runs** is never read as the stop key, and a redo adds one trial to the remaining count. Test in Task 9 (`test_redo_extends_remaining_trials`) and Task 10 (`test_stop_key_and_verdict_console_are_separate`).
5. **The deploy left without tokens in POSE mode, or a stop delayed.** Every LOOP phase publishes exactly one pose message per tick, including while waiting for the bootstrap chunk, while frozen, and while a trial's files are being written. A blocked VLM call never delays the stop RPC. Tests: Task 5 (`test_close_returns_before_slow_write`), Task 7 (`test_loop_phases_publish_every_tick`, `test_first_chunk_bootstrap_through_inference_queue`, `test_slow_trial_write_does_not_stall_ticks`) and Task 10 (`test_stop_dispatched_while_vlm_call_blocks`).

---

## Part A: Native harness (GR00T-WholeBodyControl fork)

### Task 1: Chunk anchoring in TokenStream and the profile

**Files:**
- Modify: `gear_sonic/utils/inference/lerobot_sonic_policy.py` (`TokenStream`)
- Modify: `gear_sonic/utils/inference/harness_profile.py` (`LeRobotPolicySettings`, the `lerobot_policy` validation)
- Modify: `gear_sonic/scripts/run_vla_inference.py` (pass `anchor_s` when building `TokenStream`)
- Test: `gear_sonic/tests/test_lerobot_sonic_policy.py`

**Interfaces:**
- Produces:
  - `TokenStream(source_fps: float, blend_s: float = 0.0, max_token_step: float = 0.0, anchor_s: float = 0.0)`.
  - `set_chunk(rows, t0, now)` anchors the 64 token dims when `anchor_s > 0` and a chunk was already set. The reference is `self.last_token` (the token actually sent); the 14 hand dims keep the `blend_s` cross-fade.
  - `LeRobotPolicySettings.chunk_anchor_s: float = 0.0`, validated `0 <= x <= 5.0`.

- [ ] **Step 1: Write failing tests**, porting the anchoring cases from lerobot `tests/datasets/test_g1_dex3_sonic_token_stream.py`:
  - `test_anchoring_starts_from_sent_token_and_decays`: chunk A of ones, then the token stepped to 0.5 by the slew limit, then chunk B of zeros at `now=1.0` with `anchor_s=1.0`. Assert `at(1.0)[0][:64] == pytest.approx(0.5, abs=1e-6)` and `at(2.0)[0][:64] == pytest.approx(0.5*math.exp(-1), abs=1e-5)`.
  - `test_anchoring_leaves_hands_on_crossfade`: hands go 1→0 over 0.3 s. Assert `at(now+0.15)[1] == pytest.approx(0.5)`.
  - `test_profile_chunk_anchor_s_validated`: 1.0 accepted, -0.1 and 6.0 rejected with `match="lerobot_policy"`.
- [ ] **Step 2: Run `test_lerobot_sonic_policy.py`.** Expected: 3 new FAIL.
- [ ] **Step 3: Implement.** Port `ChunkResampler.set_chunk`/`token_at` anchoring (lerobot `sonic_token_stream.py:46-122` at a6e267c6), with `anchor_dims=64` and `current=self.last_token`. Add the profile field and validation in `harness_profile.py`.
- [ ] **Step 4: Run the file.** Expected: all pass. Then run the full native suite: only the known Inspire failures remain.
- [ ] **Step 5: Mirror the profile change in VoLoAgent `registry.py`.** Add `chunk_anchor_s` cases to `tests/test_g1_registry_contract.py::test_profile_lerobot_policy_section_is_optional_and_validated`. Run that file and the mirror diff, which should show only the import line.
- [ ] **Step 6: Create the runnable loop profile `configs/g1/handover-loop.example.yaml` (VoLoAgent).** It starts from `handover.example.yaml` and sets:
  - `policy_host: 192.168.0.62`, `policy_port: 5571`;
  - `checkpoint` = the s08 path `/mnt/data/jihun/g1_models/wbt_groot_handover261015_he_full/pretrained_model`;
  - `lerobot_policy: {chunk_blend_s: 0.3, max_token_step: 0.05, dex3_right_order: dataset, chunk_anchor_s: 1.0}`;
  - the review flags stay **false**. A session copy sets `checkpoint_verified: true` (Task 14).

  Test `tests/test_g1_handover_profile.py::test_loop_example_profile_enables_anchoring`: it loads, has `chunk_anchor_s == 1.0`, and has `checkpoint_verified is False`. Run the native parser on the same file too. Commit in each repo.

### Task 2: Ready-pose asset

**Files:**
- Create: `gear_sonic/data/handover/ready_pose_261020_ep76.npz`, copied from lerobot a6e267c6.
- Create: `gear_sonic/data/handover/ready_pose_261020_ep76.json` (provenance: lerobot commit, `make_ready_pose.py` args, sha256).
- Create: `gear_sonic/utils/inference/loop_motion.py` (the asset part)
- Test: `gear_sonic/tests/test_loop_motion.py`

**Interfaces:**
- Produces:
  - `ReadyPose` frozen dataclass with fields `token (64,), hands (14,), state (28,), raise_tokens (83,64), raise_hands (83,14), lower_tokens (83,64), lower_hands (83,14)`, all float32.
  - `load_ready_pose(path: Path = READY_POSE_PATH) -> ReadyPose`. It raises `ValueError` on missing keys, wrong shapes, non-finite values, or `|token| > 1.25`.
  - `READY_POSE_PATH` is the constant asset path.

- [ ] **Step 1: Failing tests:**
  - `test_load_ready_pose_shapes_and_sha` asserts the shapes above, and that the file's sha256 equals the provenance JSON's.
  - `test_load_ready_pose_rejects_bad_shapes` covers a temp npz with `raise_tokens` (82, 64), and a NaN token.
- [ ] **Step 2: Run.** Expected: FAIL (module missing).
- [ ] **Step 3: Copy the asset with `git -C ~/work/lerobot show a6e267c6:examples/g1_dex3_training/ready_pose_261020_ep76.npz > …`, write the provenance JSON, and implement the loader.**
- [ ] **Step 4: Run.** Expected: PASS.
- [ ] **Step 5: Commit.**

### Task 3: PoseCommander (holds, blends, paths, policy, freeze/resume, slew)

**Files:**
- Modify: `gear_sonic/utils/inference/loop_motion.py`
- Test: `gear_sonic/tests/test_loop_motion.py`

**Interfaces:**
- Consumes: `TokenStream` (Task 1).
- Produces `class PoseCommander`:
  - `__init__(self, max_token_step: float)`;
  - `start(self, token, hands) -> None`: sets the last-sent reference (the slew origin) and holds it;
  - `hold(self) -> None`;
  - `blend_to(self, token, hands, ticks: int = 50, label: str = "blend") -> None`: linear from the last sent at the segment's first tick, weight `(i+1)/ticks` (lerobot `_blend_to`);
  - `play(self, tokens, hands, label: str) -> None`: one row per tick;
  - `follow(self, stream: TokenStream, label: str = "episode") -> None`: rows from `stream.at(now)`, which applies its own slew;
  - `freeze(self) -> None`: hold the last sent; remember the queued segments and the index in the current one;
  - `resume(self) -> None`: continue the frozen segment from its index, then the rest of the queue;
  - `busy: bool` (segments queued or in progress);
  - `frozen: bool`;
  - `tick(self, now: float) -> tuple[np.ndarray, np.ndarray, str]`: token (64), hands (14), label. It applies the 0.05 slew to the token in every mode except `follow`, where the stream already did;
  - `last_token`, `last_hands`.

  Segments queue in call order: `blend_to` then `play` then `blend_to` runs the raise.

- [ ] **Step 1: Failing tests** (`test_loop_motion.py`):
  - `test_blend_is_linear_from_last_sent`: start at 0, `blend_to` 1 with 50 ticks. Tick 0 → 0.02, tick 49 → 1.0, slew 0.05 not binding.
  - `test_play_emits_one_row_per_tick`
  - `test_slew_limits_token_not_hands`: start 0, `blend_to(1, ticks=1)`. Token 0.05, hands 1.0.
  - `test_freeze_mid_blend_holds_last_sent_token`: freeze at tick 10, then 20 more ticks all equal the tick-10 output.
  - `test_resume_continues_path_from_frozen_index`: freeze at row 30 of `play`, `resume`. The next tick emits row 31.
  - `test_follow_reads_stream_by_time`
- [ ] **Step 2: Run.** Expected: FAIL.
- [ ] **Step 3: Implement** as a list of segment objects with a per-segment tick index, using no threads.
- [ ] **Step 4: Run.** Expected: PASS.
- [ ] **Step 5: Commit.**

### Task 4: Episode guards and HandoverDetector

**Files:**
- Create: `gear_sonic/utils/inference/loop_episode.py`
- Test: `gear_sonic/tests/test_loop_episode.py`

**Interfaces:**
- Produces:
  - `HandoverDetector(open_s=1.0, min_hold_s=0.5, closed_above=0.19)` with `.update(right_hand_mean: float, now: float | None = None) -> bool`, a verbatim port of lerobot `eval_session.py:113-148`.
  - `ARM_LO`, `ARM_HI` (14,), verbatim from `sonic_policy_streamer.py:150-170`. `ARM_SPEED_TICKS = 3`, `TOKEN_ECHO_TOL = 1e-4`.
  - `token_echo_err(token_state, recent_tokens) -> float` and `max_arm_speed(wall0, state0, wall1, state1) -> tuple[int, float] | None`, ported from `stream_backends.py:110-140`.
  - `EpisodeGuards(cap_s=30.0, max_arm_speed=6.0, arm_range_margin=0.15, max_chunk_age_s=1.0, max_state_age_s=0.2, max_camera_age_s=0.5, token_echo_timeout_s=0.5)` with `.check(now, state_msg, state_age_s, camera_age_s, last_chunk_at, recent_tokens) -> tuple[str, str] | None`. It returns `("end", reason)` for the cap, arm speed, arm range and chunk age, and `("freeze", reason)` for state age, camera age and token echo.

- [ ] **Step 1: Failing tests:**
  - Port lerobot `tests/datasets/test_g1_dex3_eval_session.py`'s HandoverDetector cases (grasp then release fires; a flick < 0.5 s doesn't; a startup fist that only opens doesn't; a regrasp resets).
  - `test_guard_categories` checks each limit just over its value, with the expected `("end"|"freeze", reason)`.
  - `test_arm_range_uses_lerobot_limits`: the first ARM_LO/ARM_HI entries equal the reference values copied into the test.
- [ ] **Step 2: Run.** Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run.** Expected: PASS.
- [ ] **Step 5: Commit.**

### Task 5: Trial recorder (lerobot eval_session format)

**Files:**
- Create: `gear_sonic/utils/inference/loop_records.py`
- Test: `gear_sonic/tests/test_loop_records.py`

**Interfaces:**
- Produces:
  - `TrialRecorder(session_dir: Path, trial: int, args: dict, model: str)`;
  - `.tick(wall, t_ep, phase, frame, token, hands, chunk_t0, state28)`;
  - `.frame(t_recv: float, name: str, jpeg: bytes)`;
  - `.close(episode_end: str) -> None`. It writes `trial_NNN.npz` with keys `wall,t_ep,phase,frame,token,hands,chunk_t0,state,joint_ref(nan 31),episode_end,args,model,wall_start,wall_stop`, plus `trial_NNN_frames.npz` (`jpeg,offsets,t_recv,names`), exactly as lerobot `write_trial`. It does **not** write `results.jsonl`.
  - **Persistence is off the 50 Hz path.**
    - `tick`/`frame` only append to in-memory lists.
    - `close()` hands the arrays to a module-level single-worker `TrialWriter` (a daemon thread fed by a `queue.Queue`) and returns at once.
    - `TrialWriter.flush(timeout_s: float) -> bool` waits for pending writes; it runs at session end and at shutdown.
    - Files are written to a temp name and renamed, so a reader never sees a partial file.
  - `next_trial(session_dir) -> int` counts the `trial_*_frames.npz` files, plus trials queued but not yet written.

- [ ] **Step 1: Failing tests:**
  - `test_trial_files_match_lerobot_layout`: write 3 ticks and 2 frames, close, flush. Load both npz files and assert the key sets, shapes, `wall_start/wall_stop` over the "episode" phase rows only, and that the offsets round-trip the JPEG bytes.
  - `test_next_trial_counts_trial_files`.
  - `test_close_returns_before_slow_write`: monkeypatch the npz save to sleep 2 s. `close()` returns in < 0.05 s, and after `flush(5)` the files exist.
- [ ] **Step 2: Run.** Expected: FAIL.
- [ ] **Step 3: Implement** (port `_save_npz` and `write_trial`'s array part).
- [ ] **Step 4: Run.** Expected: PASS.
- [ ] **Step 5: Commit.**

### Task 6: LoopController, the RPC contract and failure taxonomy

**Files:**
- Create: `gear_sonic/utils/inference/loop_control.py`
- Modify: `gear_sonic/utils/inference/harness_contract.py` (PHASES, PARAMS, Status)
- Modify: `gear_sonic/utils/inference/harness_control.py` (dispatch `loop_*` to the LoopController; the lease-expiry and `interrupt()` paths call `loop.freeze(reason)` while a LOOP phase is active)
- Modify: `gear_sonic/tests/fixtures/g1_rpc_v1.json` (golden fixture)
- Test: `gear_sonic/tests/test_loop_control.py`

**Interfaces:**
- Consumes: `PoseCommander`, `ReadyPose`, `HandoverDetector`, `EpisodeGuards`, `TrialRecorder`.
- Produces:
  - New phases: `LOOP_HOLD`, `LOOP_MOTION`, `LOOP_TRIAL`, `LOOP_FROZEN`.
  - New methods and their params:
    - `loop_take_over {}`
    - `loop_raise {}`
    - `loop_trial {"trial": int, "session_dir": str}`
    - `loop_freeze {}`
    - `loop_resume {}`
    - `loop_end {}`
  - `Status` gains `loop_stage: str | None = None` (one of `stance, raise, ready, hold_rest, first_chunk, blend_in, episode, to_ready, frozen, lower, handback`), `loop_trial: int | None = None`, and `trial_end_reason: str | None = None`.
  - `LoopController(commander, ready_pose, hooks)` with:
    - `.take_over(feedback, now)`, `.raise_(now)`, `.trial(trial, session_dir, now)`;
    - `.freeze(reason, now)`, `.resume(now)`, `.end(now)`;
    - `.accept_chunk(epoch: int, captured_at: float, rows: np.ndarray) -> bool`;
    - `.tick(now, feedback, camera_age_s) -> tuple[token, hands, label]`;
    - attributes `stage: str`, `frozen_stage: str | None`.
  - `RuntimeHooks` gains:
    - `switch_to_pose(self) -> None`: sends start with `planner=False`;
    - `switch_to_planner(self) -> None`;
    - `request_policy_chunk(self) -> int`: enqueue one inference epoch and return it;
    - `set_policy_enabled(self, enabled)`, which already exists.

    Publishing is done by the main loop from `tick()`'s return value (Task 7), not by a hook.
- Rules:
  - `loop_take_over` requires the owner, phase `IDLE`/`COMPLETED`, a controller running, `lerobot_policy.chunk_anchor_s > 0` and `handover.checkpoint_verified`. It holds the planner token for 50 ticks, switches to POSE, then holds (stage `stance`).
  - `loop_raise` is accepted only from `LOOP_HOLD` at stage `stance`.
  - `loop_trial` is accepted only from `LOOP_HOLD` at stage `ready`. Its sequence is:
    1. `hold_rest`: hold for 75 ticks, the ready token with open hands.
    2. `first_chunk`: at the end of hold-rest, issue **one bootstrap inference request** via `hooks.request_policy_chunk() -> int` (it enqueues an inference epoch). Keep holding; the hold is published every tick. Accept that request's result through `LoopController.accept_chunk(epoch, captured_at, rows) -> bool`, which is valid in stages `first_chunk` and `episode` and checks the epoch.
    3. `blend_in`: `blend_to` row 0 over 50 ticks. That chunk's t0 is the blend's end.
    4. `episode`: `follow` with anchoring. Request a new chunk every 0.4 s from each capture time (`hooks.request_policy_chunk()`). HandoverDetector and guards run each tick.
    5. On `("end", r)`: `trial_end_reason=r`, then `to_ready` (`blend_to` ready), then `LOOP_HOLD` (stage `ready`).
    6. On `("freeze", r)`: `freeze(r)`.

    Only the commander publishes tokens. Policy outputs reach the robot only through `accept_chunk` and the `TokenStream`.
  - `loop_freeze` is valid in any LOOP phase. In `LOOP_TRIAL` it ends the trial with `trial_end_reason="operator stop"`. It records `frozen_stage`, the stage at the freeze.
  - `loop_resume` is only from `LOOP_FROZEN`. Its destination depends on `frozen_stage`:

    | Frozen stage | Resume | Ends in |
    |---|---|---|
    | `stance` | keep the hold | `LOOP_HOLD` (stage `stance`) |
    | `raise` (the raise-start blend, the raise, or the blend to ready) | resume the path from its frame, then go to ready | `LOOP_HOLD` (stage `ready`) |
    | `hold_rest`, `first_chunk`, `blend_in`, `episode`, `to_ready`, `ready` | `blend_to` ready | `LOOP_HOLD` (stage `ready`) |
    | `lower` or `handback` (the lowering-start blend, the lowering, the blend to stance, the hand-back) | resume the path from its frame, then the rest of the end sequence, then `switch_to_planner` | `COMPLETED`; the session has ended |
  - `loop_end` is only from `LOOP_HOLD` at stage `ready`:
    1. `blend_to` the lowering's first row;
    2. `play` the lowering;
    3. `blend_to` the stance (the planner token from take-over);
    4. the hand-back blend;
    5. `switch_to_planner`;
    6. `COMPLETED`.
  - Lease expiry, the harness `interrupt()` or `operator_override()` (a stray native `p`/`k`/`i`/`m` key) in a LOOP phase calls `freeze()`. It never calls `request_planner_hold`, and the key's own action (pausing, toggling the controller) is not performed. A new claim is allowed from `LOOP_FROZEN`.
  - The commander keeps the last 25 sent tokens for `token_echo_err`.

- [ ] **Step 1: Failing tests** (`test_loop_control.py`, using fake hooks that record pose messages):
  - `test_take_over_holds_planner_then_switches_to_pose`
  - `test_raise_plays_blend_path_blend_to_ready`: 50 + 83 + 50 ticks, ending at the `ready_pose.token` hold.
  - `test_trial_sequence_and_handover_end`: a scripted right hand (open, closed 0.6 s, open 1.1 s) ends the trial with `trial_end_reason == "handover"` and stage `ready`.
  - `test_stop_during_trial_freezes_and_ends_trial` (Review Focus 1)
  - `test_stale_state_freezes_not_blends`
  - `test_stray_native_key_in_loop_freezes_without_its_action`
  - `test_lease_expiry_in_loop_freezes_and_allows_reclaim` (Review Focus 2)
  - `test_resume_mid_raise_continues_raise`
  - `test_resume_mid_lowering_finishes_to_planner_and_completes`: freeze at lowering row 40, resume. Rows 41–82 play, then the blend to stance, the hand-back and `switch_to_planner`; phase `COMPLETED`.
  - `test_resume_mid_handback_completes`
  - `test_first_chunk_bootstrap_requested_after_hold_rest`: exactly one `request_policy_chunk` call at tick 75. The hold continues until `accept_chunk`, and a stale-epoch result is rejected.
  - `test_end_lowers_and_hands_back_to_planner`
  - `test_loop_methods_rejected_outside_their_phase`
  - Golden fixture round trip: the existing contract test is updated with the new methods and status fields.
- [ ] **Step 2: Run.** Expected: FAIL.
- [ ] **Step 3: Implement** `loop_control.py`, the contract additions and the dispatch delegation. `interrupt()` and lease expiry branch to `loop.freeze` when `phase.startswith("LOOP_")`.
- [ ] **Step 4: Run the new file, then `test_harness_control.py`, `test_run_vla_reset.py` and `test_handover_ready.py`.** Expected: all pass; existing behaviour is unchanged outside LOOP phases.
- [ ] **Step 5: Mirror `harness_contract.py` into VoLoAgent `contract.py`** and update its golden fixture copy. Run the mirror diff, then `tests/test_g1_registry_contract.py` and `tests/test_g1_client.py`. Commit in both repos.

### Task 7: Main-loop integration in run_vla_inference

**Files:**
- Modify: `gear_sonic/scripts/run_vla_inference.py`. In harness mode, construct the `LoopController` with a `PoseCommander(max_token_step=profile.lerobot_policy.max_token_step)` and `load_ready_pose()`. Add the new hooks to `LoopHooks`.
- Test: `gear_sonic/tests/test_loop_integration.py`, using `run_keys` from `test_run_vla_reset.py`.

**Interfaces:**
- Consumes: Task 6's `LoopController.tick` and hooks.
- Rules:
  - When `harness_control.phase` is a LOOP phase, each 20 ms loop iteration publishes **exactly one** `pack_latent_action_message` from `loop.tick(...)`. The planner-command branches, `pause_loop` and the existing MANIPULATING publish are skipped.
  - `hooks.request_policy_chunk()` puts a new inference epoch on the existing `inference_queue`. Results from `result_queue` in a LOOP phase go to `loop.accept_chunk(epoch, captured_at, rows)` in stages `first_chunk` and `episode`, and are dropped otherwise. They are never published directly.
  - The existing `action_chunk_exhausted`, `policy_or_task_timeout` and `invalid_or_failed_policy_result` interrupts are replaced, in LOOP phases, by the guards' chunk-age end and an invalid-result freeze.
  - Camera frames delivered to the policy also go to the `TrialRecorder`. `TrialRecorder.close()` is called on the loop thread but only enqueues; writing happens on the `TrialWriter` thread (Task 5).

- [ ] **Step 1: Failing tests:**
  - `test_loop_phases_publish_every_tick` (Review Focus 5): drive take-over, raise, a trial with scripted chunks, a freeze and a resume. Assert one pose message per loop iteration and no planner command after the POSE switch.
  - `test_first_chunk_bootstrap_through_inference_queue`: after hold-rest, one epoch appears on `inference_queue`. Its result, put on `result_queue`, is accepted in `first_chunk`, and the blend-in starts. No iteration lacks a pose message while waiting.
  - `test_slow_trial_write_does_not_stall_ticks`: the npz save sleeps 2 s. The iterations after a trial end still publish one message each, with no gap. The `TrialWriter` thread, not the loop thread, performed the save.
  - `test_loop_trial_uses_anchoring`: the token right after a new chunk equals the last sent, to within 1e-6.
- [ ] **Step 2: Run.** Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run the new file, then the full native suite.** Expected: only the known Inspire failures remain.
- [ ] **Step 5: Commit.**

## Part B: VoLoAgent

### Task 8: Re-arm monitor (empty case, then an upright bottle with no hand)

**Files:**
- Create: `vlm_orchestrator/harness/g1/rearm_monitor.py`
- Test: `tests/test_g1_rearm_monitor.py`

**Interfaces:**
- Consumes: `G1CompletionMonitor` (bounded VLM call, image message building) in `monitor.py`.
- Produces:
  - `RearmMonitor(vlm_call_fn, now_fn, limits)` with `.begin(step: Literal["empty", "bottle"], initial: ObservationSnapshot)` and `.check(snapshot) -> RearmDecision`.
  - `RearmDecision(outcome: Literal["yes","no","unknown","failure"], reason: str, frame_id: str, step: str)`.
  - The prompt asks for JSON `{"case_empty": bool, "bottle_upright_on_case": bool, "hand_at_bottle": bool}`. For step `empty`, yes iff `case_empty`. For step `bottle`, yes iff `bottle_upright_on_case and not hand_at_bottle`.
  - A step completes on the second "yes" from a **distinct** frame id. Any "no" or "unknown" resets the streak.

  The prompt text is copied from `docs/artifacts/g1_vlm_upright_check_20261026/upright_check.py`'s PROMPT, extended with the `case_empty` field.

- [ ] **Step 1: Failing tests:**
  - `test_rearm_requires_empty_case_before_bottle` (Review Focus 3, at the monitor level): in step `empty`, upright-bottle answers never complete it.
  - `test_bottle_step_rejects_hand_at_bottle`
  - `test_two_distinct_frames_required`
  - `test_malformed_json_is_unknown`
- [ ] **Step 2: Run.** Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run.** Expected: PASS.
- [ ] **Step 5: Commit.**

### Task 9: Verdict console and session files

**Files:**
- Create: `vlm_orchestrator/harness/g1/verdict.py`
- Modify: `vlm_orchestrator/harness/g1/cli.py` (subcommand `verdict --session-dir DIR`)
- Test: `tests/test_g1_verdict.py`

**Interfaces:**
- Produces:
  - `FAILURE_REASONS`, a verbatim copy of lerobot `eval_session.DEFAULT_FAILURE_REASONS`.
  - `pending_trials(session_dir) -> list[int]`: trial files without a `results.jsonl` line.
  - `append_verdict(session_dir, trial, verdict: Literal["success","fail","redo"], reason: str, note: str) -> dict`. It appends the lerobot `results.jsonl` line, `{"trial", "verdict", "reason", "note", "episode_end", "wall_start", "wall_stop", "frames", "model"}`, taking the episode fields from `trial_NNN.npz`.
  - `request_session_end(session_dir) -> None`, which creates `end_session.request`.
  - `remaining_trials(session_dir, n_trials) -> int`, which equals `n_trials - (trials recorded - redo verdicts)`.
  - The console loops over pending trials with the prompt `s / f / r [+ reason index] [note]`, plus `e` for end session.

- [ ] **Step 1: Failing tests:**
  - `test_append_verdict_writes_lerobot_line`
  - `test_pending_lists_unscored_trials`
  - `test_redo_extends_remaining_trials` (Review Focus 4)
  - `test_end_command_writes_request_file`
- [ ] **Step 2: Run.** Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run.** Expected: PASS.
- [ ] **Step 5: Commit.**

### Task 10: LoopRunner and the `loop` command

**Files:**
- Create: `vlm_orchestrator/harness/g1/loop.py`
- Modify: `vlm_orchestrator/harness/g1/cli.py` (subcommand `loop`)
- Test: `tests/test_g1_loop.py`

**Interfaces:**
- Consumes:
  - `G1Client`;
  - the heartbeat, lease and claim helpers in `runner.py`;
  - `RearmMonitor`;
  - `verdict.remaining_trials` and `request_session_end`;
  - the Task 6 RPC methods and status fields.
- Produces:
  - `OperatorConsole(stream=sys.stdin)` with `.wait_enter(prompt) -> None`, `.stop_pressed() -> bool` (a background reader; flushes before each gate), and `.close()`.
  - `LoopRunner(profile, client_factory, monitor, session_dir, n_trials, console, confirm_paths: bool = True)` with `.run() -> MissionResult`.
  - CLI: `volo-g1-harness loop --session-dir DIR --trials N --vlm-base-url URL --vlm-model NAME [--no-confirm-paths]`, plus the global `--profile` and `--endpoint`.
  - `EvidenceLog(session_dir)` with:
    - `.event(trial: int, kind: str, **fields) -> None`: appends one JSON line to `trial_NNN_evidence.jsonl`;
    - `.frame(trial: int, snapshot: ObservationSnapshot) -> str`: writes `evidence/trial_NNN/<sha256(frame_id)>.jpg` and returns the relative path.

    Pre-trial events belong to the trial they trigger: index = `next_trial(session_dir)` at the time. It never touches lerobot's `trial_NNN.npz`, `trial_NNN_frames.npz` or `results.jsonl`.
- **Threads:**
  - The main thread owns the RPC client. It polls `console.stop_pressed()` and the status at 20 Hz, and dispatches `loop_freeze` within one poll of a key press.
  - VLM checks run in a worker thread fed with frames the main thread fetched (`observe`). The worker never calls RPC, so a blocked VLM call cannot delay `loop_freeze`.
  - The heartbeat stays on its own thread, as in `runner.py`.
- Flow:
  1. **Claim, then read the status** to choose the entry branch:
     - `IDLE` or `COMPLETED`: the fresh-start branch, step 2.
     - `LOOP_FROZEN` (a reclaim after an orchestrator crash or lease expiry): no take-over, no startup gates, no raise. Go straight to step 6's clear-and-resume handling with the native `frozen_stage`.
     - `LOOP_HOLD` at stage `ready`: the console "restart loop?", then step 5 with the first-trial rule.
     - `LOOP_HOLD` at stage `stance`: step 4.
     - `LOOP_MOTION` or `LOOP_TRIAL` (impossible after a lease expiry, which freezes): report the error and release.
  2. Fresh start: the console "deploy ready", then "robot in position".
  3. `loop_take_over`, then the console "start loop".
  4. If `confirm_paths`: the console "raise?". Then `loop_raise` and wait for stage `ready`.
  5. Trial loop, while `remaining_trials > 0` and there is no `end_session.request`:
     - first trial after a start, restart or reclaim: re-arm step `bottle` only; later trials: `empty` then `bottle`;
     - `loop_trial`;
     - wait for stage `ready`;
     - log the trial end reason to the evidence.
  6. On the stop key at any time, or a native `LOOP_FROZEN` with a freeze reason:
     1. `loop_freeze` (skipped if already frozen);
     2. the console "clear?";
     3. `loop_resume`;
     4. then by the resulting phase: `COMPLETED` means the session ended during the lowering or hand-back, so release and finish with no restart prompt; `LOOP_HOLD` at stage `stance` goes to step 4; `LOOP_HOLD` at stage `ready` means the console "restart loop?", then step 5 with the first-trial rule.
  7. Session end: if `confirm_paths`, the console "lower?". Then `loop_end`, wait for `COMPLETED`, `TrialWriter`-style flush on the native side (session end), then release.
- **Evidence written** (`EvidenceLog`):
  - every re-arm step begin and end;
  - every VLM decision: the outcome, the parsed JSON, the raw text, `frame_id`, and the saved frame path;
  - `loop_trial` start;
  - the trial end reason;
  - stop key press time and `loop_freeze` dispatch time;
  - native freeze reasons;
  - resume, restart and reclaim events.

  Startup gates are explicit console steps; the runner never sends the native `k`.

- [ ] **Step 1: Failing tests**, against a fake client that scripts statuses and records RPCs:
  - `test_full_session_sequence`
  - `test_first_trial_skips_empty_step_later_trials_require_it`
  - `test_stop_freezes_then_clear_resumes_then_restart`
  - `test_native_freeze_handled_as_stop`
  - `test_confirm_paths_gates_raise_and_lower`
  - `test_stop_key_and_verdict_console_are_separate` (Review Focus 4: the loop's console reads only Enter; verdict text written to the verdict console never triggers `loop_freeze`)
  - `test_session_end_request_file_ends_after_current_trial`
  - `test_fresh_runner_reclaims_frozen_loop_without_takeover`: the first runner's lease expires and the status becomes `LOOP_FROZEN` (`frozen_stage="episode"`). A new `LoopRunner` claims and sends no `loop_take_over` or `loop_raise`. It prompts "clear?", sends `loop_resume`, then "restart loop?", and runs the next trial with the bottle-only re-arm.
  - `test_reclaim_frozen_during_lowering_completes_without_restart`: `frozen_stage="lower"`; after resume the status is `COMPLETED`; there is no restart prompt and control is released.
  - `test_stop_dispatched_while_vlm_call_blocks`: the fake VLM call sleeps 5 s; the stop key is pressed 0.5 s into it. `loop_freeze` is sent within 0.2 s of the key press, measured by the fake client's request times.
  - `test_evidence_sidecar_records_trigger_and_stops`: after one trial with a stop, `trial_000_evidence.jsonl` holds the re-arm decisions with existing frame files, the `loop_trial` start, the end reason, and the stop and freeze times. The lerobot trial files are untouched.
- [ ] **Step 2: Run.** Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run the new file, then the full VoLoAgent suite.** Expected: PASS.
- [ ] **Step 5: Commit.**

### Task 11: Local VLM server (Qwen3-VL-2B, OpenAI-compatible)

**Files:**
- Create: `scripts/g1_local_vlm_server.py` (stdlib `http.server`; lazy `transformers` import; runs under the lerobot venv)
- Test: `tests/test_g1_local_vlm_server.py` (a fake model; runs in the VoLoAgent venv)

**Interfaces:**
- Produces:
  - `POST /v1/chat/completions`, accepting the OpenAI message subset the monitor sends (text plus `image_url` data URLs). It returns `{"choices": [{"message": {"content": text}}]}`.
  - `GET /v1/models`.
  - `build_app(generate: Callable[[list[PIL.Image], str], str]) -> http.server.BaseHTTPRequestHandler`.
  - `main(--model PATH --port 8301 --device cuda:0 --min-free-gb 4.4)`, which exits non-zero if the free memory on the device is below `--min-free-gb`.
  - Generation is greedy, `max_new_tokens=60`.

- [ ] **Step 1: Failing tests:**
  - `test_chat_completion_round_trip_with_fake_generate`
  - `test_rejects_request_without_image`
  - `test_refuses_start_when_gpu_memory_low` (patched free-memory function)
- [ ] **Step 2: Run.** Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run, then a manual check on GPU 0.** Run `CUDA_VISIBLE_DEVICES=0 ~/work/lerobot/.venv/bin/python scripts/g1_local_vlm_server.py --model <Qwen3-VL-2B snapshot> --port 8301`. Then run `volo-g1-harness`'s RearmMonitor against `docs/artifacts/g1_vlm_upright_check_20261026/frames`. Expected: the same 17/17 combined result as `qwen3-vl-2b.json`.
- [ ] **Step 5: Commit.**

## Part C: Validation

### Task 12: Parity against lerobot's own code

**Files:**
- Create: `docs/artifacts/g1_handover_20261026_loop_parity/parity_check.py` (lerobot venv; imports lerobot's `sonic_token_stream`, `stream_backends`, `eval_session` from a6e267c6 alongside the native modules, as `g1_handover_20261021_s10_loop_sim/parity_check.py` did)

- [ ] **Step 1: Write the script.** On the s16 and s19 session logs (`/mnt/data/jihun/g1_eval_sessions/`), compare:
  - `TokenStream` with anchoring against `ChunkResampler` plus `SonicBackend.send` (token max diff ≤ 1e-6, hands equal);
  - `HandoverDetector` firing times on the logged right-hand series (identical);
  - raise and lowering tick outputs against `SonicBackend._blend_to/_play50/go_ready` (identical).
- [ ] **Step 2: Run** `~/work/lerobot/.venv/bin/python parity_check.py OUT.json`. Expected: every check passes. Save `parity-*.json`.
- [ ] **Step 3: Commit** the script and results (VoLoAgent).

### Task 13: Simulation run of the full loop

**Files:**
- Create: `docs/artifacts/g1_handover_20261026_autonomous_loop_sim/`. It reuses the 20261021 fixture's scene, the D435 camera and the scripted person: copy the files rather than edit that folder, which has another session's uncommitted edits.

**Isolation:** run on an **isolated non-zero DDS domain**, `--dds-domain 37` (deploy flag from `d0f29b3`; the sim's `DOMAIN_ID`, as the 20261021 fixture passes it). Rebuild the workstation deploy binary from the new native branch head before the run, and record its sha256 in the result. Strip the snap EGL environment for headless rendering (memory note `sim-dds-domain0-crosstalk`).

- [ ] **Step 1: Write the probe.**
  - It starts the sim, deploy, native harness and the fake or real s08 server, plus the local VLM server.
  - It answers the console gates from a script.
  - The scripted person places and takes bottles, including one trial that leaves the bottle standing and one stop mid-raise.
- [ ] **Step 2: Run 3 trials.** Expected:
  - the trial count and end reasons are as scripted;
  - the leftover bottle does not start a trial until the person removed it;
  - the stop freezes, resumes the raise, and restarts;
  - one pose message per tick throughout;
  - real box contacts are counted (not `clearance`).
- [ ] **Step 3: Commit** the evidence and a README.

### Task 14: Robot gates (supervised, user present)

- [ ] **Step 1: Capture real frames at the robot** (`record` the head camera with the harness idle):
  - a bottle lying on the case in 3 spots;
  - upright with a hand near it;
  - an empty case;
  - a person reaching in.

  Label them, and rerun `upright_check.py` and the RearmMonitor via the local server. Gate: no miss on a lying bottle or a hand.
- [ ] **Step 2: Redeploy to PC2:**
  - Native tree in a new folder `~/GR00T-WholeBodyControl-loop-<date>`: `gear_sonic` from the new branch head. Rebuild the deploy from the same head; on the robot it runs with the default domain 0.
  - The VoLoAgent package.
  - A **session copy** of `configs/g1/handover-loop.example.yaml` with `checkpoint_verified: true`, stating in its header that it is for the supervised session only.
  - Confirm the camera colours against an s08 frame.
- [ ] **Step 3: Supervised session** (`--trials 5`, with `--confirm-paths` on by default). Log the raise and lowering against the recorded paths, and score every trial.
- [ ] **Step 4: Report the results to the user.** Dropping `--confirm-paths` (unattended) needs the user's go-ahead.
