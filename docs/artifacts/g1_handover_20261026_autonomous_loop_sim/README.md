# Autonomous G1 handover loop in simulation (2026-10-26, Task 13)

SIMULATION FIXTURE ONLY. This run checks the mechanics of the whole autonomous loop end to end. It does not qualify the s08 policy, the ready pose or any physical path.

The loop under test is the one built in Tasks 1-12. VoLoAgent's `cli loop` (LoopRunner, RearmMonitor, the Enter-only operator console) drives the native harness (run_vla_inference with HarnessControl and the LoopController) over its RPC. The native harness streams one pose message per 50 Hz tick to the SONIC deploy in MuJoCo. The verdict console runs as a separate process and writes lerobot's `eval_session` files.

| Part | What runs |
|---|---|
| Policy | s08 `groot_handover261015_he_full` (the reference model), served by the already-running lerobot `sonic_policy_server.py` on :5571 (GPU 1). It is driven through the native LeRobot client with the profile's chunk blend 0.3 s, token step 0.05 and anchoring 1.0 s. |
| Loop, harness, deploy | Production code. Deploy `g1_deploy_onnx_ref` is rebuilt from the native head (sha256 below), on DDS domain 37, with `--harness-planner-hold`. |
| Camera | The simulator renders the D435 head view (`d435i_head.json`, `scene_s10.py`, copied from `g1_handover_20261021_s10_loop_sim`). |
| Re-arm vision | **Oracle.** `oracle_vlm_server.py` serves the OpenAI chat API on :8302 and answers both re-arm prompts (`upright_check`, `object_on_case`) from the simulator's bottle state. RearmMonitor runs unchanged with `--vlm-base-url` pointing at it. See "VLM side check" for why. |
| Operator console | **Scripted** (`run_autonomous_loop.py`). It writes Enter to the `loop` process's stdin for every gate. It sends one stop mid-raise and waits 1.5 s at "clear?", because R21 ignores presses in the first 1 s. |
| Verdict console | `cli verdict` as its own process. Its stdin is scripted from the person's judgement of each trial. |
| Person | **Scripted.** It places a bottle when the runner waits for one (re-arm step "bottle"). It takes the bottle from the hand once the hand has held it for 2 s and lifted it 5 cm, or from an open palm after 1 s. It clears the case after each trial. Once, it leaves a bottle standing for 10 s first. |
| Grasp | The contact-free prop of the 20261021 fixture (`loop_sim_worker.py`, copied and amended; its docstring lists the changes). The bottle attaches when the right hand closes past 0.3 within 0.10 m. Opening below 0.2 leaves it on the palm if the palm is up (wrist roll > 1.2 rad); otherwise it drops with contacts. |

The leftover-bottle trial is forced by the fixture. When the person places the second trial's bottle and no trial has yet left one standing, it writes `no-grasp-N`, and that bottle never attaches.

## Results (`runs/sim-03`, the reference run; `runs/sim-02` is an identical earlier run)

| Expected | Measured (sim-03; sim-02 in brackets) | |
|---|---|---|
| Trial count and end reasons as scripted | 3 trials, all ending `handover complete`. Verdicts: s / f "other: fixture: grasp disabled (leftover-bottle trial)" / s. The session ended `completed` after "lower?" and the hand-back. [same] | pass |
| Leftover bottle | Trial 1's bottle was left standing for 10.05 s. The empty step answered `no` 112 times and `yes` 0 times while it stood. The case was confirmed empty 0.26 s after removal. The new bottle was placed 1.37 s after removal, and trial 2 started 0.18 s after the placement. [10.03 s, 111 no / 0 yes] | pass |
| Stop mid-raise: freeze | Pressed at raise row 16. The runner got the press in 22 µs. `loop_freeze` was acked in 32.6 ms, and the first frozen tick came 31.8 ms after the press. The bound is the runner's 50 ms poll plus one tick. [row 14; 11.6 ms] | pass |
| Resume continuity | Labels: `to raise start` 50, `raise` 16, `frozen` 78, `raise` 67, `to ready` 50. That is all 83 raise rows, each played once. Frozen ticks repeat the last raise token exactly (spread 0.0). The resume step is 0.050, equal to the raise's own maximum step (the 0.05 slew). Then "restart loop?". [14 + 69 rows] | pass |
| One pose message per tick | Native-side log: 3650 loop ticks and 3650 pose messages, exactly 1 per tick, frame index +1 every time. Inside the loop there were no planner messages except the hand-back's single planner hold and the command `[1,0,1]` in the last tick. The only command was the POSE switch `[1,0,0]` at the end of the 50 take-over ticks. [3548 / 3548] | pass |
| No send gap over 25 ms in POSE | 3599 gaps, median 20.1 ms, p99 21.7 ms. **One gap of 26.5 ms**, in an episode. [3 gaps: 26.0 ms in a ready hold, 25.9 and 34.5 ms in episodes] | **fail (marginal)** |
| Real box contacts (MuJoCo `data.contact`, not `clearance`) | During trials: 0. During `to raise start`: 1 sample (left index finger). **From the lowering on, the right thumb rests on the case**: through `lower`, `blend out`, `hold rest`, the hand-back and after it (121 of 2136 samples in the session). The same contact happened in the dead-stream stance. [119 samples, lowering onwards, plus the right wrist] | counted; finding |
| "Planner timeout" at take-over (R19) | 0 lines in both take-overs, but see the note below. | pass, small margin |
| Dead pose stream (R19), separately after the session | See "Dead pose stream". The deploy stays in POSE on the last token, logs no warning, and the robot keeps standing. | observed |
| lerobot `summarize` reads `results.jsonl` | `success 2/3; other: 1` (from `eval_session.py` at lerobot a6e267c6, extracted with `git show`). | pass |

**Policy in simulation.** These results do not qualify the policy.
- In the unforced trials, s08 grasped at 3.1 cm and 5.8 cm (sim-03), and at 1.5 cm and 6.4 cm (sim-02). The person took the bottle from the hand, or once from a palm-up release.
- In the aborted `sim-01` (not kept), s08 grasped at 1.9 cm, then released palm-down 0.3 s later, and the bottle fell. The loop scored it `f 3`, went back to ready and continued, as required.

**Planner timeout margin.**
- The session's take-over followed no planner message at all: the claim sends no hold because `ready_pose_reviewed` is false. So the deploy's timer had nothing to time out.
- The dead-stream take-over followed the hand-back's 50 Hz planner hold. There the planner stream was silent for **1.006 s** before the POSE switch, 6 ms past the deploy's 1000 ms `PLANNER_TIMEOUT`. No timeout was logged, but the take-over's 50 ticks (about 0.99 s plus scheduling) race that timer.
- Per `zmq_manager.hpp`, a timeout resets planner movement to IDLE and keeps the harness-hold overrides. It is expected to be harmless, but a "Planner timeout" line at take-over on the robot is plausible and should not alarm the operator.

**Gap note.** The native loop never logged "Autonomous Loop Missed": no iteration overran its 20 ms. The late ticks were wake-up delays, with tick `now` gaps of up to 26.2 ms. They are likely GIL or scheduling contention with the inference worker on a loaded workstation (load average about 5).

## Dead pose stream (R19, `runs/sim-03`, after the session completed)

The step: a fresh claim, `loop_take_over`, then LOOP_HOLD stance with the deploy in POSE. After 1 s, the native harness was killed with SIGKILL, and the run recorded 3.5 s.

- **Deploy log.** It received the last pose (frame 3755) and kept copying the same external token (`tokens[0]=0.2500`) once per second. The "Streaming data mean delay" in its 1 Hz timing line grew from 18 ms to 72 ms and then 177 ms, and the std from 58 ms to 510 ms.
  - No "[Token Safety] No tokens received" warning appeared within 3.5 s.
  - It did not switch to PLANNER, damp or stop.
  - The process stayed alive.
- **Robot.** It kept standing: pelvis height 0.7877-0.7878 m, all 8 foot contacts kept. Its largest joint change over 3.5 s was 0.041 rad: the right arm settled from thumb contact to wrist contact on the case.
- **For the runbook.** A dead native harness in a LOOP phase leaves the robot holding its last token indefinitely, with no alarm in the deploy log except the rising streaming delay. The operator must stop it from the deploy terminal or the remote (R16f/R19).

## VLM side check (real Qwen3-VL-2B on rendered frames)

`vlm_side_check.py` sent 9 rendered frames to the local Qwen3-VL-2B server on :8301 with RearmMonitor's two prompts, image first, verbatim:
- upright: bottle steps the oracle answered yes, plus a leftover frame;
- empty: the empty steps;
- held: trial frames while the fixture state was `in_hand`.

The server was another session's, the same script and snapshot. It was already running when sim-03 started (see Limitations). The frames are in `runs/sim-03/vlm-side-check/`.

| Frames | Real VLM's answers | Re-arm step it would pass |
|---|---|---|
| Bottle upright on the case (3) | upright true, **hand_at_bottle true** (the ready-pose hand is in view beside the bottle); object_on_case true | none: the bottle step never passes |
| Empty case (3) | 2 correct; 1 says **upright true** on an empty case | empty step on 2 of 3 |
| Bottle held in the hand (3) | hand_at_bottle false; **2 say upright true** | the bottle step **on 2 of 3 (false yes)** |

The real VLM agreed with the truth on 2 of 9. A real-VLM run of this simulation would not be meaningful:
- the bottle step would never pass, because the hand stays in view;
- held-bottle frames would pass it falsely.

The simulated white cylinder and the robot's hand in view at the ready pose are not what the prompts were validated on (17/17 on real clear-PET frames). This is about the simulation's appearance, not the prompts, which were not tuned.

## Provenance

- Native `/home/jihun/work/VoLoAgent/.worktrees/g1-loop-native` at `ef8a5ee`; VoLoAgent at `77ed556` plus this folder's fixture files.
- The deploy binary was rebuilt from the native head with `just build` after `source scripts/setup_env.sh`. Its sha256 is `679c566a514ce69ee31c7b2d7b2dd34f85f3ccec777425b854a9e01a888a554b`, recorded in each `result.json`.
  - The worktree held Git LFS pointer files; `libunitree_sdk2.a` failed to link until `git lfs checkout` of `gear_sonic_deploy/thirdparty/**`.
  - The sim also needs `gear_sonic/data/robot_model/**` checked out.
  - The deploy's source is unchanged since `d0f29b3`.
- SONIC v1.1 decoder, encoder and planner, and the reference motion, come from `~/work/GR00T-WholeBodyControl`, as in the 20261021 fixture. Their `.trt` caches are there too.
- The runs were made on 2026-10-26 between 17:55 and 18:11 KST, on DDS domain 37. Another deploy (pid 1827599) was on domain 0 and the s08 server (pid 3707980) on GPU 1; neither was touched.

## Files

- `run_autonomous_loop.py`: the probe (processes, scripted console, person and verdicts, the dead-stream step).
- `analyze_run.py`: the measurements behind `analysis.json`. Gunzip a run's `*.gz` logs before re-running it.
- `oracle_vlm_server.py`, `vlm_side_check.py`, `loop_native_worker.py`: the native harness plus its message log, which records every send on the action socket and every `LoopController.tick`.
- `loop_sim_worker.py`, `scene_s10.py`, `d435i_head.json`: copied from `g1_handover_20261021_s10_loop_sim`; only the worker is amended.
- `runs/<run>/`:
  - `result.json` and `analysis.json`;
  - `events.jsonl` (probe events: console presses, person, verdicts);
  - `oracle.jsonl`;
  - the logs, gzipped: `native-messages.jsonl.gz`, `sonic.log.gz` and `sonic-lines.jsonl.gz` (deploy lines timestamped by the probe), `ground-truth.jsonl.gz` (20 Hz sim state with `box_contacts`), plus the native, loop, verdict and mujoco logs;
  - `session/`: `results.jsonl`, `trial_NNN.npz`, the `trial_NNN_evidence.jsonl` sidecars and the runner's `evidence/<session>/events.jsonl.gz`.
- **Kept out for size:**
  - `trial_000_frames.npz` is kept for sim-03 only (11 MB); the other trials' frames (9-13 MB each) and all of sim-02's are dropped;
  - the re-arm evidence JPEGs (about 6 MB per run) are dropped; `vlm-side-check/` keeps 9 representative frames.

```
# VoLoAgent worktree root; s08 server already on :5571
PYTHONPATH=. .venv/bin/python docs/artifacts/g1_handover_20261026_autonomous_loop_sim/run_autonomous_loop.py \
  --policy-endpoint 127.0.0.1:5571 --trials 3 --gpu 0 --dds-domain 37 [--no-vlm-server] --output OUT
.venv/bin/python docs/artifacts/g1_handover_20261026_autonomous_loop_sim/analyze_run.py OUT
PYTHONPATH=. .venv/bin/python docs/artifacts/g1_handover_20261026_autonomous_loop_sim/vlm_side_check.py OUT  # needs :8301
```

## Limitations

- **Oracle vision.** Re-arm decisions came from scene truth, not from the VLM; the side check shows why. The oracle answers with the state at request time, which can be up to 0.5 s after the frame was captured.
- **Fixture-forced leftover.** The leftover bottle in trial 1 came from `no-grasp`, not from the policy. The re-arm logic does not distinguish the two.
- **Scripted person.** The take-from-hand rule (held 2 s, lifted 5 cm) is a fixture choice. It is what makes the trials end `handover complete`.
- **Lowering contacts.** The right hand touching the case from the lowering on is a simulation observation at the real setup's 0.28 m case distance. The physical lowering path should be checked on the robot.
- **Two complete runs.** sim-02 and sim-03 are complete. sim-01 was aborted by a probe bug (the person thread crashed on a duplicate key) after trial 0 and is not kept. sim-02's dead-stream step failed on a probe bug, a wait that watched the already-exited loop process; it was fixed for sim-03.
- **Borrowed VLM server.** sim-02 started its own local VLM server. For sim-03, another session's server was already on :8301 (started 18:03 from this worktree with `--host 0.0.0.0`), so sim-03 ran with `--no-vlm-server` and the side check used that server.
- **Message log tail.** The native message log is written every 0.25 s, so the last 0.14 s before the SIGKILL is missing from it. The deploy log shows frame 3755 arriving.
