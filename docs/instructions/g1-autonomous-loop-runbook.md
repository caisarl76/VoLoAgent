# G1 autonomous handover loop: robot runbook

Supervised sessions of the autonomous loop (spec `docs/superpowers/specs/2026-10-26-g1-autonomous-handover-loop-design.md`).
The operator stays at the robot for every session until the user approves unattended runs (`--no-confirm-paths`).

## Hosts and processes

| Where | Process | Notes |
|---|---|---|
| Workstation 192.168.0.62, GPU 1 | s08 policy server, port 5571 | `groot_handover261015_he_full`. It is long-running and shared, so leave it as it is. |
| Workstation, GPU 0 (T0) | Qwen3-VL-4B server, port 8301 (tmux `vlm`) | `scripts/g1_local_vlm_server.py --model <Qwen3-VL-4B-Instruct snapshot> --host 0.0.0.0 --min-free-gb 9`. It uses ~8.8 GB. The 4B weights are in the snap HF cache (`~/snap/alacritty/common/.cache/huggingface/hub`). `/v1/models` still lists the name `qwen3-vl-2b`, but the model served is the one given by `--model`. |
| PC2 (`pc2_222`) | deploy (T1), native harness (T2), loop runner (T3), verdict console (T4), telemetry recorder | The runner must run on PC2, because frame ages are compared on one clock. |

The deploy binary is the 10-22 build (`~/GR00T-WholeBodyControl-harness-20261022`, native `cba799d`). The only deploy change since then is `--dds-domain` (`d0f29b3`), which defaults to 0. The loop's Python code runs from `~/GR00T-WholeBodyControl-loop-<date>`, which holds `gear_sonic`, `vlm_orchestrator`, `session-profile.yaml` and `*_SOURCE_COMMIT`. It uses the 10-22 venv, and `PYTHONPATH` selects the new code.

## Before the session

1. **Pull the code.** Archive `gear_sonic` from the native branch head (without `data/robots` and `data/assets`) and `vlm_orchestrator` from the VoLoAgent head into the new folder. Write the source commits.
2. **Session profile.** Copy `configs/g1/handover-loop.example.yaml` with `checkpoint_verified: true`. Its header must say that it is for this supervised session only.
3. **Served model.** `loop_take_over` refuses unless the server reported `profile.checkpoint` as its model. The native harness asks once, when its worker first connects; **restart T2 after any policy-server restart**. Manual check from PC2:
   `PYTHONPATH=$PWD <venv>/bin/python -c "from gear_sonic.utils.inference.lerobot_sonic_policy import LeRobotSonicPolicyClient as C; print(C('192.168.0.62', 5571, 10000).model)"`
4. **VLM server reachable from PC2.** `curl -s http://192.168.0.62:8301/v1/models`.
5. **Re-arm gate on real frames, with the robot holding the ready pose** (spec: any miss blocks the rollout).
   - In simulation, the ready-pose hand was in view and read as `hand_at_bottle`. The validation frames were taken without the arm at the ready pose.
   - **How to hold the ready pose:** start the loop (T1-T3), answer the gates through the raise, keep the case empty, and press Enter during the re-arm wait. The robot freezes at ready. Leave "clear?" unanswered.
   - **Frames to capture** (capture script in another terminal):
     - a bottle lying on the case in 3 spots;
     - an upright bottle with a hand near it;
     - an upright bottle alone;
     - an empty case;
     - a person reaching in.
   - Run them through `RearmMonitor` and the local server.
   - **Gate:**
     - bottle step "no" on every lying, hand and reach frame;
     - bottle step "yes" on the upright frames;
     - empty step "yes" on the empty frames.
   - Then answer "clear?" (the restart flow continues) or end the session.
6. **Camera colours.** Compare a saved frame with an s08 frame. The bottle cap must be yellow-green.

## Startup order

1. Power up, stand and reset the robot **away from the case**. SONIC's standing pose reaches into a desk-height case.
2. Start the deploy (T1) with `--harness-planner-hold`, and wait for it to hold the stance.
3. Start the native harness (T2). Then send **`k` once**. The harness does not read keys from its own terminal: it subscribes to ZMQ port 5580, so send the key from a PC2 shell:
   `~/GR00T-WholeBodyControl-harness-20261022/venv/bin/python -c "import zmq,time; s=zmq.Context().socket(zmq.PUB); s.bind('tcp://127.0.0.1:5580'); time.sleep(1); s.send_string('k'); time.sleep(0.5)"`
   - This starts the deploy's C++ control loop in PLANNER mode, and the robot goes to planner standing.
   - Without it, `loop` stops with "Operator must prepare a running paused executor".
   - Before starting `loop`, wait until `status` shows `controller_running: true` and `policy_ready: true`.
   - `k` is a toggle, so sending it again stops the control loop. Never send `k` again during the session.
   - The same one-shot sender delivers `i` (see Recovery).
4. Start the telemetry recorder in its own tmux session, one per session name. It refuses to start if another recorder is already writing the same file:
   `tmux new-session -d -s recorder "cd ~/GR00T-WholeBodyControl-loop-<date> && PYTHONPATH=\$PWD <venv>/bin/python g1_record_telemetry.py telemetry-<name>.npz --seconds 1800 2>&1 | tee recorder-<name>.log"`
   To stop and save early, run `kill <pid>`, using the pid printed in the first line of `recorder-<name>.log`. Do not use `pkill`/`pgrep -f`: the pattern can match your own shell.
5. Start the loop runner (T3): `python -m vlm_orchestrator.harness.g1.cli --profile session-profile.yaml --endpoint ipc:///tmp/volo-g1-harness.sock loop --session-dir sessions/<name> --trials N --vlm-base-url http://192.168.0.62:8301/v1 --vlm-model qwen3-vl-4b`. `--profile` is required: without it the runner stops with "Handover loop profile required".
   - A finished session (no trials left for `--trials`, counting redo verdicts) is refused before the claim: `session <dir> has no trials left (N recorded); use a new --session-dir`. Use a new directory, or a larger `--trials` to extend the session on purpose. A crashed session that is still frozen (LOOP_FROZEN) is reclaimed regardless, so a frozen lowering can finish.
6. Start the verdict console (T4): `verdict --session-dir sessions/<name>`.
7. The runner asks for the gates in order: "deploy ready", "robot in position", "start loop", then "raise?".
8. Bring the case in only after the raise has reached the ready pose.

## During the session

- **Enter = stop**, from the take-over on, at any time outside a gate prompt. The robot freezes and holds the last command.
- At "clear?", presses in the **first 1 s are ignored** (R21). This keeps the second tap of a panicked double press from resuming the robot. Press Enter again once the path is clear.
  - A stopped raise or lowering resumes from the frame where it stopped.
  - A stopped trial, including one stopped while holding its offer pose after the trial, blends to the ready pose. Then "restart loop?" follows, unless no trial remains or `e` was entered; then the runner goes to "lower?".
- **Re-arm before each trial.** The runner prints each decision change. A `failure` line means the VLM call failed: the server is unreachable, timed out or returned an HTTP error. Check T0. A long run of `no hand_at_bottle` with nobody near the case means the ready-pose hand is being read as a hand: stop the session and report it.
  - The first trial after a start, restart or reclaim needs only an upright bottle with no hand at it.
  - After a trial that ended with `handover complete`, the next trial also needs only the upright bottle (R30): the bottle left with the person, so you can place the next one right away. The runner prints `re-arm: empty step skipped (trial N ended with a handover)`.
  - After any other trial end (time limit, guard, stop), the case must be seen empty first, then with an upright bottle. The old bottle may still be on the case or lying on it. Clear it, then place a fresh bottle upright.
- **After each trial the arm holds its offer pose** (R24-R26) at `bottle taken? (...)`. The arm moves on to the ready pose by itself once the right-arm torque shows the bottle lifted off the palm for 1 s (`taken: detected by arm torque`, then `to ready`): shoulder and elbow unload by 0.8/0.45 N·m, or the wrist yaw by 0.35 N·m (R31). The session end goes through this hold too.
  - The camera route (R29) is **off** by default since R30: at the offer pose the head camera sees only an edge of the bottle, and in r33 it called the hand empty with the bottle still on the palm in 4 of 6 holds. Its answers are still logged. `--taken-vlm-frames 3` turns it back on, for tests only.
  - Enter is the manual override: press it when the person has taken the bottle but the arm did not move on, for example a take within the first 2 s of the hold (the detector is still measuring its baseline) or a take the detector missed. Presses in the first 1 s after the prompt are ignored, and an Enter pressed before the prompt appears is a stop.
  - An Enter pressed just as the torque detector moves the arm on counts as an answer to this prompt, not as a stop. Press Enter again to stop the move to ready.
  - **Take the bottle cleanly: do not push the palm up or lean on the bottle.** Lifting the weight off the palm is exactly what the detector looks for, so a push that carries most of the bottle's weight for 1 s moves the arm on with the bottle still there.
  - The detector's loaded reference is the arm's torque at the trial end, or the hold's first 0.3 s if that is less loaded. If the arm was still settling at the trial end, the harness log prints `take reference ... (unsteady: waiting for a steady hold)`, and the reference becomes the hold's first steady second (R30). A take before that steady window is left to your Enter.
  - **A hand already empty when the hold starts** (the bottle taken as the hand opened, or a missed grasp) leaves no unload for the torque to see. Since R32 the hold also ends when the camera sees the hand empty 3 times in a row **and** the wrist carries no bottle (wrist yaw steady at or below 0.5 N·m): `taken: the camera sees the hand empty and the wrist carries no bottle`. Neither signal alone moves the arm. If neither fires, press Enter. The harness log's `episode end torque ... from -3 s` line records the arm's load over the episode's last 3 s, so these cases can be studied (R31).
  - The harness log (T2) prints `[loop] hold +N s torque sh/el = … | sh roll/yaw = … | wrist r/p/y = …` every 0.5 s of the hold. These lines are diagnostics for a later wrist-torque detector.
  - The deploy must publish `motor_torque` (the 2026-10-27 build in `~/GR00T-WholeBodyControl-deploy-20261027`); with the 10-22 deploy there is no torque and every hold waits for Enter.
  - The `taken: no` / `taken: yes` lines printed during the hold are the VLM's view, for information only: they never move the arm. A false `taken: yes` (the bottle at the image edge, still on the palm) was seen on the robot, so only your Enter counts.
  - Repeating `failure` lines mean the VLM is down, as for the re-arm.
- **Scoring** is done in the verdict console only: `s`, `f <reason#> [note]`, `r`. Reason numbers are 1-based: 1 did not reach, 2 grasp failed / missed, 3 dropped, 4 wrong object, 5 collision, 6 stalled / froze, 7 safety stop, 8 other. Each verdict appends a line to `results.jsonl` at once; if the file is missing after a session, nothing was scored. Scoring never moves the robot. `e` ends the session after the current trial.
- **Before answering "lower?", move the case away.** In simulation the right thumb rested on the case from the lowering through the stance.
- **Expected log lines:**
  - A `Planner timeout` line at take-over is possible and harmless. The planner silence there is about 1.0 s, against the deploy's 1.0 s timeout, and the harness hold keeps the targets.
  - An occasional late tick of up to about 35 ms is harmless: the deploy holds the last token across it.

## Recovery

| Situation | What to do |
|---|---|
| Robot must stop *now*, or anything unexpected | Enter in T3. If T3 is dead, use the deploy terminal (T1) or the Unitree remote damping. |
| Native harness (T2) crashed or was Ctrl-C'd while in POSE | The deploy silently **holds the last token** forever: no alarm, no fallback (seen in simulation). Stop from T1 or the remote. Never rely on Ctrl-C of T2 as a stop. |
| Loop runner (T3) crashed | The lease lapses within 2 s and the loop freezes (LOOP_FROZEN). Start `loop` again with the same `--session-dir`. It reclaims without take-over or raise, asks "clear?", resumes, then asks "restart loop?". A frozen lowering completes the session instead. |
| Native keys in a loop phase | The motion keys (`i m p k [ ] w s a d q e z`) and prompt changes freeze the loop and skip their own action. Other keys keep their harmless action. Recover through T3. |
| Ctrl-C/EOF at a startup gate | The runner normally releases cleanly, and the next `loop` claims as usual. Only if the release failed and the lease lapsed into INTERRUPTED does the next `loop` get NOT_READY on claim. Then send `i` to port 5580 with the one-shot sender from the startup order (not typed in T2): **`i` also runs the harness's standing reset, so the robot moves; stand clear.** Then start `loop` again. |
| `loop` fails with "Policy prewarm unavailable" | The policy server did not answer when the harness first asked. Check :5571. The harness retries its client on later observations; if it keeps failing, restart T2. |
| `trial_files_missing` event at the session end | A trial's files were not written within 10 s, which is usually a disk error. Read T2's stderr. That trial has no verdict slot, but its number is used. |
| Take-over refused: served model differs | The server on :5571 is serving another checkpoint. Do not run the session until the s08 server is back. |

## After the session

- Pull `sessions/<name>`, the telemetry npz, `harness-*.log` and `*_SOURCE_COMMIT` into `docs/artifacts/g1_handover_<date>_robot_loop/`.
- Check the raise and the lowering in the telemetry against the recorded paths in `gear_sonic/data/handover/ready_pose_261020_ep76.npz`.
- `results.jsonl` is lerobot's format. `eval_session.summarize` and `eval_session_to_lerobot.py` read it directly. Redo verdicts are in `redo.jsonl`.
