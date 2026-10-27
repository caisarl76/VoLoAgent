# G1 autonomous handover loop on the VoLoAgent native harness

Status: approved design (2026-10-26). The robot rounds changed several behaviours, notably the return to ready (it now waits for "bottle taken?") and the re-arm steps; rulings R24-R32 in `.superpowers/sdd/2026-10-26-g1-autonomous-handover-loop/progress.md` and the runbook describe the current behaviour.

## Intent

The robot repeats the bottle handover on its own: rise to a ready pose, wait until a bottle stands on the case, pick it and hand it to a person, return to the ready pose, and repeat. Today this loop runs only in lerobot's `sonic_policy_streamer.py` evaluation sessions, where the operator presses Enter at every hold. The goal is an **autonomous** loop on **VoLoAgent's own native harness**. The operator starts it, watches it, scores it, and steps in only when something is wrong.

Why the native harness: it is VoLoAgent's single robot-side stack. It already has the leases, freshness checks, interrupts and the VLM monitor, and it drives s08 through a client that matches lerobot's streamer byte for byte.

## Goals

1. **Same behaviour as lerobot's validated pipeline** (caisarl76/lerobot main a6e267c6; robot sessions s18/s19 10/10). It uses the same rules and settings, listed below. Where this document does not say otherwise, the lerobot code is the reference.
2. **No Enter at the loop's holds.** The pipeline moves on by itself: raise, start of each trial, return to ready, lowering.
3. **The pipeline recognizes the bottle, with a re-arm (decided with the user, 2026-10-26).** After every trial, before the next one starts, the VLM must see, in this order and each on two distinct head-camera frames (the existing monitor rule):
   1. **the case empty**: no bottle on the black case;
   2. **a drink bottle standing upright on the case with no hand (robot or human) at it.**

   A bottle that stayed on the case after a missed grasp or the cap, a bottle lying down, a held bottle, or a hand still at the bottle never starts a trial.
4. **The operator can stop it at any moment** with one key. Stopping always freezes the robot first.
5. **Every trial is recorded and human-scored** without the robot waiting for the score.

## Non-goals

- No automatic path-clearance or setup check. The **human** owns clearance and setup.
- No automatic success scoring. The human scores every trial, permanently.
- No placing or taking of the bottle by the robot. A human does both.
- No change to the model, training or server. The model is s08 `groot_handover261015_he_full`, served by lerobot's `sonic_policy_server.py`.
- No removal of the existing harness features:
  - planner-based ready return (StandingReset with desk clearance);
  - the VLM-gated `autopilot`.

  The new loop is an additional mode.

## Anti-goals (failure even if it "works")

- The robot moves after the operator pressed the stop key, before the operator's next Enter.
- The arm moves into the space over the case by any motion other than the ones lerobot validated:
  - the recorded raise;
  - the 1 s token blends to and from the ready pose;
  - the re-encoded lowering.
- The robot waits for a verdict between trials.
- A setting silently differs from lerobot's validated pipeline, for example the blend instead of anchoring, or several attempts per trial.
- A trial is lost without its record and its verdict slot.

## Operator interaction (decided with the user)

| Moment | Who | How |
|---|---|---|
| Deploy ready (deploy started, "Init Done") | Human | Enter |
| Robot in position (planner stance) | Human | Enter |
| Start the loop (setup done) | Human | Enter |
| Raise from the stance to the ready pose | Pipeline | Automatic |
| Start each trial | Pipeline | VLM sees the case empty, then a bottle standing upright with no hand at it |
| Return to the ready pose after any trial end | Pipeline | Automatic |
| Lowering at session end | Pipeline | Automatic |
| Path not clear, at any time | Human | Press **Enter** = stop |
| Scoring every trial | Human | Verdict entry that never blocks the robot |
| Bottle placement and taking it from the palm | Human | By hand |

**Stop (decided):**
1. Enter at any time freezes the robot where it is and ends the loop. No further trials start, and a running trial ends.
2. The next Enter means "path clear": the robot moves to the ready pose.
3. A further Enter restarts the loop. It waits for a bottle again; the raise is already done.

**Trial end without a stop (decided):** the robot goes back to the ready pose and waits for the next bottle. That covers a handover, a drop, the 30 s cap and a watchdog fault. The human scores the trial.

**Session end (assumed, not contradicted by the user):** after N trials or the operator's request:
1. lower along the recorded path to the stance;
2. hand back to planner mode.

## Settings and rules to match (from lerobot, by the numbers)

The reference is lerobot `examples/g1_dex3_training/` at a6e267c6: `stream_backends.py`, `sonic_token_stream.py`, `eval_session.py`, `sonic_policy_streamer.py`.

**Policy and stream**
- s08 at 50 Hz, chunk 40, replan every 0.4 s, the same request format (already in the native adapter).
- **Chunk anchoring, `--chunk-anchor-s 1.0`:**
  - A new chunk's 64 token dims are offset to start at the token **actually sent** (after the slew limit).
  - The offset decays as `exp(-(t - t_new) / 1.0)`.
  - The 14 hand dims keep the 0.3 s cross-fade.
  - Port `ChunkResampler.set_chunk(..., anchor_s, anchor_dims=64)` and `token_at`.
- Token slew limit 0.05 per 20 ms tick, on the token only.
- Hold-rest 1.5 s with the hands open before the first observation of each trial.
- The 50 Hz stream holds the current pose while the first chunk is computed, then a 1 s blend-in to the first chunk row.

**Episode end**
- One attempt per trial: `--end-on-handover 1.0`. Port `HandoverDetector` exactly:
  - signal: the mean of the measured right Dex3 joints;
  - closed above 0.19;
  - it fires once the hand was seen open, then closed for at least 0.5 s, then open for at least 1.0 s;
  - time it with `time.monotonic()`.
- 30 s cap per trial.

**Watchdogs (end the episode → normal trial end)**
- A measured arm joint faster than 6.0 rad/s, timed on the wall clock.
- An arm joint outside the training range by more than 0.15 rad (`ARM_LO`/`ARM_HI` in the streamer).
- No valid chunk for 1.0 s.
- Robot state older than 0.2 s.
- Camera frame older than 0.5 s.
- The deploy's `token_state` not echoing the sent tokens for 0.5 s.

**Poses and paths, in POSE mode, streamed as tokens**
- Ready pose, raise and lowering come from `ready_pose_261020_ep76.npz`, copied into this repo with its provenance. It holds:
  - `token`, `hands`, `state`;
  - `raise_tokens`/`raise_hands` (83 × 64/14);
  - `lower_tokens`/`lower_hands`.
- **Session start:**
  1. Planner → POSE handoff from the deploy's `token_state` (`--start planner`: the rest token is the planner token).
  2. A 1 s blend to the raise path's first frame.
  3. The recorded raise, 83 ticks.
  4. A 1 s blend to the ready pose.
- **Between trials:** stream the ready pose at 50 Hz (the deploy needs a token every 20 ms and must never be left without one in POSE mode).
- **Trial end → ready:** a 1 s token blend from the reached pose to the ready pose.
- **Session end:**
  1. A 1 s blend to the lowering's first frame.
  2. The lowering, 83 ticks.
  3. A 1 s blend to the stance (the planner token).
  4. The hand-back blend.
  5. PLANNER mode.
- **Freeze (stop):** keep streaming the last sent token and hands (lerobot's `hold_for_operator`/`holding`). No other motion.

## Constraints

- **Robot side:** PC2 (Jetson). It uses the fork deploy built at `~/GR00T-WholeBodyControl-harness-20261022`, started with `--harness-planner-hold`. The native harness venv is on PC2, and the s08 server is on workstation GPU 1, port 5571.
- **One clock:** the loop's orchestrator must run on the same host as the native loop. VoLoAgent compares native monotonic timestamps with its own (Stage 3 finding), so the orchestrator runs on PC2.
- **Camera:** the head-camera frames the harness gets must be the same images s08 saw in training: D435i colour 640×480, RGB, as in the s08 sessions. The VLM uses the same stream.
- **VLM: a local model on one RTX 3060 (user requirement, 2026-10-26).** The checks use **Qwen3-VL-2B-Instruct** in bf16, measured on workstation GPU 0 on 17 real head-camera frames from s08/s16/s13 (`docs/artifacts/g1_vlm_upright_check_20261026/`):
  - peak memory 4.35 GB, load 29 s, median 0.77 s per 640×480 frame;
  - the combined trigger "upright on the case AND no hand at the bottle" was 17/17;
  - "upright" alone was 14/17: it called bottles held in the robot hand upright, so the hand term is required;
  - "hand at bottle" was 16/17.

  The existing `HandoverMonitor` two-frame confirmation is kept. Its criteria are replaced in this mode by the two re-arm questions (empty case; upright bottle and no hand at it), asked as structured JSON fields. The hosted `gpt-5.6-sol` is no longer required for this loop.
  **Validation gap:** no real frame showed a bottle **lying on the case**. Before the loop runs unattended, real frames are captured at the robot (bottle lying in several spots, upright with a hand near it, empty case, a person reaching in) and the check is rerun with `upright_check.py`. Any miss on a lying bottle or a hand blocks the rollout.
- **Native ownership:** the existing harness rules stay in force: lease and heartbeat, freshness, `--harness-planner-hold`, fault latching, review flags in a session profile only.
- **The planner hold is not the hold primitive of this loop.** lerobot's holds are POSE-mode token holds. Where the harness today pauses into a planner hold, this loop holds the token instead (the rules above).

## Records and scoring (my calls: the user asked only that scoring never blocks the robot)

- Each trial writes the same artifacts as lerobot's `eval_session` (the 50 Hz log, the received head-camera JPEGs, an entry in `results.jsonl`) under a session directory. lerobot's tooling (`eval_session_to_lerobot.py`) then works on our sessions too.
- The verdict (success, fail or redo, plus a reason from lerobot's failure-reason list, plus a note) is entered in a **separate operator console** that never gates the robot. Trials without a verdict are listed at session end and stay "unscored" until scored.
- Each record also holds the trigger evidence: the VLM frames and decision for the bottle check, the end reason, and stop events.

## Decided by me (please check)

1. **New mode, not a replacement.** The loop is a new orchestrator command (for example `volo-g1-harness loop`) plus native RPC methods. The existing `autopilot` and planner-based ready return stay.
2. **Enter is read by the orchestrator on PC2.** The orchestrator's terminal is the operator's console for the startup gates and the stop. Verdict entry is a second console, so typing a verdict never looks like a stop.
3. **Startup gates are explicit RPCs** ("deploy ready", "robot in position"). They don't use the native loop's `k` toggle, whose second press stops control (a hazard seen on 2026-10-22).
4. **The ported pieces get parity tests against lerobot's own code**, as the adapter did (2026-10-21):
   - anchoring against `ChunkResampler`;
   - `HandoverDetector` against `eval_session.HandoverDetector`;
   - paths and blends against `SonicBackend`, on recorded inputs.
5. **Before the robot,** a simulation run of the full loop in the MuJoCo fixture, on a non-zero DDS domain (`--dds-domain`), with the rendered head camera and a scripted person.

6. **Failure taxonomy in loop mode: anything unexpected freezes; the planner is not used mid-loop.**

   | Event | Result |
   |---|---|
   | Handover detected; the 30 s cap; arm speed/range watchdog; no valid chunk for 1.0 s | Normal trial end: 1 s blend to ready, wait for the next bottle |
   | Robot state > 0.2 s old; token echo lost; telemetry stream restart or incompatible telemetry; camera > 0.5 s old; invalid policy result; stray native key | **Freeze** (stream the last sent token and hands), end the loop, wait for the operator as for a stop. Never blend while the state is unknown. |
   | Orchestrator gone or lease expired during the loop | **Freeze**, kept by the native side indefinitely. Release only from a restarted orchestrator: its Enter starts the stop sequence (clear → ready → restart). The deploy's `O` stays the hard stop. |
   | Native process dies | The deploy holds the last token (`--harness-planner-hold`); the operator handles it at the deploy (`O`). |

   A planner hold (POSE → PLANNER switch) happens only at the session end's hand-back, never as a failure response inside the loop.
7. **Stop during a scripted motion** (raise-start blend, raise, lowering, hand-back): freeze at the current frame, remembering the position along the path. The "clear" Enter **resumes the interrupted path** from that frame to its end (raise → ready pose; lowering → stance), because the recorded path is the validated route over the case and a direct blend from mid-path is not. After a resumed raise the robot is at ready, and the restart Enter starts the loop. After a resumed lowering the session has ended.
8. **Session end and counting:**
   - The operator ends the session from the **verdict console** ("end session"), not with Enter.
   - The lowering starts only from the ready pose: after a stop it needs the clear → ready sequence first.
   - N counts trials not marked "redo". A redo verdict, which arrives during or after the next trial, adds one trial to the remaining count.
   - Each trial's files are written when it ends. Its `results.jsonl` line is written when its verdict arrives (lerobot's format). At session end, unscored trials are listed, and resuming the session keeps counting from the trial files, not from the verdict lines.
9. **Staged rollout, because the recorded raise/lowering is only lightly proven on hardware:**
   - s19 (10/10) used a direct blend from the stance to ready, not the recorded raise.
   - The raise ran in s20 only (2 trials). The lowering ran about once, unlogged.

   Before the loop runs unattended, a supervised stage runs the loop with an operator Enter kept before the raise and before the lowering, and the raise/lowering are logged against the recorded path. The confirmations are then dropped, which needs the user's go-ahead.
10. **PC2 environment (confirmed on 2026-10-22):**
    - PC2 reaches api.genon.ai: the Stage 3 VLM calls answered from PC2.
    - The harness camera was the gear_sonic RealSense composed-camera server (`ego_view`, port 5555). Its colours and exposure were checked against s08 session frames (same channel order). This is a different server from lerobot's `g1_head_camera_server.py`, so the builder keeps that check (a side-by-side frame against an s08 frame) at each bring-up.

11. **The first trial after a loop start or restart skips the empty-case step.** The operator's start Enter confirms the setup, so a bottle already standing there (upright, no hand) starts the first trial. The empty-then-upright re-arm applies between trials.
12. **Where the VLM runs:** Qwen3-VL-2B is served on the workstation (GPU 0) behind an OpenAI-compatible HTTP endpoint, so the orchestrator on PC2 calls it like the hosted model (`--vlm-base-url`). The one-clock rule is unaffected: the VLM only returns answers, and frame timestamps stay on PC2. If GPU 0 lacks the 4.4 GB at start, the loop refuses to start and does not fall back silently.

## Left to the builder

- The RPC names and shapes between the orchestrator and the native harness, as long as the operator interaction and the anti-goals hold.
- How the loop modes sit in `harness_control.py` (new phases versus a separate controller), keeping the existing phases intact.
- The internal structure of the ported streamer pieces.
- The verdict console's form (a second terminal, a small TUI, a file).

## Known gaps carried over from lerobot (not in scope to fix)

- The stance → raise-start blend (about 1 rad) and the per-trial return to ready are direct token blends, not recorded motions.
- Handover and drop are not distinguished by the robot; the verdict decides.
- s08 was not trained on ready-pose starts, yet it was faster from there (s19).

## References

- lerobot-ed's pipeline description, 2026-10-26 (relayed in the voloagent-0f session); caisarl76/lerobot PRs #43, #46–#49; research doc `docs/research/2026-10-02-g1-wbt-handover-training.md`.
- Native adapter: GR00T-WholeBodyControl commits ad20fd7, b50613c. VoLoAgent registry mirror: 70a2408.
- Robot bring-up and Stage 1–3 findings: `docs/artifacts/g1_handover_20261022_robot_s08/`.
