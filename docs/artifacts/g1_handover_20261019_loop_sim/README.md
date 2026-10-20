# G1 handover loop in simulation (2026-10-19)

SIMULATION FIXTURE ONLY. These runs do not qualify a checkpoint, a ready pose or a physical return.

The loop from `handover_bottle_261019` episodes 2–7 runs through the production harness and the VoLoAgent autopilot:
1. Start pose.
2. Ready pose.
3. A bottle stands on the box.
4. Pick it and offer it palm-up.
5. The person takes it.
6. Return to ready, and repeat.

The production parts are the native `run_vla_inference` main loop, the harness RPC, the ready return and the desk check, plus `HandoverRunner.run_autopilot` and `HandoverMonitor`. The stand-ins are:

| Part | Stand-in |
|---|---|
| Policy | One recorded cycle, replayed: episode 6, cycle 3, 6 s, 300 frames (`build_replay.py`). |
| Vision | An oracle that answers each phase from the simulator's bottle state. |
| Person | Scripted: restores the box, takes the offered bottle, places a new one. |
| Grasp | The bottle attaches when the right hand closes within 10 cm of it. |

Profile: `configs/g1/handover.example.yaml`, with these settings:
- Box top 0.845 m.
- Palm-in right ready pose.
- Left shoulder and elbow ready target; the left wrist is held.
- In the simulation copy only, `checkpoint_verified` and `ready_pose_reviewed` are set to true.

Every run starts from the recorded standing pose with the box 3 m away. The box arrives after the autopilot's startup ready return. Starting at the box is an open item; see below.

## Results (`trials.json`)

These runs use the native ready return with wrist relief and the deploy telemetry fix described below, at 3 cycles each.

| Box edge from pelvis | Runs | Cycles completed | Production ready returns | Box contact (real MuJoCo contacts) |
|---|---|---|---|---|
| 0.15 m | 1 | 3/3 | 4/4 settled | Left wrist: 140 samples, mostly while holding the offer; 19 during a return |
| 0.20 m | 3 | 9/9 | 12/12 settled | Left wrist in all three runs: 20, 27 and 2 samples, mostly while holding the offer; 2 during a return |
| 0.25 m | 1 | 3/3 | 4/4 settled | None |

Production ready returns are the autopilot's startup return plus one return per cycle. They exclude only the probe's fixture pre-pose to the start pose (tagged `fixture_start_pre_pose`), which runs without the lead guard and with a looser settling tolerance. Failed and interrupted production returns are kept.

Across all runs:
- Each grasp attached at 4–8 cm from the palm.
- Every offer released palm-up, at wrist roll 1.68–1.69 rad.
- No bottle was dropped.
- Every production startup return chose "raise first" (from the start pose the direct path is 12–18 cm into the box volume). Every return after a handover went "direct", because the hand starts above the box top.

`trials-before-wrist-relief.json` keeps the failing runs that motivated the native changes:
- 0.20 m: three runs ended in `reset_timeout`, in the startup return twice and in the cycle-2 return once (production returns 2/3, 0/1 and 0/1 settled).
- `smoke-11`: the autopilot aborted on a false planner hold.

## Native changes these runs depend on (GR00T-WholeBodyControl `work/g1-handover-loop-20261020`)

The runs used native base `25c948c` with these changes applied, before they were committed as `23bc312` (deploy telemetry) and `37a742a` (left arm and wrist relief). The deploy binary sha256 was `b1df417c…3ac2`. `trials.json` records the base commit only.

`4b4e666`, committed after these runs, applies the G1 limits to all 14 arm joints in StandingReset. Before that, only the right arm was bounded. Those bounds act only in StandingReset. Its stored references and paired measurements (5,540 pairs in the `trials.json` runs) stayed at least 0.325 rad inside the limits, so the bounds would not have clipped any return, and the runs were not repeated. Over a whole run, measured arm joints came within 0.265 rad of a limit: the right wrist roll during the palm-up offer, while the policy drives the arm and StandingReset is not running.

**1. Deploy `planner_reference_active` reported the wrong owner.**
- In token-only POSE mode (protocol v4), the planner is disabled but its last frame stays current, so the flag stayed 1.
- After `pause_manipulation`, the first new telemetry still carried that stale 1, and the harness confirmed its hold on it.
- About 80 ms later the planner reinitialized, the flag dropped to 0, and the autopilot aborted with "lost fresh control or planner hold".
- Fix: the flag is now 1 only while an initialized, enabled planner owns a valid frame.
- C++ test: `HarnessTelemetry.PlannerReferenceActiveOnlyWhileAnInitializedPlannerOwnsIt`.

**2. Ready-return wrist limits.** SONIC lagged wrist joints by more than the 0.15 rad reference lead:
- The right wrist roll after an offer stalled 0.72 rad from target. The reference was pinned at measured − 0.15, so the tracking trim never engaged.
- The held left wrist roll ended 0.07 rad off with the 0.15 rad trim saturated.

Fix, applied in ready returns only:
- Wrists get a 0.30 rad lead and trim.
- The held left wrist settles within 0.15 rad.

Shoulders, elbows, waist, fingers and the plain standing reset keep 0.15 rad lead and 0.05 rad tolerance. Tests are in `test_handover_ready.py`. They include a 0.2 rad steady wrist-lag regression and the held-wrist tolerance, and both fail with the old limits.

## Open items

- **Starting at the box.** From the recorded start pose, a box at 0.18–0.25 m is inside the raise-first path (smoke-06 and smoke-07 had real right-wrist contact). Stand up and return away from the box, as on hardware.
- **Left arm vs box.** The recorded offer posture brings the left wrist onto a box at ≤ 0.20 m, and occasionally during the return. The desk check (`ready_clearance`) models the right arm only.
- **Grasp fidelity is not tested.** The bottle is a fixed prop with no contacts until the hand closes, because the replayed reach sweeps sideways through it. The simulated pelvis turns 0.44 rad during the reach; the recording turned 0.20 rad. The bottle is placed where the replayed hand closes, measured from smoke-09 (`SIM_GRASP_M`), relative to the robot's current pelvis.
- **Clearance metric.** The shared `closest_clearance` (`mj_geomDistance`, MuJoCo 3.8.1) returns 0.0 for G1 meshes against the box when nothing touches. These results count real contacts instead.
- **Hardware.** A real policy, real vision and supervised hardware acceptance are still pending. Wrist tracking on hardware may differ from SONIC in simulation.

## Reproduce

Run from the VoLoAgent worktree with GPU 1 free (≥ 2 GiB):

```
env PYTHONPATH=. .venv/bin/python docs/artifacts/g1_handover_20261019_loop_sim/run_loop_probe.py \
    --box-front-m 0.20 --cycles 3 --box-after-startup --gpu 1 --output <new dir>
cd docs/artifacts/g1_handover_20261012_acceptance && <sim python> ../g1_handover_20261019_loop_sim/summarize_trials.py trials.json <run dirs>
```
