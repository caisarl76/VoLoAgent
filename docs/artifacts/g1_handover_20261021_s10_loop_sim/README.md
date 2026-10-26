# s10 policy in the G1 handover loop simulation (2026-10-21/22)

SIMULATION FIXTURE ONLY. These runs do not qualify a checkpoint, a ready pose or a physical return.

This continues [`g1_handover_20261019_loop_sim`](../g1_handover_20261019_loop_sim/README.md). That loop replayed one recorded cycle in place of the policy. Here the policy is the real s10 checkpoint (`groot_handover261020_he_full`, 8/10 on the robot). It is served by lerobot's `sonic_policy_server.py` and driven by the native harness through its LeRobot client, from a rendered head camera.

| Part | What runs |
|---|---|
| Policy | s10, served on a workstation GPU (bf16 backbone, noise 0), with the robot sessions' 0.3 s chunk blend and 0.05 token step |
| Camera | The simulator renders the D435 head view (`d435i_head.json`: nominal intrinsics, URDF mount) and publishes it as `ego_view` |
| Autopilot, harness, ready return, SONIC | Production code |
| Vision | An oracle that answers each phase from the simulator's bottle state |
| Person | Scripted: restores the box after the startup return, takes a palm-up bottle, places a new one |
| Grasp | The bottle is a contact-free prop. It attaches when the right hand closes past 0.3 within 0.10 m of it. Opening below 0.2 releases it: onto the palm if the wrist roll is above 1.2 rad, otherwise it drops |

The case is 0.475 × 0.35 m with its top at 0.845 m. Its edge is 0.28 m from the pelvis, which is the measured 20 cm gap from the torso front. As in the 10-19 loop, the robot is pre-posed to the recorded standing start pose with the box away, and the box arrives after the autopilot's startup ready return.

## Checks before the loop

- **Policy path parity** (`parity_check.py`, `parity-trial0.json`, `parity-trial2.json`). Two trials of a real s10 robot session were run through both paths: the native client and lerobot's own streamer code.
  - The requests were byte-identical for all 20 frames per trial.
  - The 50 Hz output matched: tokens to within 1e-9 and hand targets exactly, over 1385 and 1500 ticks.
  - No GPU or robot is needed for this check.
- **Camera and scene** (`render_check.py`, `render-check.jpg`): the simulated head view side by side with a real s10 frame.
  - The geometry is similar.
  - The appearance is not. The simulated bottle is an opaque white cylinder, while the real one is clear PET with a yellow-green cap. The floor also differs.

## Runs (`runs/`)

| Run | Policy | Result |
|---|---|---|
| `plumbing-04` | `fake_policy_server.py`: one recorded cycle behind the server protocol | Wiring check. No falls. The pre-pose, the startup return and the box restore all completed. The server answered 53 requests with camera frames. The run ended at the 20 s manipulation deadline set for plumbing. |
| `s10-01` | s10 | The startup check, the startup ready return (raise first, settled) and `ready` all passed, then s10 took over. **It dropped the bottle in the first cycle**, and the autopilot interrupted on "sim oracle: bottle dropped". No falls. |

**What s10 did in `s10-01`** (`ground-truth.jsonl.gz`, sim time):
- The hand pulses instead of closing once. The right-hand closure peaks at 0.16, 0.19 and 0.22 between 16.0 and 18.6 s, reopening to about 0.05 each time.
- At 19.3 s it reaches 0.52, and the fixture attaches the bottle with the palm 6.1 cm from it.
- At 19.6 s it dips to 0.19 with the wrist palm-in (roll 0.40). That is a release that isn't palm-up, so the bottle drops.
- At 20.0 s the hand closes again to 0.56.

This matches lerobot's Isaac result for s10 (0/20). In simulation, s10 reaches near the bottle but not onto it, and it hesitates over the grasp. The autopilot responded correctly: it stopped on the drop.

**Provenance.**
- Both runs used native `cba799d` plus the then-uncommitted LeRobot client and the `--dds-domain` flag. They were made on 2026-10-22 between 13:00 and 13:05 KST.
- The client was committed afterwards as native `ad20fd7` (13:23), and `b50613c` (13:50) only skips the Isaac-GR00T import for LeRobot profiles. The runs were not repeated on those commits.
- The deploy binary is recorded by its sha256 in each `result.json`, and it equals the binary built from this PR's native commit.

The other `runs/*` files are copied from the run outputs:
- `result.json`, `events.jsonl`, `oracle.jsonl`, the simulator ready and result files, the simulation profile and the box restore;
- the mission result and events (gzipped);
- for the plumbing run, the fake server's request log.

## Running it

```
# s10 server (lerobot):
MODEL=/mnt/data/jihun/g1_models/wbt_groot_handover261020_he_full GPU=1 PORT=5561 \
  ~/work/lerobot/examples/g1_dex3_training/g1_groot_real_run.sh server
# loop (VoLoAgent worktree root):
PYTHONPATH=. .venv/bin/python docs/artifacts/g1_handover_20261021_s10_loop_sim/run_s10_loop.py \
  --policy-endpoint 127.0.0.1:5561 --cycles 3 --box-after-startup --gpu 1 --dds-domain 37 --output OUT
```

- **DDS domain.** Upstream SONIC's deploy is fixed to DDS domain 0. Any other G1 sim and deploy on `lo` (e.g. ASPIRE's `sim_rgbd.py`) then cross-feeds `rt/lowcmd`, and the robot twists and falls; that is what broke the 10-21 runs. `--dds-domain` passes a non-zero domain to both the simulator and the deploy. It needs the native deploy's `--dds-domain` flag; without the flag the deploy behaves as before, on domain 0.
- **Snap EGL.** A shell started from a snap terminal can carry `__EGL_VENDOR_LIBRARY_DIRS` and `/snap/...` entries in `LD_LIBRARY_PATH`. The headless renderer then fails with "libEGL fatal: did not find extension DRI_Mesa". Unset that variable and remove the `/snap/` entries.
- **GPU memory.** With s10 on a 12 GB GPU, about 2.4 GB is left. That is enough for SONIC (the launcher requires 2048 MiB).

## Open items

- **Grasp in simulation.** s10's pulsed closure and its 6 cm miss are a simulation gap (camera appearance, contact-free prop), not an autopilot problem. A more faithful bottle render and contact grasping would be needed before the sim can score s10.
- **One run only.** `s10-01` shows the failure mode; it is not a rate.
- **Hardware.** The robot test of the autopilot with a real policy is the next step. On the robot, s10 opens the hand palm-up in 8 of 10 trials.
