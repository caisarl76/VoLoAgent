# Ready return at the measured desk (0.15 m edge), 15 October 2026

The desk was measured at 84–85 cm high, with the near edge 15–20 cm in front of
the pelvis center. These trials use the conservative 0.15 m edge and the
example profile's 0.063 m desk top (with the simulator's 0.787 m pelvis).

**At this desk the raise is the problem and the direct return is not.**
- **Production code:** the clearance check chooses the raise for both recorded
  offers. The raise touched the desk in 7 of 7 offer runs from the start of the
  return (in 6 of them the contact began within 0.6 s), and completed 5/7.
- **Direct return:** the fixture-emulated direct return completed 3/3 with no
  desk contact.
- **Grounded standing:** this is not a valid start at this desk. The simulated
  robot spawns with both hands resting on the desktop.

No hardware ran. Both review flags remain false.

## Setup

- **Code:** native `f356332`, binary `347044b5…`.
- **Physics:** loopback, grounded physics, zero-speed planner.
- **Scene:** desk front `--desk-front-m 0.15`, desk top at 0.85 m. The bottle
  stands 0.15 m past the edge.
- **Profiles:**
  - Production return: `configs/g1/handover.example.yaml` (edge 0.15 m).
  - Feedforward runs: [`candidate-feedforward.yaml`](candidate-feedforward.yaml),
    the same profile plus the yaw feedforward.
  - Direct emulation: [`candidate-direct-emulation.yaml`](candidate-direct-emulation.yaml),
    the same profile without desk fields. With desk fields, the production hook
    would replace the fixture's direct waypoint.
- **Contact evidence:** [`summarize.py`](summarize.py) recomputes robot–desk and
  robot–bottle contacts from saved physics, sampled at 20 Hz. Offer runs are
  counted from the desk restore, standing runs from startup. Per-run detail is
  in [`trials.json`](trials.json).

## Results

| Entry | Return | Completed | Desk contact (samples per run) | Contacting links |
| --- | --- | ---: | --- | --- |
| ep15 offer | production (raise first) | 1/1 | 28 | right middle finger, wrist yaw |
| ep30 offer | production (raise first) | 1/3 (2 `reset_timeout`) | 2, 2, 2 | right middle finger |
| ep30 offer | production + yaw feedforward | 3/3 | 3, 0, 3 | right middle finger |
| ep15 offer | direct (emulated) | 1/1 (+1 fixture pre-pose timeout) | 0 | — |
| ep30 offer | direct (emulated) | 2/2 | 0 | — |
| Grounded standing | production | 0/3 (`reset_reference_lead_exceeded`) | 199–236, from 0.03 s | both hands and wrists |
| Grounded standing | production + feedforward | 0/3 (`reset_timeout`) | 681–849, from 0.02 s | both hands and wrists |

- **Offer runs with the raise:**
  - The first contact is 0.37–0.6 s into the raise, as at the 0.40 m edge before.
  - With feedforward, the raise completed every time (dwell reached at
    2.6–3.4 s), but fingers still brushed the desk in 2 of 3 runs.
  - Without it, ep30 timed out 2/3 times, the known right-yaw stall.
- **Direct returns:**
  - They took 1.9–5.2 s.
  - Conservative clearance stayed at 13–18 mm.
  - The preserved waist and left arm stayed within 0.054–0.077 rad.
- **Bottle:** untouched in every offer run.
- **The ready pose itself:** the right wrist ends about 0.2 m forward and
  0.13–0.17 m above the pelvis, just above the desktop. The conservative
  clearance at the end of a return was 6–30 mm, with the right elbow closest.
- **Grounded standing:**
  - At startup the simulated standing pose puts both wrists about 0.2 m forward
    at 0.885 m height. That is on top of an 85 cm desk whose edge is 0.15 m away.
  - The desk pushes the robot back about 13 cm and the hands shove the bottle;
    one run shows the bottle against the right ankle.
  - These runs show the stand-up pose does not fit this desk. They say nothing
    about the raise.
- **Lost run:** `e15-direct-01` was lost to the fixture: its pre-pose timed out
  before the return under test.

## What this means for hardware

1. **The raise is unsafe at this desk.** From offer poses the production check
   chooses it, and it lowers the fingers onto the desk. It was meant to lift a
   hand that starts below the desk edge.
2. **The direct return clears, but narrowly.** It had 13–18 mm of conservative
   clearance in simulation, and the 3 cm FK check rejects it. The check boxes
   are lower bounds, so the true clearance is larger.
3. **Standing up in front of the desk is unsafe.** The standing hands overlap
   the desktop, so the robot should not power up, stand or be reset there. It
   needs to stand farther back.

## Decisions before the first hardware ready return (not taken)

- **Raise rule:** at this desk the raise should not be chosen from above-desk
  starts. Possible changes:
  - Skip it whenever the hand starts above the desk top.
  - Use exact geometry instead of boxes.
  - Lower the 3 cm margin.

  Each changes reviewed behaviour.
- **Desk placement:** decide where the robot stands and gets reset, and keep that
  out of the desk.
- **Real pelvis height:** confirm it on the robot. The ready-pose clearance is
  only centimetres.

## Reproduce

From the VoLoAgent worktree, with GPU 1 checked for 2 GiB free:

```bash
env PYTHONPATH=. .venv/bin/python docs/artifacts/g1_handover_20261014_post_handover_entry/run_entry_probe.py --entry source30_f420 --return profile --gpu 1 --desk-front-m 0.15 [--candidate-profile docs/artifacts/g1_handover_20261015_desk_edge_015/candidate-feedforward.yaml] --output /tmp/new-dir
env PYTHONPATH=. .venv/bin/python docs/artifacts/g1_handover_20261014_post_handover_entry/run_entry_probe.py --entry source30_f420 --return direct --gpu 1 --desk-front-m 0.15 --candidate-profile docs/artifacts/g1_handover_20261015_desk_edge_015/candidate-direct-emulation.yaml --output /tmp/new-dir2
env PYTHONPATH=. .venv/bin/python docs/artifacts/g1_handover_20261014_torque_probe/run_torque_probe.py --gpu 1 --desk-front-m 0.15 --output /tmp/new-dir3
/home/jihun/work/VoLoAgent/.worktrees/g1-harness-native/.venv/bin/python docs/artifacts/g1_handover_20261015_desk_edge_015/summarize.py
```
