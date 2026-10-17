# Online pelvis height and tilt for the ready clearance check, 15 October 2026

The robot's standing pelvis is about 74–76 cm and moves with SONIC's knee bend.
In simulation it stands at 0.787 m. A 1–3 cm change flips the ready-return
decision for the recorded offers, so a fixed number in the profile is not
enough. Native `ddd6008` estimates the pelvis height and tilt from the measured
state at each decision. The profile now gives the desk top above the floor
(`desk_top_above_floor_m: 0.85`).

**The live estimate was within 0.4–1.1 mm of the simulator's pelvis height in all
9 completed runs.**
- **Measured desk** (0.85 m top, 0.15 m edge): both offers returned directly,
  6/6, with no contact.
- **Desk raised to 0.887 m:** this is how the 0.85 m desk looks from a 0.75 m
  pelvis. Episode 30 still returned directly, 3/3. It is a near fit: one run
  recorded one wrist-contact sample.
- No hardware ran. Both review flags remain false.

## What changed in the check (`ready_clearance.py`)

1. **Pelvis height:** the lower ankle-roll frame's depth below the pelvis, plus
   the 0.035 m ankle-to-sole offset.
   - Uses the measured legs, gravity-aligned with `base_quat`.
   - The offset comes from the foot collision box in `g1_29dof_with_hand.xml`;
     the sim data fit gave 0.0344 m.
   - It keeps the raise when the ankles differ by more than 2 cm, meaning a foot
     is off the floor, or when the estimate is outside 0.65–0.85 m.
2. **Pelvis tilt:** the arm boxes are placed in the same level frame (z against
   gravity, x along the pelvis heading). At 0.45 m reach, the 4° pelvis pitch
   seen in simulation moves the hand about 3 cm.
3. **No URDF clipping of measured joints.** The ready pose and recorded entries
   hold the right shoulder roll past its URDF limit. The old clipped FK shifted
   the arm boxes by 21–64 mm.
4. **Logged reason:** the hook logs the decision with its reason, the pelvis
   estimate and the level-frame desk top, for example
   `[harness] ready return: direct (hand above desk top (pelvis 0.787 m, desk top +0.063 m))`.

New native tests (26 in `test_ready_clearance.py`) cover:
- the estimate against MuJoCo sole contact (within 1 cm, upright and 3° pitch,
  two knee bends);
- the sole offset against the foot box;
- a 6° forward pitch flipping episode 15 to a raise;
- a lowered pelvis flipping it to a raise;
- a lifted foot, a deep crouch and a bad `base_quat` all keeping the raise.

## Trials

Same loopback setup as before:
- desk front at 0.15 m;
- native `ddd6008`, binary `347044b5…` (the change is Python only);
- the production profile.

`--desk-height-m` is a new optional argument of the acceptance probe (default
0.85 m). For the raised desk the profile is
[`candidate-desk-0887.yaml`](candidate-desk-0887.yaml).
[`trials.json`](trials.json) holds per-run details, written by [`summarize.py`](summarize.py).

| Desk top (sim) | Entry | Decision (logged reason) | Completed | Pelvis estimate vs true | Min conservative gap | Desk contacts |
| ---: | --- | --- | ---: | --- | ---: | ---: |
| 0.85 m | ep15 offer | direct ×3 (hand above desk top, +0.062 m) | 3/3 | 0.787–0.788 vs 0.7865–0.7876 m | 0.1–5.6 mm | 0 |
| 0.85 m | ep30 offer | direct ×3 (hand above desk top, +0.062 m) | 3/3 | 0.787–0.788 vs 0.7862–0.7872 m | 24–25 mm | 0 |
| 0.887 m | ep30 offer | direct ×3 (hand above desk top, +0.100 m) | 3/3 | 0.786–0.787 vs 0.7851–0.7868 m | 0 (boxes touch) | 0, 0, 1 |
| 0.887 m | ep15 offer | — | not run (fixture pre-pose failed) | — | — | — |

- **Desk and bottle:** no robot–bottle contact in any run, and the feet stayed
  on the floor.
- **`low-e30-03`:** the one contact sample is the right wrist-roll link, 0.375 s
  into the return. In this emulation, episode 30's lowest hand point is 0–3 mm
  above the desk top. It fits, but barely.
- **`low-e15-01`:** the fixture's pre-pose never ran from a settled stance.
  - At the 0.887 m desk the simulated standing hands spawn just under the
    desktop.
  - The robot stumbled: its true pelvis dropped to 0.63 m at 1 s and settled
    near 0.72 m.
  - The pre-pose decision logged a 0.743 m pelvis, tracking that drop, then the
    pre-pose timed out.
  - Recorded episode 15's hand would sit 3 cm below this desk top anyway, so the
    entry does not fit this geometry.

## What this means for the robot

- **Pelvis measurement:** none is needed. The decision uses whatever pelvis
  height the legs show at that moment. A one-time tape-measure check of the
  logged estimate is still worth doing; it tests the model and the 0.035 m sole
  offset.
- **Low pelvis:** at a 74–76 cm pelvis the 85 cm desk top sits 9–11 cm above it.
  An offer must end with the hand clearly above that to get the direct return.
  The two recorded offers barely do, or don't, at that height.
- **Unchanged:** the robot must still stand up and reset away from the desk.

## Reproduce

From the VoLoAgent worktree, with GPU 1 checked for 2 GiB free:

```bash
env PYTHONPATH=. .venv/bin/python docs/artifacts/g1_handover_20261014_post_handover_entry/run_entry_probe.py --entry source30_f420 --return profile --gpu 1 --desk-front-m 0.15 --output /tmp/new-dir
env PYTHONPATH=. .venv/bin/python docs/artifacts/g1_handover_20261014_post_handover_entry/run_entry_probe.py --entry source30_f420 --return profile --gpu 1 --desk-front-m 0.15 --desk-height-m 0.887 --candidate-profile docs/artifacts/g1_handover_20261015_pelvis_estimate/candidate-desk-0887.yaml --output /tmp/new-dir2
/home/jihun/work/VoLoAgent/.worktrees/g1-harness-native/.venv/bin/python docs/artifacts/g1_handover_20261015_pelvis_estimate/summarize.py
```

Earlier evidence folders keep their as-run profiles, which use the old
`desk_top_above_pelvis_m` field; the current parser rejects them.
