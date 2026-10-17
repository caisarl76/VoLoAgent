# Ready return from post-handover entry poses, 14 October 2026

Follows the [waypoint screen](../g1_handover_20261014_low_load_waypoint/README.md),
which found no raised waypoint that works from grounded standing.

**From actual end-of-offer poses no raise is needed, and the current raise is
harmful.** The native ready return now skips the raise when forward kinematics
shows the direct path keeping ≥3 cm from a configured desk. Starting from the
recorded offers, it chose the direct return 6/6 and completed 6/6. From grounded
standing it still raises. That raise path is unchanged and still stalls.
No hardware ran, and both review flags remain false.

## Entry poses and screen

The entries are the last frames of the user-confirmed handovers: episode 15,
frame 431 and episode 30, frame 420. They include the waist, both arms and both
hands. `screen.py ENTRY` in the waypoint folder reruns the static screen from
each entry. Its path metric now counts only yaw motion that gravity opposes.

| Entry | Direct return: opposed yaw sag | Direct return: desk gap | Current waypoint: desk gap |
| --- | ---: | ---: | ---: |
| Grounded standing | 0.025 rad | 0 mm (overlap) | 49.5 mm |
| Episode 15 offer | 0.000 | 51.2 mm | 0 mm (overlap) |
| Episode 30 offer | 0.000 (0.046 assisted) | 66.8 mm | 0 mm (overlap) |

The offer pose clears the desk by sitting just behind its near edge. Its
lowest point is only ~1 cm above desk-top height, so a height-only rule cannot
recognise it. The check therefore uses the desk as a box: top height and near edge.

## Simulation fixture

[`run_entry_probe.py`](run_entry_probe.py) drives the existing loopback probe
twice through `reset_ready`. Both the SONIC binary and the policy are unchanged.

1. **Fixture pre-pose.** [`entry_worker.py`](entry_worker.py) moves the waist
   and both arms to the recorded entry at 0.5 rad/s with tracking trim.
   - It has no lead guard: a left-wrist transient tripped it when 15 joints
     started together.
   - It settles within 0.12 rad, because SONIC leaves the recorded left arm
     about 0.1 rad off.
   - The right hand opens, as on the offered palm.
2. **Desk handling.** [`sim_entry_worker.py`](sim_entry_worker.py) keeps the
   desk and bottle 3 m away during the pre-pose. It restores them only when the
   conservative hand/forearm-to-desk gap is positive (59.6–80.6 mm across all 17 runs).
3. **The return under test.** This step runs the production StandingReset with
   the production 0.05 rad tolerance, lead guard, dwells and deadline.

## Results

[`trials.json`](trials.json) (`summarize.py`) gives per-run details. Desk
evidence comes from saved physics after the desk was restored:
- conservative box gap;
- MuJoCo robot–desk contacts recomputed from saved states (sampled at 20 Hz,
  so brief touches between samples can be missed);
- foot contact.

**Old native code (`16ba4204…`): direct return emulated vs. current waypoint.**
The emulation passes the measured arm as the waypoint, so stage 0 only dwells.

| Return | Entry | Completed | Min desk gap | Robot–desk contact samples |
| --- | --- | ---: | ---: | ---: |
| Direct (emulated) | ep15 | 3/3 | 61.6–70.6 mm | 0, 0, 0 |
| Direct (emulated) | ep30 | 3/3 | 49.3–52.1 mm | 0, 0, 0 |
| Current waypoint | ep15 | 0/2 | 0 mm | 59, 53 |
| Current waypoint | ep30 | 2/3 | 0 mm | 0, 2, 1 |

- **ep15 waypoint runs:** both put the index/middle fingers on the desktop
  ~0.4 s into the raise. With the fingers pinned, wrist yaw could not follow
  its reference, and the lead guard correctly interrupted.
- **ep30 waypoint runs:** the fingertip came within 0.2 mm of the desk edge
  even in the runs with no recorded contact.
- **Excluded:** `INVALID-ep15-profile-03` ran after the template edit. The new
  hook raised the fixture's own pre-pose, which then never settled. The
  fixture now forces its pre-pose target.

**New native code (`6987b44d…`): the production hook decides.**

| Entry | Decision logged | Completed | Return time | Min desk gap | Contacts | Feet on floor |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| ep15 offer | direct ×3 | 3/3 | 2.7–3.1 s | 52.3–60.4 mm | 0 | all samples |
| ep30 offer | direct ×3 | 3/3 | 2.4–2.7 s | 58.9–61.0 mm | 0 | all samples |
| Grounded standing | raise first ×2 | 1/2 (one `reset_timeout`) | — | — | — | — |

Preserved waist/left arm stayed within 0.05–0.08 rad during motion, and every
offer-entry run passed the production settling check. From grounded standing the
hook kept the raise, which behaved as before (old code: 1/3).

**After merging `main` (native `9329b16`, rebuilt binary `65ce96a7…`).**
These runs repeat the production-hook runs with the new SONIC mode handoff and Dex3 feedback-validity flags.
They are the `merge-prod-*` trials.

| Entry | Decision logged | Completed | Return time | Min desk gap | Contacts | Feet on floor |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| ep15 offer | direct ×3 | 3/3 | 1.3–2.8 s | 54.7–56.2 mm | 0 | all samples |
| ep30 offer | direct ×3 | 3/3 | 2.7–3.1 s | 53.5–65.1 mm | 0 | all samples |
| Grounded standing | raise first ×3 | 1/3 | — | — | — | — |

- **Offer entries:** results match the pre-merge runs. Preserved waist/left arm stayed within 0.044–0.075 rad.
- **Grounded standing:** the two failed runs end at `reset_reference_lead_exceeded` and `reset_timeout`.
  Both fail during the raise, with right shoulder yaw (motor 24) furthest off (0.19 and 0.26 rad).
  This is the gravity-load stall diagnosed before the merge, not a merge effect.

## Production change

- **Native repository:** adds `gear_sonic/utils/inference/ready_clearance.py`.
  It uses the native pinocchio model in its own instance, because forward
  kinematics overwrites the cache that inference uses. It samples the exact
  StandingReset direct reference: right arm to target, right hand opening,
  0.5 rad/s per joint, everything else held.
- **What is checked:** world boxes around the collision boxes of the right
  elbow, wrist and hand links. A test regenerates those boxes from
  `g1_29dof_with_hand.xml`. Pinocchio frames matched MuJoCo bodies within 1.5 µm.
- **Decision:**
  - Skip the raise only if the gap stays ≥3 cm.
  - Without the two optional desk fields, or with invalid feedback, keep the raise.
  - At startup, the hook logs `[harness] ready return: direct|raise first`.
- **Profile fields:** `desk_top_above_pelvis_m` and `desk_front_from_pelvis_m`
  must appear together and be finite. VoLoAgent's registry mirrors the parser
  byte-for-byte.
- **Example template:** `configs/g1/handover.example.yaml` sets 0.063/0.40.
  These come from the 85 cm desk, the 0.787 m simulated standing pelvis and the
  provisional 40 cm edge, and must be measured before hardware use.

FK lower bounds for the direct path, against the 3 cm margin:

| Entry | Bound | Decision |
| --- | ---: | --- |
| Episode 15 | 40.7 mm | direct |
| Episode 30 | 52.4 mm | direct |
| Grounded standing | −34.5 mm | raise |

**Verification:**
- 690 native tests passed, 16 of them new.
- 751 VoLoAgent tests passed with 19 skipped, 6 of them new.
- 13/13 explicit cross-process tests passed.

## Still open

- **Real desk pose and pelvis height:** must be measured. Torso tilt is ignored
  (neutral pelvis frame). The check assumes ideal tracking, although measured
  gaps were 52–61 mm against a 41–52 mm bound.
- **Low starts:** the raise from grounded standing still stalls (right-yaw
  gravity load and left-arm drift). That matters only if a real return starts
  with the hand below the desk. Follow-up: opt-in yaw gravity feedforward via
  `tau` completed it 5/5 in simulation ([yaw feedforward](../g1_handover_20261015_yaw_feedforward/README.md)).
- **Entries tested:** only the two recorded entries ran. Real post-handover
  poses come from VLA rollouts, so the full connected VLA cycle, swept or
  person clearance and supervised hardware acceptance remain untested.
