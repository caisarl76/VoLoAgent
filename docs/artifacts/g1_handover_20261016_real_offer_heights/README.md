# Real-robot offer end heights against the measured desk, 16 October 2026

The ready return skips its raise when the right wrist and hand start above the
desk top (native `ready_clearance.py`). The simulation evidence used two recorded
offers. This checks where real-robot VLA handover episodes end, from the s07
and s08 evaluation sessions of 2026-10-16.

**The newer handover model (s08) ends most offers 2–8 cm above the desk-top
line, even at a 0.74 m pelvis. The older model (s07) mostly ends within a few
centimetres of it, so at the robot's 74–76 cm pelvis several returns would raise
first.** At this desk, raising first is what touched the desk in simulation.

## Data

The evaluation drafts are read-only, under
`/mnt/data/jihun/g1_eval_sessions/_visualizer/workspace/drafts/`. These are the
viewer's `local/annotation-<sha256(path)[:16]>` datasets.

| Session | Draft | Model | Episodes | Success (human-reviewed) | Failures |
| --- | --- | --- | ---: | ---: | --- |
| s07 | `0f94fe9a…` | `wbt_groot_he_full` | 5 | 4/5 | ep3: grasp missed (frame 104); the bottle toppled |
| s08 | `2ee431e2…` | `wbt_groot_handover261015_he_full` | 10 | 9/10 | ep6: stopped beside the bottle without contact (frame 88) |

The operator verdicts and reviewed labels agree on every episode. All episodes
end with an operator stop.

## Check

[`check_offer_heights.py`](check_offer_heights.py) writes [`results.json`](results.json).
It takes each episode's last frame, which comes after the hand-off at the
operator stop. It places the right wrist and hand collision boxes with the same
forward kinematics as the native check, without URDF clipping. It then reports
the lowest wrist or hand point against an 0.85 m desk top for three pelvis
heights.

Hand minus desk top (positive means above, so the return goes straight back):

| Session / episode | Verdict | Pelvis 0.74 m | Pelvis 0.76 m | Pelvis 0.787 m (sim) |
| --- | --- | ---: | ---: | ---: |
| s07 ep0 | success | −36 mm | −16 mm | +11 mm |
| s07 ep1 | success | −16 mm | +4 mm | +31 mm |
| s07 ep2 | success | −11 mm | +9 mm | +36 mm |
| s07 ep3 | fail | −18 mm | +2 mm | +29 mm |
| s07 ep4 | success | +15 mm | +35 mm | +62 mm |
| s08 ep0 | success | +17 mm | +37 mm | +64 mm |
| s08 ep1 | success | +2 mm | +22 mm | +49 mm |
| s08 ep2 | success | −89 mm | −69 mm | −42 mm |
| s08 ep3–ep9 | 6 success, ep6 fail | +14 to +36 mm | +34 to +56 mm | +61 to +83 mm |

Above the desk top:

| Pelvis | s07 | s08 |
| --- | ---: | ---: |
| 0.74 m | 1/5 | 9/10 |
| 0.76 m | 4/5 | 9/10 |
| 0.787 m | 5/5 | 9/10 |

## Limits

- **Last frame as the offer's end:** the last frame is the operator stop,
  after the person took the bottle. s08 ep2's low hand is probably the arm
  already lowering.
- **Pelvis assumed:** the recorded state has no legs, waist or `base_quat`, so
  the pelvis is assumed upright and the waist zero. On the robot the native
  check estimates both from the legs and IMU; a forward pelvis pitch of a few
  degrees lowers the hand by centimetres.
- **One test only:** only the hand-above-desk-top test is applied. Hands below
  the line fall back to the 3 cm forward-kinematics path check, which this does
  not evaluate.
- **Not a return test:** these episodes did not run the harness ready return.
  The robot log's `[harness] ready return:` line gives the real decision.
