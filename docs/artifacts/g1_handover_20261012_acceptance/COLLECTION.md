# New handover recording and review checklist

Use the existing instruction: `pick drink bottle from the table and handover`.
The current checkpoint has not been trained on separate prompts for approach,
pick, offer or ready return. This mission keeps the G1 stationary and uses native
ready return after the person removes the bottle.

For each new operator-controlled demonstration:

- Start with the right hand visibly empty in the intended ready pose. The bottle
  rests on the 85 cm desk within reach, and the person waits across the desk.
- Pick the bottle, lift it clear, turn the palm upward and open the fingers.
  Keep the bottle supported without rolling, slipping or help from the person.
- Hold that visible offer for 10–15 seconds initially. This is a provisional
  recording target based on the observed vision-call latency; monitor acceptance
  still requires two distinct fresh affirmations. Existing successful recordings
  contain much shorter offer-to-person-contact intervals.
- Let the person take the bottle, then move their hand out of the camera's view.
  Keep the robot hand still and clearly empty long enough to evaluate two fresh
  frames. During software/hardware testing, wait for the monitor's empty-hand
  confirmation before a native return.
- Finish in the empty-hand ready pose with at least 0.5 seconds of measured
  settling. Preserve the full recording, including waiting and removal.

Mark the frame intervals for bottle visible, first secure grasp, first stable
open-palm support, human contact/removal and first clearly empty hand. Record
uncertain boundaries as brackets. A closed finger command alone does not label
a secure grasp, and a single still image does not label stable support.

Record failures separately: rolling off the palm, falling, human rescue,
incomplete grasp, early withdrawal and occlusion. Do not stage dangerous drops
to obtain negative examples. Source 16 is the existing confirmed roll/rescue
negative; later empty-hand frames do not turn it into success.

Before a new training run, reserve whole new collection sessions for validation
and test and keep them out of training. Every crop or window from an episode
inherits its episode/session split. Preserve source episode IDs and hashes.
Sources 15/30 remain training references; the 18 episodes excluded from the
baseline are not automatically held-out successes. Review their outcomes before
assigning any evaluation role.

Keep pick-to-stable-offer training cuts separate from monitor evaluation of the
long offer/removal wait. Review the frame boundaries and weighting before
training, so long stationary waits do not silently dominate the action samples.
The previously proposed cuts remain unaccepted.

For the simulation fixture, use the grasp-derived hand positions as the first
reach estimate. Measure or annotate where the bottle sits relative to the palm
and desk edge. Height, diameter and fill weight are useful if available. The
earlier 20 cm / 8 cm / 300 g bottle values remain assumptions for this handover.
Camera/scene calibration is still needed for metric bottle contact and human
clearance, rather than for basic VLA loading.
