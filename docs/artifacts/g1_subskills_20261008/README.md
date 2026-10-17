# User corrections to bottle review, 8 October 2026

Source 52 includes the return to the starting position after the previous
episode. The user confirmed removing original frames **0–1812 inclusive**.
Source 78 contains a failed left-hand placement: lowering the bottle-holding
hand onto the basket pushed the basket off the stool and it fell to the ground,
around original frames **1500–1600**. The later bottle release does not make
that placement successful.

Open the [corrected review page](review.html). It shows original source frame
numbers alongside the rebased clip playback time.

| Recording | Correction | Result |
|---|---|---|
| Source 52, prepared episode 51 | Exclude `[0, 1813)` | [Trimmed video](source_000052_trimmed.mp4), original frames 1813–2749, 937 frames / 18.74 s |
| Source 78, prepared episode 77 | Failed `place_held_bottle_on_stool` | [Incident clip](source_000078_basket_failure.mp4), original frames 1450–1650, including one second of context on each side |

At 50 fps, the source 52 cut removes 36.26 seconds. The first trimmed frame is
original frame 1813. The incident bracket for source 78 is the user's
approximate annotation, not a frame-exact estimate of ground contact. Sampled
views show the hand/basket interaction; the failure label and ground-fall
description come from the user's review.

## Data and label status

[The retained-row extract](source_000052_retained_rows.parquet) physically removes
the first 1,813 rows from prepared episode 51. It preserves every remaining
column/value and the original frame, time, episode and global index values.
It is an unlabeled review extract, not a GR00T training dataset. Original
source/prepared datasets and the initial review media have not been modified.

The current [candidate manifest](../g1_subskills_20261007/candidate-manifest.json)
records the excluded interval and source 78's failed placement. The review gate
rejects positive segments overlapping exclusions and rejects positive windows
for a known failed skill, even if someone marks such a segment as accepted
success. It still permits an independently reviewed pick before a later
placement failure. No pick interval has been accepted from either recording.

The [candidate placement criteria](../g1_subskills_20261007/candidate-skills.json)
now require the basket and other objects on the destination to remain supported
and stable. This updates the future skill definition; the active workstation
profile and current checkpoint are unchanged.

**Accepted positive subskill segments and training windows remain zero.** The
cut and failure annotation alone do not establish an example of stopping with
the bottle held above the source table. Training has not started.

## Corrected interpretation of earlier results

Earlier sampled review of source 78 focused on release around 37.2–37.4 seconds
and the bottle resting on the stool afterward. That evidence missed the earlier
basket incident around 30–32 seconds. Source 78 must not be treated as a
successful placement reference on that basis. Previous vision scores describe
the old bottle-only criteria and do not validate this corrected whole-placement
label. Retain this recording as a failure case for future monitor evaluation.

Completion monitoring still needs validation over the incident history, so a
later acceptable-looking bottle pose cannot erase a basket-fall failure. No
new vision-model call or live-monitor deployment is claimed here.

## Reproduce and verify

From the VoLoAgent worktree:

```bash
/home/jihun/work/Isaac-GR00T/.venv/bin/python \
  docs/artifacts/g1_subskills_20261008/prepare_corrected_review.py

PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest -q \
  tests/test_g1_subskill_dataset.py
```

[Corrections](review-corrections.json) preserve the user's labels.
[Media metadata](corrected-media.json) records source/output hashes, frame maps,
counts and the retained parquet identity. The frame maps retain original source
coordinates for all 1,138 frames in the two new videos. [Validation](validation.json)
records the gate tests and independent media/data checks.

All 50 focused gate tests passed. An independent standards/specification review
found no confirmed defects and independently reran the same 50 tests successfully.
