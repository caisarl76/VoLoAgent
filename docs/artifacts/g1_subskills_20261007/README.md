# Bottle subskill preparation, 7 October 2026

**Updated 8 October:** the user confirmed that source 52's frames 0–1812 belong
to the previous episode's return to start, and source 78's placement failed
because the basket was knocked off the stool. Open the
[corrected review](../g1_subskills_20261008/review.html) and
[correction report](../g1_subskills_20261008/README.md). The original media below
remain historical references; source 78's later release is not placement success.

The harness can now finish a manipulation while preserving its measured hand
targets, then start another manipulation without a standing reset. The two
candidate prompts are `pick the bottle and hold it above the source table` and
`place the held bottle on the green stool`.

The current checkpoint still has the whole-task prompt
`pick drink bottle and place it on the right table`. The active workstation
profile has not changed. Neither candidate prompt has demonstrated policy
competence, and no new training has run.

## What changed

Each registered manipulation may specify `completion_action: hold`. After
vision reports completion, the coordinator pauses VLA, checks the new action
epoch, and waits for advancing native telemetry confirming the planner hold.
Hand targets stay at their measured positions. A following manipulation starts
only while that hold remains fresh. A busy inference worker may reject the
start with `NOT_READY`; retries stop at the policy deadline, including when
reading a new observation consumes that deadline. Ambiguous transport failures
do not cause another coordinator start attempt. Existing RPC transport retries
retain their request identity and native deduplication.

An omitted completion action still resets standing with open hands. For the
future bottle pair, picking uses `hold`; placement uses `reset_standing`. A
sequence ending at a confirmed hold releases its lease without cancelling that
successful skill or opening the hands. Request logs now include attempts that
the native client rejects.

The [design](../../superpowers/specs/2026-10-07-g1-subskills-design.md) defines the
scope. The IPC fixture uses a memory publisher and scripted vision decisions.
It proves lifecycle and hand-target behavior, not contact force, bottle
stability, learned manipulation, balance, or real robot readiness.

[Validation results](validation.json) record 655 passing VoLoAgent tests (six
skipped), 67 passing native harness checks, and the unchanged active profile
hash. The [saved IPC evidence](handoff-evidence/coordinator-events.jsonl) shows
the pick pause, rejected busy-worker start, placement start and final reset.

## Review the recordings

Open [the frame-stepping review page](review.html), or open these videos:

| Original source | Prepared episode | Material | Status |
|---|---:|---|---|
| 52 | 51 | [Original untrimmed reference, 55 s](source_000052_hold_candidate.mp4) | Frames 0–1812 now excluded; see the corrected 18.74 s copy |
| 78 | 77 | [Failure reference, first 46 s](source_000078_placement_candidate.mp4) | User-confirmed basket-fall placement failure around frames 1500–1600 |
| 18 | 17 | [Success reference, 40.48 s](source_000018_held_out_success.mp4) | Reserved whole source for future evaluation |
| 5 | 5 | [Failure reference, 19.54 s](source_000005_held_out_failure.mp4) | Reserved whole source for future evaluation |

All clips preserve the original 50 fps and start at source frame zero. Episode
78's original recording has 3,952 frames; this review copy stops at frame 2,300
exclusive. Original parquet hashes, video hashes, clip hashes and sampled-sheet
hashes are recorded in [review-clips.json](review-clips.json). There are 8,051
review frames across the four copies.

Contact sheets are sampled views. The continuous copies allow a reviewer to
inspect the intervals between those samples. Preparing or inspecting a sheet
does not accept a subskill label. Current accepted subskill segments: **zero**.
Sources 5 and 18 were already seen by the existing checkpoint; this reservation
only establishes a split for a future training run.

## Label the skills

[candidate-skills.json](candidate-skills.json) gives the entry and completion
conditions. The user confirmed that picking must stop immediately after lifting
above the source table. Source 52's late held-bottle views near the stool do not
qualify as that finish. If no existing demonstration stops above the source
for at least two seconds, collect that behavior explicitly.

For picking, identify an empty-hand start with a reachable bottle and a stable
held-bottle finish above the source table. For placement, start with that held
pose above the source and finish after release onto the stool. Require at least
100 consecutive stable terminal frames at 50 fps. Exclude setup, human resets,
unrelated objects and subsequent behavior. Keep failure/recovery demonstrations
as separate evaluation material unless their own training policy is defined.

Review the full original episode, choose `start_frame` and
`end_frame_exclusive`, and add a segment to its source entry in
[candidate-manifest.json](candidate-manifest.json). An accepted segment needs:

```json
{
  "skill_id": "pick_bottle_and_hold",
  "task_prompt": "pick the bottle and hold it above the source table",
  "start_frame": 0,
  "end_frame_exclusive": 0,
  "review_state": "pending",
  "outcome": "unknown",
  "reviewer": "",
  "entry_evidence": [],
  "terminal_evidence": [],
  "terminal_stable_frames": 0
}
```

These are placeholders, not suggested boundaries. Evidence should identify
the original recording, frame intervals, entry state and terminal state. Mark
success and acceptance only after continuous review. Assign eligible sources
to `train` as whole episodes; never move the reserved references into training.
Do not count two segments from the same recording as independent evaluation.

## Check the reviewed metadata

From the VoLoAgent worktree:

```bash
.venv/bin/python docs/artifacts/g1_subskills_20261007/subskill_dataset.py \
  docs/artifacts/g1_subskills_20261007/candidate-manifest.json \
  --catalogue docs/artifacts/g1_subskills_20261007/candidate-skills.json \
  --output /tmp/g1-reviewed-subskill-windows.json
```

The shipped manifest returns structured rejection and exit code 1. It produces
no output file and no training windows. It has 170 sources needing review,
two held-out references and two quarantined mixed/setup sources. Each new
skill needs accepted training and held-out segments. Windows contain all 40
action frames within one skill segment; source identity, content duplicates,
split contamination, overlapping cuts and malformed metadata are rejected.
The 8 October update additionally rejects excluded intervals and positive
segments for a known failed skill. Earlier pick segments remain separately
reviewable without relabeling the failed placement.

Passing this checker only prepares a window manifest. It trusts review
assertions and recorded hashes; it does not check label truth or export a
GR00T dataset. The exporter must independently verify source bytes and mapping,
slice synchronized data/video, apply the new task indices, preserve native
motion/hand channels, recompute metadata/statistics and consume only eligible
window starts. This export and a new training run remain pending.

## Next work

1. Accept exact segment cuts for the confirmed source-table handoff, or collect
   independent pick-and-hold and already-held placement demonstrations.
2. Export the reviewed dataset and train/evaluate a new multi-prompt policy.
   Evaluate each skill from its own entry state and evaluate the pair together.
3. Activate only prompts supported by that checkpoint. Add arm preparation,
   recovery or red-block skills after their own data and evaluation.
4. Continue the existing camera mounting/scene work and SONIC turning diagnosis
   before drawing conclusions from physical simulation or moving the G1.

Walking toward a table still needs SONIC plus perception/navigation. The
current bounded walking command does not identify or navigate to a table.
The deferred monitoring dashboard remains outside this session.
