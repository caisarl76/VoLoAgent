# Handover follow-up, 9–12 October 2026

The stationary handover loop is implemented and tested. Offline checks found
and corrected a native joint-order defect, located an existing handover
checkpoint, and exercised the vision monitor with real Genon responses.
The example profile remains disabled. Physical handover and robot clearance
are not established by these checks.

The desk height is **0.85 m**, with the person directly across it, as confirmed
by the user. Bottle distance from the robot, desk footprint and hand clearance
remain unspecified; the previous stool-placement geometry is not a calibrated
scene for this task.

## What changed

The ready-return code assumed contiguous left/right arm blocks. The actual
planner vector interleaves the arms; physical right-arm motors 22–28 occupy
planner slots `[4,6,8,10,12,14,16]`. The old assignment moved left wrists and
put a 1.7013 rad wrist-roll target in right wrist pitch, exceeding its XML
limit. The maximum target error was 1.7974 rad. The fix preserves all measured
upper-body targets, then changes the correct right slots, including limit
validation and tracking compensation. Named-motor regression tests and the
[before](palm-diagnostics/native-ready-mapping-check.before.json)/
[after](palm-diagnostics/native-ready-mapping-check.json) probes document it.

The monitor now retains up to three recent valid phase images. Taking-frame
evidence remains visible through both later empty-hand judgments. History
resets at phase entry and supplies context only; two distinct fresh yes
judgments are still required. Failure remains latched for the cycle.

## Existing training baseline

[Training audit](baseline-training-audit.json): the completed 20,000-step run
is on SSH host `h100` (`192.168.75.173`), under
`/mnt/data01/jhkim/gr00t_runs/handover_bottle_260930_subset_261002_n17_20k_gpu6_20261006`.
The candidate is `train/handover-bottle-261002-20k-gpu6/checkpoint-20000`.
Both saved exit codes are zero; the recorded average train loss is 0.07586.
Training loss does not establish task success.

Its exact trained prompt is **`pick drink bottle from the table and handover`**.
The disabled [profile](../../../configs/g1/handover.example.yaml) now uses it.
The baseline has 43 episodes and 18,970 frames, all assigned to training.
Its model files declare N1.7, 40-step actions, the G1 SONIC embodiment and the
pinned Cosmos revision `9ce19a195e423419c349abfc86fd07178b230561`.

| Original source episode | Filtered subset episode | Baseline use |
|---|---:|---|
| 15 | 10 | Training; user-confirmed successful reference |
| 30 | 19 | Training; user-confirmed successful reference |
| 16 | Excluded | User-confirmed roll/rescue failure |

Episodes 15/30 can check compatibility and reference behavior, but cannot
serve as unseen policy evaluation for this checkpoint. Source 16 remains
failure-monitor evidence, not a positive demonstration. New policy evaluation
needs genuinely unseen successful episodes and whole-episode splits before
40-frame windows are extracted.

[CPU metadata preflight](policy-metadata-preflight.json) verifies the selected
checkpoint shards, processor files, prompt and episode mapping in the existing
training container. It did not load weights or run inference. The user allocated
`h100_174` GPUs 4/5; the planned disconnected test uses GPU 4. Staging through
the workstation relay was too slow. At the user's request, both verified relay
processes were stopped and partial model files preserved for manual rsync.
[Transfer script](transfer_handover.sh) uses direct encrypted host-to-host copying
with temporary SSH agent forwarding. GPU inference has not run; completion
requires the manual transfer, its verification, then the disconnected test.

## Recorded vision results

The evaluator calls `openai/gpt-5.6-sol` at `https://api.genon.ai/v1` through
the real `HandoverMonitor`, with the production 10-second deadline and no robot
client. The API initially rejected `thinking_token_budget` as unsupported for
this model; removing it made the requests work. The diagnostic response is in
[vision-request-diagnostic/result.json](vision-request-diagnostic/result.json).
Credentials and request headers are excluded from the artifacts.

In the [13-case check](vision-compatible/result.json), all cases had available
decisions. There were three positive-label disagreements and no false-positive
frames within that case set. Episode 15's offer confirmed; episode 30 had an
initial unknown because palm support was obscured. Empty-hand-only images
remained unknown because they omitted the intervening taking action. Genon
identified episode 16's bottle roll as failure; the two subsequent rescue/tail
cases used the failure latch without new provider calls.

Real calls in that set took **2.94–5.36 s**. The sampled offer-to-contact
intervals are approximately 0.4 s in source 15 and 0.6 s in source 30. These
are sample brackets, not calibrated continuous-support measurements. The two
source-15 offering calls took about **9.47 s** in total. The offline evaluator
reads selected recorded timestamps while provider wall time elapses; its
confirmations do not demonstrate real-time pause before human removal.

In the [temporal replay before](transfer-before/result.json), the monitor
recognized taking/empty hand and reached complete at frame 415, then became
unknown at an extra frame-425 check after discarding the taking frame. The
[replay after](transfer-after/result.json) remained affirmative at 415 and 425.
This establishes retained context and one observed change in model judgments;
it does not show that the actual autopilot failed before the fix, because it
transitions at the first complete decision.

Frame 405 is still a review boundary: the conservative investigator label is
no/unknown, while Genon says the lifted bottle leaves the hand clear. This
disagreement is preserved in both reports. No collision-clearance verdict is
claimed. The historical post-fix summary counts two complete decisions as two
phases; there is one unique phase. The evaluator now counts phases separately.

## Review clips and palm diagnosis

[Cut manifest](review-cut-manifest.json) records zero-based source coordinates,
source/clip hashes and exact frame counts. The positive cuts are proposals;
**accepted new positive training cuts remain zero**. Original recordings and
labels are untouched.

| Clip | Source frame interval | Duration |
|---|---|---:|
| [15: pickup to offer](review-clips/source_000015_proposed_pick_to_offer.mp4) | [62,376) | 6.28 s |
| [30: pickup to offer](review-clips/source_000030_proposed_pick_to_offer.mp4) | [60,391) | 6.62 s |
| [16: roll and rescue](review-clips/source_000016_failure_roll_and_human_rescue.mp4) | [300,387) | 1.74 s |

[CPU FK diagnosis](palm-diagnostics/README.md) maps measured joints by name and
checks root orientation against recorded projected gravity across 1,240 frames.
Under the stated, uncalibrated palm-normal hypothesis, failed episode 16 frame
335 has about 9.62° inclination, compared with 4.01° from wrist roll alone.
Other joints and root orientation matter. Successful episode 15 also reaches
an inclined pose while fingers remain curled, so this is not a validated angle
threshold. Bottle placement, contact and release dynamics still need checking.

![Measured reference palm comparison](palm-diagnostics/palm-angle-comparison.png)

## Verification and next work

Fresh software verification: **709 VoLoAgent tests passed, 19 skipped**;
**68 focused native tests passed**; **13 explicit IPC integration tests passed**.
The native full suite's previously documented missing dependencies remain
outside these focused results. Two fresh reviews found no material runtime
defect in the mapping/history fixes; one evaluator counting issue was corrected.
The regressions were observed failing before their respective fixes.

The next acceptance work is:

1. Verify the disconnected checkpoint result, then evaluate behavior in a
   calibrated scene with genuinely unseen attempts. Action shape/finiteness
   is a compatibility check, not a manipulation-success metric.
2. Record real stable open-palm offers long enough for the measured verifier
   latency, with the person waiting before taking. A provisional collection
   target is 10–15 s of stable dwell, to be checked against observed latency
   tails. Do not extend the current clips by duplicating frames and call that
   a demonstrated stable hold. Reserve complete episodes for validation/test.
3. Resolve the frame-405 clearance label, calibrate the actual palm support
   surface, and measure bottle/desk positions. Validate bottle contact and the
   swept hand/forearm ready-return path in SONIC simulation at the 0.85 m desk.
4. Confirm the serving model and prompt, then perform supervised hardware
   acceptance before enabling either review flag. The deferred dashboard
   remains work for another session.

No physical robot command has been sent. The saved clips are recorded episodes;
the angle plot is measured-joint FK. Neither is a successful closed-loop VLA
simulation video.
