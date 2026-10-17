# Measured hand classifier — 2026-10-13

Investigation started October 13; final verification completed October 14.

The handover harness now uses measured right Dex3 finger joints to classify
`open`, `closed`, `partial`, or `unknown`. It supplies the measured posture to
the VLM only during `pick_offer`. Every live autopilot entry point requires
distinct, aligned open-hand samples alongside the existing two affirmative
vision judgments. Opening the hand never proves bottle support, an empty
hand, or successful removal.

## Rules and order

The input is measured radians in native order:
`thumb0, thumb1, thumb2, index0, index1, middle0, middle1`.
The recorded 43-joint data is mapped by exact names, using columns
`40, 41, 42, 36, 37, 38, 39`; no contiguous source slice is treated as native
hand order.

| State | Rule |
| --- | --- |
| open | All seven angles within ±0.05 rad of zero; preserves the existing native open tolerance |
| closed | Four index/middle angles ≥0.35 rad, thumb1 and thumb2 ≤−0.25 rad |
| partial | Valid angles that satisfy neither pose rule |
| unknown | Invalid/missing angles, invalid feedback timing, missing clock metadata or another clock domain |

Angles must also lie within the native G1 XML joint bounds, with 0.05 rad of
measurement margin. Open is checked by signed bounds rather than `abs()`,
which prevents a NumPy integer-minimum value from overflowing into an open
verdict. Closed thresholds are provisional bottle-grasp rules checked on
existing episodes, not a general contact classifier.

The [recorded-joint analysis](classification-analysis.json) checks three
closed references and four open offer references. Transitional examples remain
partial. Source 16's slipping bottle also has open fingers: detecting that
failure still requires the visual sequence. Sources 15/30 are training
references; this is not held-out success validation.

## Timestamp association

C++ publishes the original `StateLogger::Entry.right_hand_received_at` beside
the measured joints. Advancing overall telemetry packets cannot refresh that
receipt or create new hand sample identities. Python derives a stable sample
ID from that receipt and accepts an image association only when:

- original DDS receipt age is between 0 and 0.5 seconds;
- hand receipt and camera-cache receipt differ by at most 0.1 seconds;
- controller and native Python have the same Linux boot ID and time namespace.

The clock identity travels as an ASCII numeric vector in existing ZMQ output.
This avoids treating a remote host's monotonic timestamp as local time.
Different clock domains, older C++ binaries without metadata, and unavailable
clock identity yield unknown. Supporting remote controller clocks needs
explicit clock mapping before this association can be accepted.

These timestamps are host receipt times, not proof of synchronized optical
exposure and encoder measurement. Physical sensor latency still needs review.
The native independent `right_hand_open` status guard remains in place.

## Vision integration

The `pick_offer` prompt gets a measured-posture hint for the current image.
Vision still checks bottle lift/support, palm orientation, rolling, collision
and human rescue. A closed or partial measured hand blocks a visual yes.
Unknown or repeated measurement identity cannot complete a live offer.
Visual failure takes priority and remains latched for the cycle.

Startup emptiness, bottle availability and person-removal prompts do not
receive an open-hand hint. In an initial diagnostic, adding the cue to every
phase changed source 30's occluded-removal judgment from unknown to yes,
against its conservative review label. That experiment is retained in
[qwen-phase-cases/result.json](qwen-phase-cases/result.json); it is not the final
integration.

## Qwen with offer-only cues

| Measure, same 13 recorded cases | Previous Qwen | Final offer-only cues |
| --- | --- | --- |
| Agreement with review labels | 8/13 | 9/13 |
| Offer frames returned yes | 0/4 | 0/4 |
| Initial source 16 slip | unknown | failure |
| Yes on a negative review frame | 0 | 0 |
| Median provider latency | 1.67 s | 1.58 s |

All four offer frames are mechanically open according to measured joints,
but Qwen still rejects the bottle-support/offer judgment. Some reasons continue
to call it a closed gripper despite the hint; others say the bottle remains
on the desk. The sensor rule prevents those visual guesses from determining
finger posture, but it does not establish correct visual bottle support.
This feature does not justify adopting Qwen as the production monitor.

[Final Qwen evidence](qwen-offer-cues/result.json) includes joints, classifications,
judgments, token usage and latency. Its receipt times are synthesized when
reading recorded rows; it is not a real-time control test. The source videos
were rehashed and JPEG bytes compared with the saved baseline. The recorded
joint cache hash is pinned in the analysis. Timing comparisons are sequential
runs without isolated load or cache control.

## Verification and deployment scope

Unit checks cover invalid readings, native hand order, metadata decoding,
legacy unknown defaults, mismatched clocks, actual DDS age, repeated DDS
samples, visual failure precedence and two distinct open-hand confirmations.
Cross-process checks use synthetic tracking and a memory-only publisher.
C++ telemetry tests verify unchanged receipts across new logger packets and
empty receipts for absent/stale hand measurements. The updated controller and
test binary were built; no controller was launched or robot commanded.

The paired native and VoLo checkouts must include the updated protocol and
controller telemetry. Hardware profiles and checkpoint/ready-review flags
remain unchanged. [verification.json](verification.json) records the final
test results and [review.json](review.json) records independent review.
The native implementation is commit `b2a6432` on
`work/volo-g1-harness-20261001` in `caisarl76/GR00T-WholeBodyControl`.

Final checks: **739 VoLo tests passed, 19 skipped; 81 native checks passed;
13 explicit cross-process checks passed; 3 rebuilt C++ telemetry/timeout
checks passed**. The artifact evaluator's five checks also passed.

To repeat the recorded Qwen experiment, open the same localhost SSH tunnel
described in [the earlier report](../g1_handover_20261013_qwen/README.md), then
run the evaluator with `--disable-thinking --max-tokens 256` and:

```bash
--measured-hand-cache docs/artifacts/g1_handover_20261009_followup/palm-diagnostics/measured-extract.npz
```

The classifier itself runs on CPU and needs no new model or VLA training.
