# Offline diagnosis of four native-loop turns

Both failures are measured tracking failures under the existing lead and deadline
bounds. The saved evidence supports neither a simple sign inversion nor a wrap
discontinuity, and the deadline/dwell decisions reproduce correctly. It does not
yet identify whether the underlying tracking shortfall is generated-trajectory
bias, trajectory timing, lower-controller behavior, or simulator contact/state.
No new simulator or hardware trial was run, and no limit or production code was
changed.

Reproduce from the harness worktree:

```sh
python3 docs/artifacts/g1_harness_20261006_acceptance/turn-diagnosis/analysis.py
ruff check docs/artifacts/g1_harness_20261006_acceptance/turn-diagnosis/analysis.py
```

The script reads the four immutable `g1_harness_20261006_followup/native-turns/`
archives and writes [findings.json](findings.json) plus per-case timeline CSVs
here. It uses only Python's standard library and records hashes of input traces,
results, and archived bounded-planner source. All four archived results name the
same controller binary SHA-256
`2d42a14cb0ef9c1fef4d00d44da0b7a34b1593b3af53d9df6dbb45daccac4e0b`.

## Where the heading falls behind

All angles below are relative to measured turn entry, not the setup heading.
“Motion target” is yaw extracted from `base_quat_target`. Although the older plot
calls it “decoded base target,” current C++ source builds it from the active motion
frame plus heading correction. It is not a direct decoder motor output, and the
first gap cannot be attributed exclusively to the decoder.

| Case | Final facing reference | Final motion target | Final measured yaw | Measured goal error | Result |
| --- | ---: | ---: | ---: | ---: | --- |
| negative-zero | −12.154° | −8.947° | −7.171° | 7.829° | Deadline |
| positive-zero | +17.684° | +15.997° | +12.815° | 2.185° | Complete |
| negative-wrap | −17.756° | −14.607° | −12.845° | 2.155° | Complete |
| positive-wrap | +16.959° | +14.360° | +11.966° | 3.034° | Deadline |

The goal remains ±15°. References beyond that goal in three cases are the
existing post-nominal trim, still subject to the 5° measured lead bound.

Over the last two saved seconds, the direction-adjusted mean gaps are:

| Case | Facing minus motion target | Motion target minus measured | Net measured rate |
| --- | ---: | ---: | ---: |
| negative-zero | 3.181° | 1.819° | −0.682°/s |
| positive-zero | 2.238° | 2.706° | +2.061°/s |
| negative-wrap | 3.267° | 1.731° | −1.583°/s |
| positive-wrap | 2.332° | 2.668° | +0.104°/s |

These are simultaneous reported-signal gaps, not latency-corrected steady-state
bias estimates. They expose two separate contributions to the overall tracking
shortfall; ordinary temporal lag can contribute to both.

Negative-zero spends 95.2% of saved turn ticks within 0.01° of the 5° lead limit.
Its reference never reaches the nominal −15°, so the existing trim never starts.
At roughly 0.50 seconds its reference has already moved −4.483°, while the motion
target and measured yaw are still +0.707° and +0.517°. Later both move in the
requested negative direction but too slowly. The first divergence is already
visible between facing and motion target, before considering measured tracking.
Changing only post-nominal trim cannot address this run's saturated lead limit.

Positive-wrap reaches nominal facing at 3.110 seconds; its reconstructed trim
eventually saturates at +5°. Actual facing remains limited by measured yaw +5°,
and 89.8% of its samples are at the lead limit. At the end the target is near the
nominal goal (+14.360°), while measured yaw is +11.966°. More trim has no available
authority under the same lead bound. This is not evidence that the bound should
be widened.

Negative-wrap has almost the same final-window gap split as negative-zero, but
more than twice the magnitude of net measured rate and completes at 8.489 seconds.
That comparison rules out a universal inability to turn negatively; it does not
prove that world heading causes the speed difference. Each trial has a different
physical entry state and timing, and there is only one run per configuration.

## Temporal, conversion, and wrap checks

The independent scalar recurrence reconstructs every checked pre-finish reference
update exactly (maximum error 0° in all four cases), including the measured-lead
clamp, 10°/s rate clamp, and post-nominal trim. This uses recorded `dt_s`, previous
reference, measured heading, and integrated trim. It audits the saved recurrence;
it is not an independent execution of the full native loop. Maximum saved
reference rate is 10°/s and lead is 5°, to floating-point precision.

Using only advancing telemetry indices gives these measured tolerance spans:

| Case | In-tolerance interval(s) from first saved turn tick | Longest span |
| --- | --- | ---: |
| negative-zero | None; minimum error 7.829° | 0 s |
| positive-zero | 2.870–3.412 s, 23 advancing indices | 0.542 s |
| negative-wrap | 7.927–8.488 s, 29 advancing indices | 0.562 s |
| positive-wrap | 9.090–9.130 s and 9.190–9.231 s, three indices each | 0.040 s |

Positive-wrap briefly reaches a minimum error of 2.962°, but never holds the 3°
criterion for 0.5 seconds. The deadline result is therefore correct even though
there are individual passing samples. Both successful runs satisfy the dwell.
The traces include 30, 13, 20, and 33 repeated indices respectively; removing
them does not alter the outcomes. No index regressions occur. Maximum saved tick
gaps are 20.7–22.6 ms, providing no evidence of a host-loop stall near deadline.
Log times are recorded after `tick`, so exact dwell-clock equality is not claimed.

World-facing to reference-frame-facing angular offsets remain constant per run:
−0.043762°, −0.444054°, −170.862178°, and +169.800585°. Their maximum drift is
2.55×10⁻¹⁴°. Every saved conversion has a matching facing in a planner packet
received within 150 ms, to at worst 1.33×10⁻⁶°. The asynchronous receipt matching
checks consistency, not exact packet identity or controller latency. The source
reference quaternion itself was not captured, so its absolute correctness cannot
be independently proven from this archive.

Both wrap runs cross the raw ±π representation boundary while circular reference
increments remain rate-bounded. Both signs ultimately produce correctly directed
motion targets and measured changes. The saved 10 Hz simulator yaw independently
follows measured telemetry: nearest-host-time mean absolute differences are
0.024°, 0.127°, 0.035°, and 0.030°. These asynchronous comparisons do not certify
sub-tick settling or identify a synchronization delay.

The archived failed runs retain fresh hold acknowledgements after 0.091 and
0.070 seconds and 49/50 subsequent stationary planner packets. This diagnosis
does not add new stop evidence or reproduce the older lead/rate-guard failure.

## Relevant source boundaries

The archived `as-run/bounded_planner.py:84–125` checks the deadline before
advancing the turn, gates trim on nominal reference arrival, bounds facing against
measured yaw, and accumulates dwell only on advancing indices.
`as-run/harness_control.py:328–408` updates accepted feedback and turns planner
exceptions into interruption; completion additionally waits for hold feedback.
`as-run/run_vla_inference.py:518` converts before publication, while line 819
validates conversion when accepting a planner command.

The following native sources were inspected at the task's native worktree
(`7994c8b`), rather than recovered from the as-run binary; their correspondence to
the binary's internals was not independently reconstructed:

- `gear_sonic/utils/inference/planner_heading_frame.py:10`: inverse rotation by
  the heading telemetry, using vectors rather than wrapped scalar subtraction.
- `gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/include/input_interface/zmq_manager.hpp:638–659`:
  received facing passes through vector normalization to the movement state.
- `.../src/g1_deploy_onnx_ref.cpp:3702–3820`: changed facing triggers replanning;
  the planner thread runs at 10 Hz (`planner_dt_=0.1`, line 2173).
- `.../include/localmotion_kplanner.hpp:554–567`: planning uses existing motion
  context and a two-frame lookahead, then generates/resamples the trajectory.
- `.../include/output_interface/output_interface.hpp:210–250`: target orientation
  is the active motion quaternion with heading correction.

These boundaries explain why a correctly transported facing vector need not
equal the currently reported target. The traces lack the effective planner input,
generated future root-yaw trajectory, context, and active-frame index needed to
separate learned target bias from replanning or frame-selection lag. Likewise,
target-to-measured error alone does not separate decoder behavior from physics.

## Smallest discriminating next experiment

Run **one instrumented negative-zero repeat** through the existing native-loop
probe, preserving startup, the one-second backward command, models, scene, DDS
loopback, and all acceptance/stop limits. Change only observability: capture into
preallocated memory at the C++ planner boundary and flush after the trial, so
logging does not introduce synchronous I/O into the controller loop.

Capture timestamped/indexed received facing and its source sequence, heading
correction quaternion, effective ONNX facing/context, generated future root-yaw
trajectory, active frame and target quaternion, measured quaternion, and inference
start/end and publication times. Existing CSV trajectories are the baseline.

- Wrong recovered world-facing at the controller boundary identifies a frame or
  input mismatch.
- Correct facing but a future trajectory that stays short implicates trajectory
  generation for that input/context.
- Future trajectory reaches facing but active frames stay behind or are repeatedly
  replaced implicates scheduling/context rollout.
- Correct active target with lagging measured yaw localizes the remaining shortfall
  below trajectory generation; motor/decoder/contact evidence is then needed.

This experiment changes no command behavior and does not loosen the 10°/s rate,
5° lead, 3°/0.5 s measured criterion, or 10-second deadline. Keep the existing
post-interruption hold check. One diagnostic repeat is not repeatability or
hardware acceptance. It is proposed only; nothing was launched for this report.
