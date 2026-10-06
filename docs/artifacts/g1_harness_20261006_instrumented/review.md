# Trace implementation review

An independent gpt-6-astra reviewer at high reasoning inspected the native diff,
new recorder/tests and the probe against the approved bounded diagnostic.
The reviewer sent no robot commands and ran no GPU jobs.

Three initial findings were addressed before the trial:

1. `saved_frame_for_observation_window_` is a window span, not a selected frame.
   Rename the exported field to `observation_window_frames`. The selected
   quaternion is the first frame; the encoder also reads future samples.
2. `Initialize()` does not populate per-stage `last_timing_`. Initial gather,
   inference and resample durations now serialize as null. A regression failed
   on all three fields before the fix, then passed.
3. A file can exist before serialization finishes. Flush writes a `.partial`
   file, checks close and renames; the probe parses the entire final trace and
   verifies declared counts after shutdown. Truncation/count tests pass.

Follow-up review reported no confirmed remaining findings in those fixes.
Main-agent inspection additionally added list/object event counterexamples:
two tests reproduced `TypeError`, then both passed after a string-type guard.
The recorded as-run validator remains preserved; current validation agrees on
the actual complete trace.

Final evidence review independently reproduced the consequential numbers and
approved the report with no blocking findings. It identified a presentation
caveat: generated and merged timelines differ by two frames. The analysis and
plot now align the future hit index by that recorded offset. The 41/48 count is
unchanged, and the report notes that blending can shift the actual merged hit.
