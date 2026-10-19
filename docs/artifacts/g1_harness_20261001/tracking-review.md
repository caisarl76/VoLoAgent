# Tracking correction review — 2026-10-02

A fresh gpt-6-astra reviewer with high reasoning inspected the native reset/turn
diff, the simulator probe, and actual loopback evidence. The review covered
unchanged physical goals, rate/lead/trim bounds, profile compatibility, missing
feedback, cancellation, deadlines, wraparound, and the meaning of measured
completion. It did not establish hardware or manipulation acceptance.

Two important findings were fixed in one pass:

1. Reset correction used a fixed 0.025-rad deadband. A supported 0.01-rad
   tolerance could never settle against a 0.02-rad steady bias. The regression
   reproduced that settling failure, then passed after deriving the deadband
   from half the profile tolerance and forwarding that tolerance from the
   runtime and probe.
2. The probe reused output folders, overwriting evidence and honoring an old
   support-release marker at startup. The regression demonstrated resource
   creation before rejection. The probe now requires a fresh directory and
   raises FileExistsError before opening resources; existing evidence remains
   unchanged.

The main agent inspected the final changes. The native full suite passes 622
tests, the required cross-process suite passes nine with zero skips, and the
artifact guard passes. Ruff and diff checks pass. These corrections are in
native commit `e1485e0`; no second final review was requested.

The reviewer also identified a minor cleanup robustness issue: a failed final
publish or process-signal race can interrupt probe teardown/result persistence.
That instrumentation issue is deferred. All owned processes stopped normally
in the reported final runs, and the temporary ports were checked afterward.

Recorded reset/turn completion establishes the required fresh measured dwell.
The following walking skill freezes the current measured pose and heading, so
later drift from an earlier goal cannot be used to infer false completion.
These runs do not establish long-duration stationary accuracy or object retention.

## Subsequent empirical correction

A later negative-turn trace exceeded the measured 5-degree lead limit during
backward body drift. Native `d513de5` projects the reference goal into the
measured lead interval before rate limiting. Incompatible feedback jumps raise
before assigning the reference and interrupt the execution with a measured hold.
Three planner regressions were demonstrated RED then GREEN; a lifecycle
regression verifies interruption, retained hands, zero movement and no resume.
The main agent reviewed this subsequent diff. The earlier fresh reviewer did
not review `d513de5`; no second independent review is claimed.

Final native verification: 626 passed in 7.18 seconds. Required cross-process
verification: nine passed, zero skips, 27.23 seconds. Ruff passes. Current signed
physical turns still miss the tolerance and interrupt at their deadline; these
software checks do not close turn acceptance.
