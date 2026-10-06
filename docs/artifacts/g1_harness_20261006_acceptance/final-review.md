# Acceptance investigation review — 2026-10-06

No actionable significant findings. Severity: no blocking or major issues found
within this artifact/analysis-only scope. This is not manipulation, turning,
training-data or hardware acceptance.

Reviewed the new README and provenance, the two documentation diffs, dataset
checker/tests/plan/readiness and control-mode audit, camera proxy helper/result
and intrinsics helper, saved vision manifests/responses/rescore, both native-loop
turn results/traces and summarizer, and the earlier-turn diagnosis. Production
code was outside the change scope. CodeGraph reported a different-worktree index;
the explicitly scoped artifact files were read directly.

Fresh local verification:

- Focused dataset, monitor and planner-wire tests: **27 passed**, no skips
  (`PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python
  -m pytest -p no:cacheprovider -q
  docs/artifacts/g1_harness_20261006_acceptance/test_dataset_review_gate.py
  tests/test_g1_monitor.py
  docs/artifacts/g1_harness_20261006_followup/test_planner_wire.py`).
- Independently joined all 174 plan entries to audit identities/checksums/frame
  counts; confirmed zero segments, mode totals and the blocked checker result.
- Recomputed both saved turn summaries exactly from raw traces/results. Checked
  stable post-interruption world-facing requests, advancing hold telemetry and
  absence of pose packets. The baseline's retained scalar reference is distinct
  from its subsequently published measured-heading hold.
- Compared all eight saved raw vision responses with corrected labels/outcomes.
  The only manifest change is `candidate` to `in_progress` for the first positive
  frame; original exit-1 evidence is preserved.
- Checked all six reproduction commands' option names against parser definitions;
  also inspected the recorded native-probe, vision and calibration CLI options.

The report appropriately preserves the material limits: sampled review supplies
no new accepted episode labels; references 5/18 are future-only holdouts; the
D435i optical convention is assumed and intrinsics remain unmeasured; both new
turns fail despite fresh hold; and the earlier `base_quat_target` is an active
motion-frame target rather than a direct decoder motor output. No simulator,
external API, camera device or hardware operation was launched for this review.
