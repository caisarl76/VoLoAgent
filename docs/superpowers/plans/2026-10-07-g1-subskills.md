# G1 subskill preparation implementation plan

**Goal:** Support a grasp-preserving manipulation handoff and prepare reviewable
pick/placement training segments without enabling unvalidated prompts.

**Architecture:** Optional skill completion behavior in both profile loaders;
native PAUSED readiness gate; coordinator hold acknowledgement and bounded
handoff readiness retries; standalone candidate data review gate and clips.

**Tech stack:** Existing Python harness, native Python control loop, pytest,
JSON metadata, ffmpeg. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-10-07-g1-subskills-design.md`

**Global Constraints:** Existing V/G worktrees, live profile unchanged, no robot
commands, no training on pending labels, archived evidence unchanged, DCO.

**Review Focus:** New handoff epochs, fresh planner feedback, no unintended hand
opening, rejected readiness versus ambiguous RPC outcomes, segment/split bounds.

### Task 1: Grasp-preserving completion and native handoff

Files: `vlm_orchestrator/harness/g1/{registry,runner}.py`,
`tests/test_g1_subskill_handoff.py`, and native mirror
`gear_sonic/utils/inference/{harness_profile,harness_control}.py`,
`gear_sonic/tests/test_harness_control.py`.

1. Add failing tests for default/hold completion, malformed completion actions,
   pick-to-place ordering, missing acknowledgement, pending inference and epoch
   mismatch. Add native rejection tests before fresh hold feedback.
2. Add optional `completion_action` with default `reset_standing` in both loaders.
3. Wait for fresh PAUSED hold with matching execution and epoch. Keep held hands
   through successful cleanup. Retry only explicit NOT_READY during handoff.
4. Require fresh confirmed planner hold for native PAUSED starts.
5. Run focused tests and inspect their actual failures/results.

Verification: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest -q
tests/test_g1_subskill_handoff.py tests/test_g1_runner.py tests/test_g1_registry_contract.py`
and native `PYTHONPATH=. .venv/bin/python -m pytest -q gear_sonic/tests/test_harness_control.py`.

### Task 2: Cross-process handoff proof

Files: `tests/test_g1_harness_e2e.py`, native
`gear_sonic/tests/harness_fake_runtime.py`.

1. Add failing IPC test using a temporary two-skill profile and scripted vision.
2. Extend the memory-only fixture to prove preserved hands, rejected old action
   results during PAUSED, and prompt switching before the placement reset.
3. Run cross-process tests; document that these prove lifecycle behavior only.

Verification: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest -q
tests/test_g1_harness_e2e.py` with the existing native-runtime fixture.

### Task 3: Candidate dataset and review report

Files: `docs/artifacts/g1_subskills_20261007/` and
`tests/test_g1_subskill_dataset.py`.

1. Add failing review-gate tests for valid segments, no boundary crossing,
   source split leaks, malformed metadata and incomplete per-skill coverage.
2. Implement structured review and 40-frame window-manifest preparation.
3. Build a candidate catalogue and source-bound manifest from existing prepared
   metadata; preserve held-out and quarantined sources.
4. Produce continuous clips and a simple review document showing approximate
   review ranges, entry/terminal conditions and missing accepted labels.
5. Run focused/full appropriate checks, request one fresh whole-change review,
   fix confirmed findings, save a validation report, commit with sign-off and
   push the already-authorized development branches.

Verification: focused dataset tests, ffprobe of generated clips, gate CLI
structured rejection on the pending manifest, full VoLoAgent suite and focused
native harness tests.
