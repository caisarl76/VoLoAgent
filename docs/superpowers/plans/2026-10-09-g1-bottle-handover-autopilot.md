# G1 Bottle Handover Autopilot Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Implement the stationary, repeating bottle handover loop approved by the user, with a native measured ready return and explicit phase evidence.

**Architecture:** VoLoAgent owns bottle detection, pickup-and-offer monitoring, paused human waiting and repetition under one lease. The native G1 inference loop remains the sole actuator publisher, using its existing bounded reset machinery for a named right-arm ready target. A separate, disabled handover profile prevents activation with the old placement checkpoint.

**Tech Stack:** Python, existing OpenAI-compatible vision client, YAML profiles, local ZeroMQ RPC, SONIC planner, pytest.

**Spec:** [Approved design](../specs/2026-10-09-g1-bottle-handover-autopilot-design.md).

## Global Constraints

- Robot remains stationary: person → desk → G1; no walking, turning or dashboard work.
- One exact trained instruction: `pick up the drink bottle and offer it on your open palm to the person`.
- Preserve the other arm, waist and heading at ready-return entry; open the right hand using the existing zero preset.
- Keep the current 0.5 rad/s joint-rate cap, 0.15 rad measured-lead cap, 0.05 rad settling tolerance, 0.5-second dwell and 15-second reset deadline.
- Keep camera/feedback freshness at 0.5 seconds, lease at 2 seconds and independent heartbeat at 0.25 seconds; do not relax decision expiry.
- Two distinct fresh affirmative frames per transition; unknown or contradictory evidence clears the streak.
- Bind decisions to runtime, cycle, phase, execution, epoch and source frame; latch observed failures for the cycle.
- Episode 16 is a user-confirmed bottle roll caused by palm angle, followed by human rescue. Later empty-hand evidence cannot count that cycle as success.
- Preserve original recordings and the active placement profile. No physical robot commands during implementation or verification.

## Review Focus

1. A bottle rolls off an open palm and is rescued: failure persists across phases and empty-hand judgments (Task 2).
2. A late response arrives after a phase, cycle or execution change: reject it without motion (Tasks 2–3).
3. A person holds the bottle or obscures the robot hand: remain waiting, never reset or trigger another pickup (Tasks 2–3).
4. A ready target is missing, out of joint limits or unreviewed, or the checkpoint is unverified: reject activation before movement (Tasks 1–3).
5. The controller stalls, loses planner acknowledgement or ownership while vision runs: interrupt; no late policy chunk resumes motion (Tasks 1–3).

---

### Task 1: Shared handover profile and native ready return

**Files:** Modify `vlm_orchestrator/harness/g1/{registry,contract}.py` and their native mirrors `gear_sonic/utils/inference/{harness_profile,harness_contract}.py`; modify native `standing_reset.py`, `harness_control.py`, `gear_sonic/scripts/run_vla_inference.py`; add `configs/g1/handover.example.yaml`; update both RPC golden fixtures.

**Interfaces:** Add optional `G1Profile.handover: HandoverSettings | None` with `skill_id`, `checkpoint_verified`, `ready_pose_reviewed` and seven named `ready_right_arm_joints`. Add RPC `reset_ready(execution_id: str)` (empty ID for startup from IDLE/COMPLETED), status `ready_return_enabled: bool` and `right_hand_open: bool | None`. Add `StandingReset(..., right_arm_target=None)` and native hook `begin_ready_reset(feedback, right_arm_target)`.

- [x] Write `tests/test_g1_handover_profile.py` and native `gear_sonic/tests/test_handover_ready.py`: valid candidate ordering, malformed/list/nonfinite targets, joint bounds, unverified checkpoint rejection, preserved left arm/waist/heading/left hand, right-hand zero target, measured rate/lead bounds, .5-second settling and unavailable hook rejection.
- [x] Run the new test files. Expected: failures because the profile/RPC/ready target are unsupported.
- [x] Implement the interfaces. Validate seven named joints against the checked G1 XML bounds; require a hold completion action for the handover skill. Native ready capability requires a reviewed target and implemented hook. Reject handover policy start unless its checkpoint is verified. Example flags remain false.
- [x] Run new tests plus existing profile, RPC, standing reset and native control tests. Expected: all pass; mirrors and fixtures identical.
- [x] Commit with matching author and `Signed-off-by` in each repository.

### Task 2: Phase monitor with persistent roll/drop failure

**Files:** Create `vlm_orchestrator/harness/g1/handover_monitor.py`, `tests/test_g1_handover_monitor.py`; record episode 16 in `docs/artifacts/g1_handover_20261009/`.

**Interfaces:** `HandoverMonitor.begin_phase(phase: str, cycle_id: int, execution: Execution, initial: ObservationSnapshot, not_before: float) -> None`; `check(snapshot) -> HandoverDecision`. Decision carries existing completion fields plus `runtime_id`, `cycle_id` and `phase`. Phases are `ready`, `pick_offer`, `wait_empty`.

- [x] Write tests for two-frame detection, a held/person-held bottle, occlusion resetting confirmations, level/stable support, episode-16-style roll followed by rescue, same-cycle failure surviving phase changes, stale/repeated/wrong-epoch snapshots, malformed responses and expired calls.
- [x] Run the new tests. Expected: missing monitor implementation.
- [x] Reuse bounded off-thread vision calls. Supply initial, previous and current images with phase-specific criteria; accept strict JSON outcomes `yes/no/unknown/failure`. A roll or drop takes precedence over every transition. Keep the failure latched until a new cycle starts.
- [x] Run the new tests and existing monitor tests. Expected: all pass. Store source checksums and sampled incident bracket [335,361), explicitly not an exact event boundary or accepted positive cut.
- [x] Commit with matching author sign-off.

### Task 3: One-session loop, CLI and cross-process acceptance

**Files:** Modify `runner.py` to extract common session setup without changing sequence behavior; create `autopilot.py`; modify `cli.py`; add `tests/test_g1_handover_autopilot.py` and native test fixture support; extend `tests/test_g1_harness_e2e.py`; update harness docs and artifact report.

**Interfaces:** `HandoverRunner(HarnessRunner).run_autopilot(max_cycles: int | None = None) -> MissionResult`. Default repeats until cancellation; positive finite limits support supervised checks. Reuse one lease/heartbeat, monitor and RPC client. Add CLI `autopilot --max-cycles N`, retaining existing vision options and `GENON_API_KEY` fallback.

- [x] Write tests for three cycles with one lease, initial and post-removal ready settling, held-bottle no-trigger, paused waiting through occlusion, closed measured fingers blocking offering, wrong-phase/cycle results, failure/stale evidence/ownership loss/cancel and unavailable native capability. Add a memory-only native IPC run with real contracts and bounded reset, synthetic tracking and scripted vision.
- [x] Run new tests. Expected: missing autopilot and native fixture ready hook.
- [x] Implement phase waits off the main control loop, continuing status/ownership checks. Pause and acknowledge the new epoch before waiting for removal. Return ready only after clear empty-hand confirmation, then wait for native settling before the next cycle. Never auto-restart an interrupted session.
- [x] Run both repositories' appropriate suites and explicit cross-process tests with `--require-g1-e2e`. Expected: all runnable tests pass; report environmental collection failures by name rather than hiding them.
- [x] Request a fresh `gpt-6-astra` high-reasoning standards/spec review. Inspect the diff and resolve material findings with regression tests.
- [x] Update evidence, docs and plan checkboxes with actual results; signed commits and push the existing feature branches.

## Acceptance boundaries

Software acceptance uses synthetic tracking and scripted perception; it does not prove a trained policy can hand over a bottle. Recorded Genon scoring, exact positive cuts from episodes 15/30, episode-level held-out split, handover training/checkpoint verification, palm calibration and desk/person clearance in simulation remain prerequisites for supervised hardware operation. The example profile cannot activate the new task until those reviews are recorded.
