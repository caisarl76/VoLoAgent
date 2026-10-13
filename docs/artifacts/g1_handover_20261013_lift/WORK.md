# Lift before reach, 13 October 2026

User instruction: inspect handover demonstrations and avoid desk contact by
raising the hand before operating over the desk. This continues the existing
stationary handover acceptance work. No physical robot commands or activation
of hardware flags.

- VoLoAgent baseline 3009290, native baseline 44f9bba; both worktrees clean.
- Inspect source 15/30 successful approaches and source 16 as a comparison.
  Source 16's later slip/rescue remains a failure, regardless of its approach.
- Compare full hand/forearm geometry for the previous direct ready path and
  raised intermediate poses from the measured demonstrations. Preserve the
  existing right-arm final target, other arm/waist/heading, rate/lead/dwell limits.
- First test candidate geometry with ideal measured tracking; then exercise a
  candidate in the bounded native simulation if geometry supports it. Any
  simulation remains provisional and cannot qualify real person clearance.
- Changes to the controller, if justified, must have regression tests. A
  simulation-only prototype must be labelled as such and must not silently
  enable or alter hardware behavior.

Findings and execution:

- Source 15/30 approach video reviewed at frames 0–160. Under the provisional
  85 cm desk / 40 cm near edge / 78.7 cm pelvis / neutral root assumptions,
  whole-hand height exceeds tabletop by 3 cm at frames 71/52, before crossing
  the near edge at frames 92/78. These are model priors, not measured table pose.
- Six measured intermediate-pose candidates screened with ideal native tracking.
  Source 30 frame 60 gives the largest conservative gap for the corrected
  finger-preserving lift: 49.51 mm. Direct reference bounds overlap the desk.
- Geometry checks include hand mesh and thumb box collision geometry. Initial
  mesh-only assumption rejected by an assert; box corners included explicitly.
  Raw signed queries sometimes yield zero despite separated conservative bounds.
  Use positive bounds gaps as an independent sufficient separation screen;
  zero bounds gaps remain unknown, not a collision proof.
- Tests RED for missing bounds helper, then GREEN on analytic counterexamples.
  Staged reset RED for missing module, then GREEN. First open-near-body prototype
  timed out at 15 s because finger tracking stayed slow while the arm hung low;
  preserved evidence in `staged-native`.
- Three regression failures observed before correction: opening fingers during
  lifting, wrong phase sequence, finger error preventing progress of an already
  raised arm. Corrected to lift with entry finger targets, then open/reach;
  intermediate arm/heading settling on fresh indices, full final settling.
- Ruling: test this sequence as an isolated simulation wrapper — user asked to
  inspect and avoid desk collision; real desk pose remains unverified — cost if
  misused: a provisional reference could be mistaken for a hardware-safe path.
  No production native file or hardware profile changes.
- `lift-first-native`: COMPLETED in 8.942 s, maximum right-arm error in the final
  pre-completion 0.5 s 0.03182 rad. Minimum conservative desk gap in 189 saved
  physics samples 53.21 mm; zero hand/forearm desk contact snapshots. All samples
  grounded, no virtual support/reset. Zero-speed planner packets only; latent
  publication forbidden. Both attempts' owned children stopped.
- The bounded fixture uses 2 GiB free GPU preflight based on the previous 897 MiB
  TRT allocation. The ordinary direct probe still defaults to 3 GiB. Existing
  jobs on GPU 1 were left untouched. No driver/service/job termination.
- Fixture/scene tests: 17 passed. Project suite: 709 passed, 19 skipped,
  46 warnings. Source rehash/profile checks and fresh review recorded below.
- Root rehashed all 126 original source files; unchanged. Both hardware
  profiles match baseline 3009290; native checkout remains clean at 44f9bba.
- Fresh gpt-6-astra/high read-only review found no material defects. Reviewer
  independently reproduced 17 fixture/scene tests, both native comparisons,
  assets/prototype hashes, source30/f60 waypoint, eight source recording hashes,
  native checkout and hardware example. Broader 126-file rehash and full suite
  were verified by root, not repeated by reviewer. See `review.json`.
