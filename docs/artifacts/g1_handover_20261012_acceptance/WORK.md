# Handover acceptance continuation, 12 October 2026

Continue the approved stationary handover plan. Do not repeat its completed
implementation tasks or enable the hardware example profile.

1. Compare the checkpoint with recorded 40-frame action targets, adding grasp
   and holding observations to the prior start/offer smoke test. Preserve raw
   predictions and report errors per action group. These are training references,
   not an unseen success test.
2. Build an explicit provisional scene: 0.85 m desk, person proxy directly across,
   bottle 0.20 m high / 0.08 m diameter / 0.30 kg. Bottle dimensions carry forward
   the earlier bottle setup; their applicability to handover is unverified.
   Sweep the unspecified desk distance rather than claiming real calibration.
   Check model/joint mapping and collision detection before running physics.
3. Exercise the corrected native ready-return path in headless SONIC/MuJoCo
   simulation, with one action publisher and loopback communication only.
   Render recorded simulation states after the run. Keep geometric screening,
   actual controller tracking, and bottle contact hypotheses distinguishable.
4. Audit training membership and prepare a collection/split review checklist.
   Newly collected physical demonstrations and supervised hardware acceptance
   require the operator; software checks cannot create those observations.

Progress:

- Baseline: 742f820, native 44f9bba; worktrees clean at start.
- Confirmed local source: 61 episodes, 30,035 frames, 50 Hz. Baseline trained on
  43 episodes; confirmed successes 15/30 are included, failure 16 is excluded.
- Ruling: retain the existing fixed right-arm ready candidate and all native
  rate/lead/deadline/settling limits. Scene inputs are provisional; no passing
  simulation can authorize a real return through an unmeasured person's space.
- Source assets explicitly warn the hand model is unstable for simulation.
  Contact results must retain that limitation and cannot establish real bottle
  support without calibrated contact geometry and parameters.

Acceptance continuation through 13 October:

- Expanded checkpoint comparison completed on authorized h100 GPU 0, network
  disabled, original model/data/cache read-only. 13 training observations from
  sources 15/30; 78 saved prediction/reference arrays independently checked.
- Scene tests: feature RED (missing `scene` module after correcting the pytest
  environment), then 6 GREEN. Initial Python environments had no pytest or an
  unrelated ROS plugin dependency (`lark`); disable plugin autoload.
- Ready attempt `ready-040`: local GPU 0 planner startup failed with OOM.
  Hardware connection false; all owned children stopped. Add selected-GPU free
  memory preflight; do not disturb the existing GPU 0 allocation.
- Ready attempt `ready-040-gpu1`: idle GPU 1 loaded SONIC, but native correctly
  rejected the unverified checkpoint flag. No ready motion in that attempt.
- Ruling: set readiness gates only in an explicitly labelled isolated simulation
  fixture to test the ready-return component — required by native gating — cost
  if misused: fixture flags could be mistaken for real acceptance. The hardware
  example stays false/false; fixture checkpoint remains unconfigured and no
  policy action can be published by the scripted prewarm worker.
- `ready-040-fixture`: completed after 5.339 s; independent final 0.5 s measured
  arm error <=0.04860 rad; all 121 physics samples grounded, no support/reset.
  Minimum signed middle fingertip/desk distance -0.264 mm, so clearance screen
  fails. These are actual native dynamics, not a full handover or post-removal
  starting-pose test. All exact owned children stopped.
- Render: 123 frames / 20 fps, 6.15 s, saved measured qpos at wall-time cadence.
  OSMesa unavailable; successful offline EGL rendering on GPU 1. No controller
  or new physics is launched by rendering.
- User steering: estimate distance from grasp moments instead of requiring a
  new chest-distance measurement. Checked grasp windows agree with stored EEF
  within 0.00027 mm; elsewhere stored EEF differs by up to 3.16 cm. Palm mesh
  centres ~30–31 cm forward of torso model origin are scene priors, not bottle
  pose or actual chest-surface measurements. Mass is not in the dataset.
- Geometric desk sweep uses saved ready states and assumed neutral-base source
  poses. It does not rerun dynamics or qualify any relocated desk position.
- New collection/split checklist prepared. Zero confirmed unseen successful
  episodes, zero accepted new positive cuts; no new training.
- Full VoLoAgent suite: initial ROS-plugin collection error, then sandbox abort
  in `test_gr00t_protocol.py` binding local ZMQ. Corrected command outside the
  sandbox with plugin autoload disabled: 709 passed, 19 skipped, 46 warnings.
- Fresh-context gpt-6-astra/high read-only acceptance review: no material
  findings. Reviewer independently reproduced 126 source hashes, 18 asset
  hashes, 78 arrays and ready/grounded/clearance statistics. Root inspection
  retains all pending physical acceptance boundaries. See `review.json`.

Reproduction (from the VoLoAgent worktree):

```bash
env PYTHONPATH=. .venv/bin/python docs/artifacts/g1_handover_20261012_acceptance/run_reference_eval.py --output /tmp/volo-reference-new-run
env PYTHONPATH=. .venv/bin/python docs/artifacts/g1_handover_20261012_acceptance/ready_probe.py --output /tmp/volo-ready-new-run --desk-front-m 0.40 --gpu 1
/home/jihun/work/VoLoAgent/.worktrees/g1-harness-native/.venv/bin/python docs/artifacts/g1_handover_20261012_acceptance/estimate_grasp.py
/home/jihun/work/VoLoAgent/.worktrees/g1-harness-native/.venv/bin/python docs/artifacts/g1_handover_20261012_acceptance/analyze_results.py
env PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /home/jihun/work/VoLoAgent/.worktrees/g1-harness-native/.venv/bin/python -m pytest docs/artifacts/g1_handover_20261012_acceptance/test_scene.py -q
env PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest -q --tb=short
```

Control attempts need their own new output directory. The model runner refuses
to overwrite output and uses a bounded offline container on the authorized GPU.
Rendering and sweep scripts inspect the saved fixed `ready-040-fixture` run;
the sweep refuses to overwrite its scene directory. The original simulator
environment, read-only native model assets and legacy grounded-startup/hand-order
helpers must exist at their recorded absolute paths on this workstation.
