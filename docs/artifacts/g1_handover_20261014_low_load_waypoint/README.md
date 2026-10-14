# Lower-load raised waypoint (option 1): none found, 14 October 2026

Follows the [torque comparison](../g1_handover_20261014_torque_probe/README.md),
which traced the ready-return stall to right-shoulder-yaw gravity load.
**No raised waypoint passed.** Three candidates failed 15/15 trials. The final
screen keeps 0 poses, because two constraints conflict. The hardware template
`configs/g1/handover.example.yaml` is unchanged. No limits, gains, physics or
review flags changed, and no hardware ran.

## Candidates tested in SONIC

All candidates ran 5 independent loopback trials each. The setup matched the
unchanged-physics probe from the torque comparison: native `5deabf4`, binary
`c0058137…`, grounded standing entry, zero-speed planner and no VLA actions.

| Candidate | Raised right arm vs current | Completed | Blocker |
| --- | --- | ---: | --- |
| Current, source 30 f60 (reference) | — | 1 / 3 | right shoulder yaw stall |
| Source 30 f103 (endpoint-sag screen) | arm forward, elbow −0.45 | **0 / 5** | left shoulder yaw drift in 4 runs; lead guard in 2 |
| Source 30 f60, pitch −0.1 / roll +0.1 (synthesized) | less abduction | **0 / 5** | right shoulder yaw stall |
| Source 15 f72 (path-max screen) | arm forward | **0 / 5** | left shoulder yaw drift (0.26–0.42 rad) |

See [`trials.json`](trials.json) (`summarize.py`) for per-trial terminal
reasons and raise-stage worst joints. Two runs end at
`reset_reference_lead_exceeded`: f103-02 at 1.2 s and f103-05 at 4.5 s.

## What the trials established

1. **The stall happens along the path, not at the endpoint.**
   - The reset moves all joints at the same rate. Roll abducts early while
     yaw still has ~1.2 rad to swing.
   - Usable lead is about 0.115 rad: 0.15 lead minus ~0.014 frictionloss
     minus ~0.02 of opposite SONIC offset.
   - The arm stalls if the yaw gravity sag (load/Kp) anywhere on the path
     exceeds that.
   - The path maxima predicted the outcomes:

     | Pose | Path-max sag (rad) | Result |
     | --- | ---: | --- |
     | Current | 0.156 | stalls |
     | Synthesized | 0.129 | stalls 5/5 |
     | Source 15 f72 | 0.095 | yaw settles |
     | Source 30 f103 | 0.057 | yaw settles |

2. **Preserved left shoulder yaw drifts with forward reach.**
   - Its gravity load is only ~0.5 Nm (~0.03 rad sag).
   - The recorded command sits at measured + gravity hold. SONIC holds that
     joint where the right-arm swing pushes it instead of restoring the
     reference. The lead guard then retracts the reference after it.
   - Median drift by 3 s, against static right-hand forward reach:

     | Reach (m) | Drift (rad) | Pose |
     | ---: | ---: | --- |
     | 0.306 | 0.035 | current |
     | 0.327 | 0.053 | synthesized |
     | 0.353 | 0.092 | source 30 f103 |
     | 0.359 | 0.112 | source 15 f72 |

   - The 0.05 rad tolerance is crossed at about 0.32 m. This is a 4-pose
     correlation, not a model.

## Screen

[`screen.py`](screen.py) scores 853 recorded frames from confirmed episodes 15
and 30, plus 616 perturbations of the current pose. Each perturbation moves
every joint no further from entry than the current waypoint does (+0.05 rad).
It keeps poses that:
- hold the whole hand ≥3 cm above the 85 cm desk while behind the 40 cm edge;
- are not already the ready pose;
- have a positive ideal-route conservative desk gap;
- have a path-max right-yaw sag ≤ 0.10 rad;
- reach no further than 0.32 m forward.

[`screen.json`](screen.json) has the full results and the sag/reach Pareto frontier.

**0 poses pass.** On the frontier:

| Pose | Path-max sag (rad) | Reach (m) |
| --- | ---: | ---: |
| Lowest-sag pose within 0.32 m reach | 0.137 | 0.302 |
| Grid neighbour (pitch −0.1, roll +0.15) | 0.119 | 0.326 |
| Sag ≤ 0.10 needs at least | 0.054 | 0.354 |

Within this probe, a waypoint alone cannot satisfy both the right-yaw lead
budget and the 0.05 rad preserved-left-arm check.

## Limits of this conclusion

- The entry is grounded standing, not an actual post-handover pose. Path load
  and the left-arm disturbance both depend on where the return starts. That
  remains an open item.
- Synthesized grid poses are only screened statically. One grid pose was
  simulated; the 0.119 rad frontier neighbour was not.
- Clearance is ideal-tracking and sampled in the provisional scene, not swept or physical.

## Remaining directions (decisions, not taken)

- **Re-run this screen from real post-handover entry poses.** Use the recorded
  end-of-offer frames or VLA rollouts as entry. Do this before concluding
  that option 1 is impossible.
- **Option 2: yaw gravity feedforward via `tau`.** Simulation passed 3/3.
  Unvalidated for SONIC on hardware.
- **Option 3, or a check change.** Stage the raise: abduct roll after yaw has
  swung, which needs a new reset sequence. Or decide whether the preserved
  left arm must hold 0.05 rad during the raise. Both change existing
  behaviour or acceptance.

## Reproduce

From the VoLoAgent worktree, with GPU 1 checked for 2 GiB free:

```bash
/home/jihun/work/VoLoAgent/.worktrees/g1-harness-native/.venv/bin/python docs/artifacts/g1_handover_20261014_low_load_waypoint/screen.py
env PYTHONPATH=. .venv/bin/python docs/artifacts/g1_handover_20261014_torque_probe/run_torque_probe.py --gpu 1 --desk-front-m 0.40 --candidate-profile docs/artifacts/g1_handover_20261014_low_load_waypoint/candidate-source15-f72.yaml --output /tmp/new-dir
/home/jihun/work/VoLoAgent/.worktrees/g1-harness-native/.venv/bin/python docs/artifacts/g1_handover_20261014_low_load_waypoint/summarize.py
```

The candidate YAMLs are isolated simulation profiles, and their review flags
remain false. Of each trial's large streams, only `pd-trace.jsonl.gz` was
dropped, except for f103-01, f103-03 and synth-01, which keep theirs.
