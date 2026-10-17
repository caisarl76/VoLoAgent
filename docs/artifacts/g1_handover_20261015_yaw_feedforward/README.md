# Right-shoulder-yaw gravity feedforward via `tau`, 15 October 2026

Follows the [torque comparison](../g1_handover_20261014_torque_probe/README.md)
(option 2) and the [post-handover entry](../g1_handover_20261014_post_handover_entry/README.md)
work, where the raise from grounded standing still stalled.

**With the controller sending right-shoulder-yaw gravity torque through `LowCmd`
`tau`, the raise from grounded standing completed 5/5. It completed 3/8 without.**
The offer entries still return directly, 6/6, with no desk contact.

- The physics is unchanged; only the deploy's own command differs.
- The switch is a profile field, `handover.ready_gravity_feedforward`. It defaults
  to off, and the example template does not set it.
- No hardware ran. Both review flags remain false.

## Change (native `f356332`, binary `347044b5…`)

- **Python:** native `gear_sonic/utils/inference/ready_feedforward.py`
  computes the static gravity torque of motor 24 only.
  - It uses pinocchio with the measured joints, zero velocity and the fixed-base
    model with the pelvis upright.
  - Arm joints are not clipped to URDF limits. The ready pose's right shoulder
    roll sits past its URDF limit, and clipping it gave −0.79 Nm instead of −0.41.
  - Against MuJoCo at the same joints, the error is within 0.01 Nm at the
    waypoint, the ready pose and all three entries. A test checks this.
- **Sending:** during a ready return, every planner message carries
  `upper_body_tau`: 17 f32 values in planner order, with only index 8 nonzero.
  With stale feedback the message carries none.
- **C++:** the deploy adds the field to `tau_ff`. It treats `upper_body_tau` as
  per message and applies zero when:
  - the field is absent, malformed, non-finite, or above 5 Nm on any joint;
  - the planner message times out, including under the harness hold;
  - the deploy enters or leaves PLANNER.

  The reference lead, rate, tolerances, dwells, deadline, gains and the policy
  are unchanged.

## Trials

Same loopback setup as before:
- grounded physics and a zero-speed planner;
- no VLA actions;
- desk front at 0.40 m;
- GPU 1.

The profile is [`candidate-feedforward.yaml`](candidate-feedforward.yaml): the
example template plus the switch.

The probes are unchanged:
- Grounded standing uses `run_torque_probe.py` with no physics feedforward. Its
  PD trace records the delivered `tau`.
- The offers use `run_entry_probe.py --return profile`.

[`trials.json`](trials.json) holds per-run details, written by [`summarize.py`](summarize.py).

| Entry | Decision | Completed | Stage-1 entry | Return time | Min desk gap | Contacts | Preserved max error |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Grounded standing | raise first ×5 | **5/5** | 4.9–5.8 s | — | — | — | 0.041–0.065 rad (raise tail) |
| ep15 offer | direct ×3 | 3/3 | — | 2.7–2.9 s | 57–65 mm | 0 | 0.059–0.071 rad |
| ep30 offer | direct ×3 | 3/3 | — | 2.2–3.6 s | 59–61 mm | 0 | 0.063–0.072 rad |

**Delivered torque, standing runs:**
- Motor 24 received −0.21 to −2.18 Nm.
- No other motor received feedforward (max 0.0 Nm).
- `tau` stayed within 0.13–0.18 Nm of the simulator's actual static gravity
  (mean offset 0.01–0.06 Nm). The pelvis tilt and the 50 Hz update account for
  the difference.
- There was no torque clipping.
- Yaw command minus measured over the reset's last 2 s was −0.014 to +0.003 rad.

**Comparison from grounded standing, ready-return raise:**

| Condition | Completed |
| --- | ---: |
| No feedforward: old native code (`5deabf4`, torque E0) | 1/3 |
| No feedforward: production hook before merge | 1/2 |
| No feedforward: production hook after merge | 1/3 |
| Simulation-side gravity on motor 24 (torque E2, physics modified) | 3/3 |
| **Controller `tau` on motor 24 (this run)** | **5/5** |

The earlier left-arm drift and wrist (motors 27/28) residuals still appear at
0.04–0.08 rad near the end of the raise, but they did not block a 0.5 s dwell in
any run.

**Offer-entry note:** the fixture pre-pose is the first `reset_ready`, so it also
carried feedforward. It is not under test, and every pre-pose completed with the
desk restored. The direct returns match the earlier production-hook runs
(52–65 mm, 0.044–0.08 rad).

## Limits

- **Simulation only:** this is MuJoCo with SONIC unchanged. Sending `tau` changes
  the torque model SONIC was trained with, and a real G1's friction, sag and
  motor `tau` response were not measured.
- **Gravity model:** it assumes an upright pelvis and the URDF masses. A held
  object or a different hand would change the load.
- **Trial count:** 5 standing runs and 6 offer runs. The full connected VLA cycle,
  swept or person clearance and supervised hardware acceptance remain untested.

## Reproduce

From the VoLoAgent worktree, with the native binary rebuilt and GPU 1 checked
for 2 GiB free:

```bash
P=docs/artifacts/g1_handover_20261015_yaw_feedforward/candidate-feedforward.yaml
env PYTHONPATH=. .venv/bin/python docs/artifacts/g1_handover_20261014_torque_probe/run_torque_probe.py --gpu 1 --desk-front-m 0.40 --candidate-profile $P --output /tmp/new-dir
env PYTHONPATH=. .venv/bin/python docs/artifacts/g1_handover_20261014_post_handover_entry/run_entry_probe.py --entry source15_f431 --return profile --gpu 1 --desk-front-m 0.40 --candidate-profile $P --output /tmp/new-dir2
/home/jihun/work/VoLoAgent/.worktrees/g1-harness-native/.venv/bin/python docs/artifacts/g1_handover_20261015_yaw_feedforward/summarize.py
```

`trials/` keeps every run's result, logs, scene, profile and gzip-compressed
streams, including the standing runs' `pd-trace.jsonl.gz`.
