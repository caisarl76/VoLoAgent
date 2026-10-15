# Dex3 smoke replay on the merged native branch, 15 October 2026

**Accepted, 5/5 episodes, no gate failures.**

The merge of `main` into the native branch changed what the smoke replay
checks for hand commands. `*_hand_q_measured` became the real Dex3 DDS reading.
`main`'s same-tick hand-command echo moved to new `left/right_hand_q_target`
fields, and `replay_unitree_sonic_smoke.py` was switched to match on them. This
run confirms the switch against a live MuJoCo sim.

## What ran

- **Code:** native `f356332`. That is the merge `9329b16` plus the opt-in yaw
  feedforward, which this replay does not use (no harness flags).
- **Binary:** `347044b5…`.
- **Dataset:** converted with `convert_unitree_unifolm_to_sonic.py --kind dex3 --smoke`
  from the pinned smoke lock (`unitreerobotics/G1_Dex3_Pouring_Dataset@c9552eb`,
  episodes 0, 78, 155, 233, 310). The converter validated 5 episodes with 0
  failures ([`pipeline-report.json`](pipeline-report.json)).
- **Models:**
  - Encoder and observation config: the pinned `nvidia/GEAR-SONIC@9c0ff22`
    `low_latency` files, hash-checked by the replay.
  - Decoder: the matching `low_latency/model_decoder.onnx` (`c4ac2e74…`).
  - Planner: V2.
- **Replay:** `replay_unitree_sonic_smoke.py --network-interface lo` on GPU 1. It
  starts a fresh sim and deploy per episode.
  - Each 50 Hz command must get an exact token and `*_hand_q_target` echo within
    20 ms, or the replay raises an error.
  - 15,409 samples were checked across the 5 episodes.

| Gate | Limit | Worst over 5 episodes |
| --- | --- | ---: |
| Root height | 0.45–1.05 m | 0.708–0.810 m |
| Roll / pitch | ≤ 0.7 rad | 0.080 / 0.335 rad |
| Joint-limit overshoot | ≤ 0.02 rad | 0.011 rad |
| Joint velocity | ≤ 50 rad/s | 16.3 rad/s |
| Torque ratio | ≤ 1.05 | 1.00 |
| Contact force | ≤ 10 × mass × g (3548 N) | 465 N |
| Controller faults, non-finite samples | 0 | 0, 0 |

## Environment note

No single environment had both `lerobot` and the sim stack.
[`replay-report.attempt1-no-lerobot.json`](replay-report.attempt1-no-lerobot.json)
records the first attempt: it failed every episode on the `lerobot` import,
before any sim started.

The accepted run used `.venv_teleop` with `.venv_data_collection`'s
site-packages first on `PYTHONPATH`. That gives torch 2.6, `lerobot` and numpy
from `.venv_data_collection`, and `mujoco`, `cyclonedds` and `unitree_sdk2py`
from `.venv_teleop` and its external dependencies. Both environments are in the
main GR00T-WholeBodyControl checkout and were not modified.

## Files

- [`replay-report.json`](replay-report.json): the accepted report with provenance
  (binary, model and dataset hashes).
- Dataset identity: [`source-manifest.json`](source-manifest.json),
  [`dataset-checksums.sha256`](dataset-checksums.sha256) and
  [`merge-validation.json`](merge-validation.json).
- Logs: [`convert.log`](convert.log) and `replay.log.gz`.
- Not kept: the converted dataset itself (43 MB) and the 4.1 GB download cache.

## Reproduce

From the native worktree, after `source gear_sonic_deploy/scripts/setup_env.sh`.
Here `$DC` is `.venv_data_collection`'s site-packages and `$LL` is a folder with
the pinned `low_latency` files:

```bash
PYTHONPATH=. .venv_data_collection/bin/python gear_sonic/scripts/convert_unitree_unifolm_to_sonic.py --output-root OUT --cache-dir CACHE --kind dex3 --smoke
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=.:$DC .venv_teleop/bin/python gear_sonic/scripts/replay_unitree_sonic_smoke.py \
  --dataset-root OUT --source-episode-ids 0 78 155 233 310 --network-interface lo \
  --policy-file $LL/model_decoder.onnx --encoder-file $LL/model_encoder.onnx --observation-config $LL/observation_config.yaml \
  --planner-file <checkout>/gear_sonic_deploy/planner/target_vel/V2/planner_sonic.onnx \
  --motion-data-path <checkout>/gear_sonic_deploy/reference/example --output-report REPORT.json
```
