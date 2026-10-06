# `pnp_bottle_260916` raw GR00T N1.7 fine-tune

Launched September 30, 2026, inside the pinned GR00T Docker image on H100 physical GPU 7. No packages were installed on the H100 host.

The immutable source is `/mnt/data/jihun/datasets/G1_WBT_GR00T/pnp_bottle_260916`. `prepare_dataset.py` copied its 179 episodes, omitted only source `discarded_episode_indices` 13, 94, 99, 106, and 109, and renumbered the remaining data and video files as a contiguous LeRobot v2.1 dataset. The derived dataset has 174 episodes and 274,241 frames. Its single task string is exactly `pick drink bottle and place it on the right table`. All transferred files passed an `rsync -rcn` checksum comparison.

One retained episode has 37 frames, shorter than the configured 40-frame action horizon. It remains in the derived raw dataset; GR00T yields no training sample from it. The native loader built 262 shards and decoded representative 640×480 ego frames with the exact prompt, 46-dimensional state, 78-dimensional action, and 40-step action horizon.

Run directory on H100: `/mnt/data01/jhkim/gr00t_runs/pnp_bottle_260916_n17_raw_gpu7_20260930`. Container: `jihun_gr00t_n17_pnp_bottle_260916_raw_gpu7_20260930`. Physical GPU 7 is exposed as CUDA device 0 inside the container. Docker image: `sha256:917a790c576f3f00e1a3007e4594b350f75dfd33f77765da093afc3d2593d1db`. The base checkpoint and Cosmos revision are pinned in `run_training.sh` and `launch_gr00t_n17_pinned_finetune.py`.

The run completed 20,000 steps with exit code 0. The pinned preflight and native loader probe both exited 0. The effective argument audit confirmed batch 32, BF16, AdamW, learning rate `1e-4`, cosine schedule, no DeepSpeed, and checkpoints every 1,000 steps. `checkpoint-20000` contains all three model shards.

[W&B run](https://wandb.ai/jihun-kim/gr00t-n1.7-pnp-table/runs/4ofvmcgv)

Monitor without changing the job:

```bash
ssh h100 docker exec jihun_gr00t_n17_pnp_bottle_260916_raw_gpu7_20260930 \
  tail -f /outputs/train/train.log
```

The completion code is written to `train/exit` in the H100 run directory. The training argument audit is `train/training-arguments.json`, the dataset probe is `evidence/native-loader-probe.json`, and checkpoints are under `train/pnp-bottle-260916-raw-gpu7/`.

## Workstation inference server

The completed checkpoint's model and processor files were copied to `/mnt/data/jihun/models/pnp_bottle_260916_n17_raw_gpu7/checkpoint-20000` on the workstation; optimizer and scheduler state were omitted. Transferred files passed checksum comparison. The workstation uses the same GR00T model, policy, server, and embodiment-config source files as the H100 image, with the exact pinned Cosmos snapshot already cached locally.

`serve_workstation_gpu1.sh` launches the official GR00T PolicyServer through `gear_sonic/scripts/serve_pnp_bottle_offline.py`, using workstation physical GPU 1 (RTX 3060, exposed to the process as `cuda:0`). It runs in tmux session `gr00t_bottle_gpu1`, bound to `0.0.0.0:15558`; the workstation LAN endpoint is `192.168.0.62:15558`. The log is `/mnt/data/jihun/models/pnp_bottle_260916_n17_raw_gpu7/server.log`.

Two queries from `smoke_inference.py` used recorded ego-view video and robot state with the exact task prompt. Both returned finite `[1, 40, 64]` motion tokens and `[1, 40, 7]` actions for each hand. The first query took 6.27 seconds during warmup; the second took 0.13 seconds. GPU 1 used about 6.3 GB after inference. This validates model serving and action shape; it does not execute robot motion.

```bash
tmux attach -t gr00t_bottle_gpu1
tail -f /mnt/data/jihun/models/pnp_bottle_260916_n17_raw_gpu7/server.log
```
