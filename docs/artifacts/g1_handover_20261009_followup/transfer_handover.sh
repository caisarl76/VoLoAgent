#!/usr/bin/env bash
# Run on h100 (.173) after connecting from the workstation with: ssh -A h100
# Copies inference inputs and the pinned image to h100_174 (.174).
set -euo pipefail

src=/mnt/data01/jhkim/gr00t_runs/handover_bottle_260930_subset_261002_n17_20k_gpu6_20261006
dst=kube@192.168.75.174:/mnt/data01/jhkim/model_weight/volo-handover-smoke-20261012

rsync -aLP --append-verify --relative --no-owner --no-group \
  --exclude=optimizer.pt --exclude=scheduler.pt --exclude=rng_state.pth \
  --exclude=training_args.bin --exclude=trainer_state.json \
  "$src"/./train/handover-bottle-261002-20k-gpu6/checkpoint-20000 \
  "$src"/./dataset_view/meta \
  "$src"/./dataset_view/data/chunk-000/episode_{000010,000019}.parquet \
  "$src"/./dataset_view/videos/chunk-000/observation.images.ego_view/episode_{000010,000019}.mp4 \
  "$dst"/

image=sha256:3d3fb48f355e59780d86201b61ea7e0251029166b61403bc70d4c6839a2b6c01
if ! ssh kube@192.168.75.174 docker image inspect "$image" >/dev/null 2>&1; then
  docker save "$image" | ssh kube@192.168.75.174 docker load
fi
ssh kube@192.168.75.174 docker image inspect "$image" --format '{{.Id}}'
