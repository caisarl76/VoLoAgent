#!/usr/bin/env bash
# Run on h100 (.173) after connecting from the workstation with: ssh -A h100
# Copies inference inputs and the pinned image to h100_174 (.174).
set -euo pipefail

src=/mnt/data01/jhkim/gr00t_runs/handover_bottle_260930_subset_261002_n17_20k_gpu6_20261006
data_src=/mnt/data01/jhkim/gr00t_runs/handover_bottle_260930_subset_261002_n17_gpu2_20261002/dataset
target=kube@192.168.75.174
dst_path=/mnt/data01/jhkim/model_weight/volo-handover-smoke-20261012
dst="$target:$dst_path"
container=jihun_gr00t_n17_handover_bottle_261002_20k_gpu6_20261006

# dataset_view/data and videos point to /dataset inside Docker, not the host.
# Export metadata as the container user who can read the root-owned stats.
# Host tar creates user-owned copies; original metadata permissions stay intact.
metadata_stage=$(mktemp -d /tmp/volo-handover-meta.XXXXXX)
docker exec --user 0 "$container" \
  tar -C /outputs/dataset_view -cf - meta | \
  tar --no-same-owner --no-same-permissions -C "$metadata_stage" -xf -

for ep in 000010 000019; do
  test -r "$data_src/data/chunk-000/episode_$ep.parquet"
  test -r "$data_src/videos/chunk-000/observation.images.ego_view/episode_$ep.mp4"
done
test -r "$metadata_stage/meta/stats.json"
test -r "$metadata_stage/meta/relative_stats.json"
ssh "$target" mkdir -p \
  "$dst_path/dataset_view/data/chunk-000" \
  "$dst_path/dataset_view/videos/chunk-000/observation.images.ego_view"

rsync -aLP --append-verify --relative --no-owner --no-group \
  --exclude=optimizer.pt --exclude=scheduler.pt --exclude=rng_state.pth \
  --exclude=training_args.bin --exclude=trainer_state.json \
  "$src"/./train/handover-bottle-261002-20k-gpu6/checkpoint-20000 \
  "$dst"/

rsync -aP --no-owner --no-group \
  "$metadata_stage/meta/" "$dst/dataset_view/meta/"
rsync -aP --append-verify --no-owner --no-group \
  "$data_src"/data/chunk-000/episode_{000010,000019}.parquet \
  "$dst/dataset_view/data/chunk-000/"
rsync -aP --append-verify --no-owner --no-group \
  "$data_src"/videos/chunk-000/observation.images.ego_view/episode_{000010,000019}.mp4 \
  "$dst/dataset_view/videos/chunk-000/observation.images.ego_view/"

image=sha256:3d3fb48f355e59780d86201b61ea7e0251029166b61403bc70d4c6839a2b6c01
if ! ssh "$target" docker image inspect "$image" >/dev/null 2>&1; then
  docker save "$image" | ssh "$target" docker load
fi
ssh "$target" docker image inspect "$image" --format '{{.Id}}'
