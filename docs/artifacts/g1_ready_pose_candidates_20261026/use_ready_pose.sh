#!/usr/bin/env bash
# use_ready_pose.sh A|B|C|orig : install a ready-pose candidate as the loop's ready pose (restart the native harness after).
set -euo pipefail
cd "$(dirname "$0")/.."
dst=gear_sonic/data/handover/ready_pose_261020_ep76.npz
[ -f ready_candidates/orig.npz ] || cp -p "$dst" ready_candidates/orig.npz
case "$1" in
  A|B|C) cp ready_candidates/ready_pose_261026_cand$1.npz "$dst" ;;
  orig) cp ready_candidates/orig.npz "$dst" ;;
  *) echo "usage: $0 A|B|C|orig"; exit 2 ;;
esac
echo "$1" > ready_candidates/ACTIVE
echo "ready pose: $1 ($(sha256sum "$dst" | cut -c1-12)); restart the native harness (T2) to load it"
