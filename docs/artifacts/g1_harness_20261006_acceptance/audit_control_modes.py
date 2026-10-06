"""Audit every retained recording; control mode is not a task/outcome label."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


MODE_NAMES = {0: "OFF", 1: "POSE", 2: "PLANNER", 3: "PLANNER_FROZEN_UPPER_BODY",
              4: "POSE_PAUSE", 5: "PLANNER_VR_3PT"}


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--mapping", type=Path, required=True)
    parser.add_argument("--enum-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    entries = [json.loads(s) for s in args.mapping.read_text().splitlines()]
    retained = [e for e in entries if e.get("prepared_episode") is not None]
    assert len(retained) == 174, len(retained)
    records, all_modes = [], Counter()
    total = 0
    for episode in retained:
        index = episode["source_episode"]
        path = args.dataset / "data/chunk-000" / f"episode_{index:06d}.parquet"
        t = pq.read_table(path, columns=["teleop.stream_mode", "action.motion_token", "timestamp"])
        modes = np.asarray(t["teleop.stream_mode"].to_pylist()).reshape(-1)
        tokens = np.asarray(t["action.motion_token"].to_pylist())
        timestamps = np.asarray(t["timestamp"].to_pylist()).reshape(-1)
        n = t.num_rows
        assert tokens.shape == (n, 64) and modes.shape == timestamps.shape == (n,)
        assert np.isfinite(tokens).all() and np.isfinite(timestamps).all()
        assert (np.diff(timestamps) > 0).all()
        ids, counts = np.unique(modes, return_counts=True)
        mode_counts = {int(i): int(c) for i, c in zip(ids, counts)}
        assert set(mode_counts) <= set(MODE_NAMES), mode_counts
        all_modes.update(mode_counts)
        boundaries = np.r_[0, np.flatnonzero(np.diff(modes)) + 1, n]
        runs = [{"mode": int(modes[a]), "mode_name": MODE_NAMES[int(modes[a])],
                 "start_frame": int(a), "end_frame_exclusive": int(b),
                 "start_s": float(timestamps[a]), "last_frame_s": float(timestamps[b - 1]),
                 "frames": int(b - a), "full_40_frame_starts": max(0, int(b - a) - 39)}
                for a, b in zip(boundaries[:-1], boundaries[1:])]
        uniform = sum(run["full_40_frame_starts"] for run in runs)
        windows = max(0, n - 39)
        records.append({"source_episode": index, "prepared_episode": episode["prepared_episode"],
                        "frames": n, "parquet_sha256": sha256(path),
                        "mode_counts": {MODE_NAMES[i]: c for i, c in mode_counts.items()},
                        "runs": runs, "full_40_frame_windows": windows,
                        "mixed_mode_40_frame_windows": windows - uniform,
                        "pose_only_40_frame_windows": sum(run["full_40_frame_starts"]
                                                         for run in runs if run["mode"] == 1),
                        "zero_token_frames": int((np.max(np.abs(tokens), axis=1) < 1e-6).sum()),
                        "max_abs_token": float(np.max(np.abs(tokens)))})
        total += n
    assert total == 274241, total
    report = {"retained_episodes": len(records), "frames": total,
              "mode_interpretation": "Numeric IDs interpreted using the archived current exporter enum; historical source revision is unverified.",
              "mode_counts": {MODE_NAMES[i]: c for i, c in sorted(all_modes.items())},
              "pose_frame_fraction": all_modes[1] / total,
              "mixed_mode_40_frame_windows": sum(r["mixed_mode_40_frame_windows"] for r in records),
              "pose_only_40_frame_windows": sum(r["pose_only_40_frame_windows"] for r in records),
              "full_40_frame_windows": sum(r["full_40_frame_windows"] for r in records),
              "zero_token_frames": sum(r["zero_token_frames"] for r in records),
              "not_a_clean_training_set": True,
              "limitations": ["POSE mode can still contain wrong-object tasks or setup.",
                              "Frozen-upper-body planner frames are not automatically invalid training examples.",
                              "No task or outcome labels are accepted from control mode alone.",
                              "Window counts describe source frames, not the exact training sampler or padded tail policy."],
              "dataset_modified": False, "training_performed": False,
              "sources": {str(p): sha256(p) for p in [Path(__file__), args.mapping, args.enum_source]},
              "episodes": records}
    (args.output / "stream-mode-enum-as-read.py").write_bytes(args.enum_source.read_bytes())
    (args.output / "audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "episodes"}, indent=2))


if __name__ == "__main__":
    main()
