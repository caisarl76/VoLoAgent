#!/usr/bin/env python3
"""Copy raw LeRobot episodes, omitting only the source's discarded IDs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def scalar_stats(values: np.ndarray) -> dict:
    return {
        "min": [int(values.min())],
        "max": [int(values.max())],
        "mean": [float(values.mean())],
        "std": [float(values.std())],
        "count": [len(values)],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prompt", required=True)
    args = parser.parse_args()

    source = args.source.resolve(strict=True)
    output = args.output
    if output.exists():
        raise FileExistsError(output)
    info = json.loads((source / "meta/info.json").read_text())
    episodes = read_jsonl(source / "meta/episodes.jsonl")
    stats = {row["episode_index"]: row for row in read_jsonl(source / "meta/episodes_stats.jsonl")}
    discarded = set(info["discarded_episode_indices"])
    if len(episodes) != info["total_episodes"] or set(stats) != {e["episode_index"] for e in episodes}:
        raise ValueError("source episode metadata is inconsistent")
    if not discarded <= set(stats):
        raise ValueError("source discarded IDs are missing")
    if len(read_jsonl(source / "meta/tasks.jsonl")) != 1:
        raise ValueError("expected exactly one source task")

    output.mkdir(parents=True)
    (output / "meta").mkdir()
    shutil.copy2(source / "meta/modality.json", output / "meta/modality.json")
    retained = []
    retained_stats = []
    mapping = []
    offset = 0
    for episode in episodes:
        old_id = episode["episode_index"]
        if old_id in discarded:
            continue
        new_id = len(retained)
        length = episode["length"]
        old_data = source / info["data_path"].format(
            episode_chunk=old_id // info["chunks_size"], episode_index=old_id
        )
        table = pq.read_table(old_data)
        if len(table) != length:
            raise ValueError(f"episode {old_id}: parquet length differs from metadata")
        if not np.array_equal(table["frame_index"].to_numpy(), np.arange(length)):
            raise ValueError(f"episode {old_id}: frame indices are not contiguous")
        if not np.all(table["episode_index"].to_numpy() == old_id):
            raise ValueError(f"episode {old_id}: parquet episode ID differs from metadata")

        new_indices = np.arange(offset, offset + length, dtype=np.int64)
        for key, values in (("episode_index", np.full(length, new_id, dtype=np.int64)),
                            ("index", new_indices)):
            position = table.schema.get_field_index(key)
            table = table.set_column(position, key, pa.array(values, type=table.schema.field(key).type))
        new_data = output / info["data_path"].format(
            episode_chunk=new_id // info["chunks_size"], episode_index=new_id
        )
        new_data.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, new_data, compression="zstd", row_group_size=1024)

        for video_key, feature in info["features"].items():
            if feature["dtype"] != "video":
                continue
            old_video = source / info["video_path"].format(
                episode_chunk=old_id // info["chunks_size"],
                episode_index=old_id, video_key=video_key,
            )
            new_video = output / info["video_path"].format(
                episode_chunk=new_id // info["chunks_size"],
                episode_index=new_id, video_key=video_key,
            )
            new_video.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(old_video, new_video)

        retained.append({"episode_index": new_id, "tasks": [args.prompt], "length": length})
        row = stats[old_id].copy()
        row["episode_index"] = new_id
        row["stats"] = row["stats"].copy()
        row["stats"]["episode_index"] = scalar_stats(np.full(length, new_id, dtype=np.int64))
        row["stats"]["index"] = scalar_stats(new_indices)
        retained_stats.append(row)
        mapping.append({"source_episode_index": old_id, "episode_index": new_id, "length": length})
        offset += length

    info.update(
        total_episodes=len(retained), total_frames=offset, total_videos=len(retained),
        total_chunks=(len(retained) + info["chunks_size"] - 1) // info["chunks_size"],
        total_tasks=1, splits={"train": f"0:{len(retained)}"},
        discarded_episode_indices=[],
    )
    (output / "meta/info.json").write_text(json.dumps(info, indent=2) + "\n")
    write_jsonl(output / "meta/episodes.jsonl", retained)
    write_jsonl(output / "meta/episodes_stats.jsonl", retained_stats)
    write_jsonl(output / "meta/tasks.jsonl", [{"task_index": 0, "task": args.prompt}])
    (output / "source_mapping.json").write_text(json.dumps(mapping, indent=2) + "\n")
    print(json.dumps({"episodes": len(retained), "frames": offset, "discarded": sorted(discarded),
                      "prompt": args.prompt, "output": str(output)}))


if __name__ == "__main__":
    main()
