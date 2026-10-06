"""Save timestamped video samples without changing source labels or recordings."""

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np


JOBS = [
    (5, "failure_reference", 0, 19.52, 0.5),
    (18, "approach_reference", 0, 18, 0.5),
    (18, "release_reference", 34, 40.46, 0.2),
    (52, "release_or_drop", 49, 54.98, 0.2),
    (78, "release", 35, 44, 0.2),
    (155, "task_boundaries", 0, 170, 2),
    (159, "setup", 0, 7.94, 0.2),
]


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    records = []
    for episode, label, start, end, step in JOBS:
        video = args.dataset / "videos/chunk-000/observation.images.ego_view" / f"episode_{episode:06d}.mp4"
        cap = cv2.VideoCapture(str(video))
        if not cap.isOpened():
            raise ValueError(f"Cannot open {video}")
        fps = cap.get(cv2.CAP_PROP_FPS)
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        assert fps > 0 and count > 0
        stop = min(end, (count - 1) / fps)
        assert 0 <= start <= stop
        requested = np.r_[np.arange(start, stop, step), stop]
        indices = sorted(set(min(count - 1, round(float(t) * fps)) for t in requested))
        assert max(np.diff(indices), default=0) <= round(step * fps) + 1
        sheets = []
        for offset in range(0, len(indices), 16):
            subset = indices[offset:offset + 16]
            canvas = np.full((1110, 1280, 3), 245, np.uint8)
            title = f"Source {episode}: {label}; samples every {step:g}s, not continuous video"
            cv2.putText(canvas, title, (12, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.65, (20, 20, 20), 1, cv2.LINE_AA)
            samples = []
            for tile, index in enumerate(subset):
                cap.set(cv2.CAP_PROP_POS_FRAMES, index)
                ok, frame = cap.read()
                assert ok, (video, index)
                decoded = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
                assert decoded == index, (index, decoded)
                x, y = tile % 4 * 320, 50 + tile // 4 * 265
                h, w = frame.shape[:2]
                ratio = min(320 / w, 240 / h)
                resized = cv2.resize(frame, (round(w * ratio), round(h * ratio)))
                canvas[y:y + resized.shape[0], x:x + resized.shape[1]] = resized
                cv2.putText(canvas, f"{index / fps:.2f}s / frame {index}",
                            (x + 4, y + 258), cv2.FONT_HERSHEY_SIMPLEX,
                            0.45, (20, 20, 20), 1, cv2.LINE_AA)
                samples.append({"frame_index": index, "actual_s": index / fps})
            name = f"episode_{episode:06d}_{label}_{offset // 16}.jpg"
            assert cv2.imwrite(str(args.output / name), canvas, [cv2.IMWRITE_JPEG_QUALITY, 92])
            sheets.append({"path": name, "frames": samples})
        cap.release()
        records.append({"source_episode": episode, "kind": label,
                        "interval_s": [start, stop], "maximum_sample_gap_s": step,
                        "video": str(video), "video_sha256": sha256(video),
                        "fps": fps, "frame_count": count, "sheets": sheets})
        print(json.dumps({"episode": episode, "kind": label,
                          "samples": len(indices), "sheets": len(sheets)}), flush=True)
    report = {"method": "Timestamped sampled frames; events between frames can be missed.",
              "continuous_video_review": False, "dataset_modified": False,
              "accepted_new_outcome_labels": 0, "jobs": records,
              "script_sha256": sha256(Path(__file__))}
    (args.output / "index.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
