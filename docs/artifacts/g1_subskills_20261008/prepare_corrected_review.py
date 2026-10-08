"""Build the confirmed source52 crop and source78 failure review evidence.

Original videos/parquet and the initial review media stay unchanged. Retained
rows keep original frame/time/index values; this is not a training dataset.
"""

import argparse
import csv
import hashlib
import html
import importlib.util
import json
from pathlib import Path
import subprocess

import pyarrow.parquet as pq


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    root = Path(__file__).resolve().parent
    previous = root.parent / "g1_subskills_20261007"
    manifest = json.loads((previous / "candidate-manifest.json").read_text())
    corrections = json.loads((root / "review-corrections.json").read_text())
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root", type=Path, default=Path(manifest["source_root"])
    )
    parser.add_argument(
        "--prepared-root", type=Path, default=Path(manifest["prepared_root"])
    )
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location(
        "review_media", previous / "prepare_review_clips.py"
    )
    media = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(media)
    initial = json.loads((previous / "review-clips.json").read_text())
    metadata = {
        "date": "2026-10-08",
        "fps": 50,
        "frame_coordinate_system": "original_source_zero_based",
        "accepted_positive_segments": 0,
        "training_dataset_exported": False,
        "manifest_sha256": sha(previous / "candidate-manifest.json"),
        "corrections_sha256": sha(root / "review-corrections.json"),
        "clips": [],
    }
    cases = [
        (
            52,
            "source_000052_trimmed",
            corrections["source52"]["retained_start_frame"],
            corrections["source52"]["retained_end_frame_exclusive"],
            [36.26, 37, 39, 41, 43, 44, 45, 46, 47, 49, 53, 54.98],
            "Source 52: previous return-to-start removed",
            "Retained rows remain unlabeled. A hold near the stool does not qualify as the source-table pick finish.",
        ),
        (
            78,
            "source_000078_basket_failure",
            corrections["source78"]["incident_start_frame"] - 50,
            corrections["source78"]["incident_end_frame_exclusive"] + 50,
            [29, 29.5, 30, 30.2, 30.4, 30.6, 30.8, 31, 31.2, 31.6, 32, 33],
            "Source 78: basket-fall placement failure",
            "User-confirmed failure around source frames 1500–1600. Later bottle release does not make this a successful placement.",
        ),
    ]
    cards = []
    for source, stem, start, end, samples, title, note in cases:
        entry = next(e for e in manifest["episodes"] if e["source_episode"] == source)
        parquet = args.source_root / f"data/chunk-000/episode_{source:06d}.parquet"
        video = (
            args.source_root
            / f"videos/chunk-000/observation.images.ego_view/episode_{source:06d}.mp4"
        )
        original = next(c for c in initial["clips"] if c["source_episode"] == source)
        if (
            sha(parquet) != entry["source_sha256"]
            or sha(video) != original["source_video_sha256"]
        ):
            raise ValueError(f"Original source {source} bytes changed")
        source_info = media.probe(video)
        if (
            source_info["r_frame_rate"] != "50/1"
            or int(source_info["nb_frames"]) != entry["frames"]
        ):
            raise ValueError(f"Source {source} frame count/rate changed")
        if not 0 <= start < end <= entry["frames"]:
            raise ValueError("Invalid source frame cut")
        clip, sheet = root / (stem + ".mp4"), root / (stem + ".jpg")
        subprocess.run(
            [
                "ffmpeg",
                "-nostdin",
                "-v",
                "error",
                "-y",
                "-i",
                str(video),
                "-vf",
                f"trim=start_frame={start}:end_frame={end},setpts=PTS-STARTPTS",
                "-frames:v",
                str(end - start),
                "-an",
                "-c:v",
                "libx264",
                "-preset",
                "fast",
                "-crf",
                "25",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(clip),
            ],
            check=True,
        )
        info = media.probe(clip)
        if info["r_frame_rate"] != "50/1" or int(info["nb_frames"]) != end - start:
            raise ValueError("Rendered cut lost frame alignment")
        media.sheet(video, samples, sheet, source, 50)
        mapping = root / (stem + "-frame-map.csv")
        with mapping.open("w", newline="") as stream:
            writer = csv.writer(stream, lineterminator="\n")
            writer.writerow(
                ["clip_frame", "source_frame", "clip_time_s", "source_time_s"]
            )
            for index in range(end - start):
                writer.writerow(
                    [index, start + index, index / 50, (start + index) / 50]
                )
        metadata["clips"].append(
            {
                "source_episode": source,
                "prepared_episode": entry["prepared_episode"],
                "source_parquet_sha256": entry["source_sha256"],
                "source_video_sha256": original["source_video_sha256"],
                "start_frame": start,
                "end_frame_exclusive": end,
                "frames": end - start,
                "duration_s": (end - start) / 50,
                "video": clip.name,
                "video_sha256": sha(clip),
                "sheet": sheet.name,
                "sheet_sha256": sha(sheet),
                "frame_map": mapping.name,
                "frame_map_sha256": sha(mapping),
                "note": note,
            }
        )
        cards.append(
            f'<article data-start="{start}" data-end="{end}"><h2>{html.escape(title)}</h2>'
            f"<p>{html.escape(note)} Original frames {start}–{end - 1}, {(end - start) / 50:.2f} seconds.</p>"
            f'<video controls preload="metadata" src="{clip.name}" poster="{sheet.name}"></video>'
            '<p class="position"></p><button class="back">Previous frame</button><button class="forward">Next frame</button>'
            '<label>Original source frame <input class="seek" type="number"></label>'
            f'<details><summary>Sampled source frames</summary><img src="{sheet.name}"></details></article>'
        )
    entry = next(e for e in manifest["episodes"] if e["source_episode"] == 52)
    prepared = (
        args.prepared_root
        / f"data/chunk-000/episode_{entry['prepared_episode']:06d}.parquet"
    )
    mapping = json.loads((args.prepared_root / "source_mapping.json").read_text())
    if not any(
        m["source_episode_index"] == 52
        and m["episode_index"] == entry["prepared_episode"]
        and m["length"] == entry["frames"]
        for m in mapping
    ):
        raise ValueError("Prepared/source episode mapping differs")
    table = pq.read_table(prepared)
    if table.num_rows != entry["frames"] or table["frame_index"].to_pylist() != list(
        range(entry["frames"])
    ):
        raise ValueError("Prepared frame coordinates differ")
    start, end = (
        corrections["source52"]["retained_start_frame"],
        corrections["source52"]["retained_end_frame_exclusive"],
    )
    rows = root / "source_000052_retained_rows.parquet"
    retained = table.slice(start, end - start)
    pq.write_table(retained, rows)
    if not pq.read_table(rows).equals(retained):
        raise ValueError("Retained parquet values differ")
    metadata["retained_rows"] = {
        "file": rows.name,
        "sha256": sha(rows),
        "prepared_episode": entry["prepared_episode"],
        "prepared_parquet_sha256": sha(prepared),
        "rows": retained.num_rows,
        "first_source_frame": start,
        "last_source_frame": end - 1,
        "indices_rebased": False,
        "scope": "Unlabeled retained-row review extract; original columns and indices preserved. Not a GR00T training dataset.",
    }
    page = (
        """<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>Corrected G1 bottle review</title><style>body{font:16px system-ui;margin:24px auto;max-width:960px;background:#101721;color:#e5edf7;padding:0 16px}
article{margin:28px 0;padding:18px;background:#1c2837;border-radius:8px}video,img{width:100%;max-width:800px}
button,input{font:inherit;margin:5px;padding:5px}a{color:#95c7ff}input{width:120px}h2{font-size:20px}</style>
<h1>Corrected bottle review, 8 October</h1><p>Source 52 excludes original frames 0–1812. Source 78's placement failed when the basket was pushed off the stool.
All displayed source frame numbers remain in the original 50 fps recording. Clip time is rebased only for playback.</p>"""
        + "\n".join(cards)
        + """
<p><a href="../g1_subskills_20261007/source_000078_placement_candidate.mp4">Source 78, full first 46 seconds: failure reference including later release</a></p>
<p><a href="README.md">Corrections and training status</a>. No positive subskill labels have been accepted; training has not started.</p>
<script>document.querySelectorAll('article').forEach(a=>{const v=a.querySelector('video'),p=a.querySelector('.position'),i=a.querySelector('.seek'),s=Number(a.dataset.start),e=Number(a.dataset.end);
i.min=s;i.max=e-1;function show(){const f=Math.min(e-1,s+Math.round(v.currentTime*50));p.textContent=`Clip ${v.currentTime.toFixed(2)} s | original source frame ${f} | original time ${(f/50).toFixed(2)} s`;}
v.addEventListener('timeupdate',show);v.addEventListener('loadedmetadata',show);
function step(n){v.pause();v.currentTime=Math.max(0,Math.min((e-s-1)/50,(Math.round(v.currentTime*50)+n)/50));}
a.querySelector('.back').onclick=()=>step(-1);a.querySelector('.forward').onclick=()=>step(1);
i.onchange=()=>{const f=Number(i.value);if(!Number.isFinite(f))return;v.pause();v.currentTime=(Math.max(s,Math.min(e-1,Math.round(f)))-s)/50;};});</script></html>"""
    )
    (root / "review.html").write_text(page + "\n")
    (root / "corrected-media.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(
        json.dumps(
            {
                "retained_source52_frames": retained.num_rows,
                "first_original_frame": start,
                "source78_failure_clip_frames": metadata["clips"][1]["frames"],
                "accepted_positive_segments": 0,
            }
        )
    )


if __name__ == "__main__":
    main()
