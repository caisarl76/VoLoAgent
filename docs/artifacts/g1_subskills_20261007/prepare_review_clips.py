"""Create checksum-bound review videos; does not accept labels or alter datasets."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import cv2
import numpy as np


CASES = [
    (52, "hold_candidate", None, [4, 8, 12, 16, 20, 24, 28, 32, 36, 44, 49, 53]),
    (78, "placement_candidate", 46, [4, 8, 12, 16, 20, 24, 28, 32, 35, 37.2, 39, 44]),
    (18, "held_out_success", None, [4, 8, 12, 14, 16, 20, 24, 28, 32, 35.4, 36, 40]),
    (5, "held_out_failure", None, [0, 2, 4, 6, 8, 10, 12, 14.5, 16, 16.5, 18.5, 19.4]),
]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def probe(path):
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,r_frame_rate,nb_frames",
            "-of",
            "json",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(result.stdout)["streams"][0]


def sheet(video, times, output, source, fps):
    capture = cv2.VideoCapture(str(video))
    canvas = np.zeros((3 * 268, 4 * 320, 3), np.uint8)
    try:
        for index, seconds in enumerate(times):
            frame_index = round(seconds * fps)
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = capture.read()
            if not ok:
                raise ValueError(f"Cannot decode source {source} frame {frame_index}")
            tile = np.zeros((268, 320, 3), np.uint8)
            tile[:240] = cv2.resize(frame, (320, 240))
            cv2.putText(
                tile,
                f"source {source} | {seconds:.2f}s | f{frame_index}",
                (5, 259),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (240, 240, 240),
                1,
            )
            row, col = divmod(index, 4)
            canvas[row * 268 : (row + 1) * 268, col * 320 : (col + 1) * 320] = tile
    finally:
        capture.release()
    if not cv2.imwrite(str(output), canvas):
        raise ValueError(f"Cannot write {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path)
    args = parser.parse_args()
    output = Path(__file__).resolve().parent
    manifest = json.loads((output / "candidate-manifest.json").read_text())
    root = args.source_root or Path(manifest["source_root"])
    metadata = {
        "schema_version": 1,
        "fps": 50,
        "accepted_subskill_labels": 0,
        "scope": "Continuous review copies and sampled sheets; no accepted frame boundaries.",
        "clips": [],
    }
    cards = []
    for source, role, seconds, samples in CASES:
        entry = next(e for e in manifest["episodes"] if e["source_episode"] == source)
        parquet = root / f"data/chunk-000/episode_{source:06d}.parquet"
        if sha(parquet) != entry["source_sha256"]:
            raise ValueError(f"Source {source} parquet checksum changed")
        video = (
            root
            / f"videos/chunk-000/observation.images.ego_view/episode_{source:06d}.mp4"
        )
        original = probe(video)
        if (
            original["r_frame_rate"] != "50/1"
            or int(original["nb_frames"]) != entry["frames"]
        ):
            raise ValueError(
                f"Source {source} video frame count/rate differs from source metadata"
            )
        frames = (
            entry["frames"]
            if seconds is None
            else min(round(seconds * 50), entry["frames"])
        )
        stem = f"source_{source:06d}_{role}"
        clip, contact = output / (stem + ".mp4"), output / (stem + ".jpg")
        subprocess.run(
            [
                "ffmpeg",
                "-nostdin",
                "-v",
                "error",
                "-y",
                "-i",
                str(video),
                "-frames:v",
                str(frames),
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
        rendered = probe(clip)
        if rendered["r_frame_rate"] != "50/1" or int(rendered["nb_frames"]) != frames:
            raise ValueError(f"Source {source} review copy lost frame alignment")
        sheet(video, samples, contact, source, 50)
        metadata["clips"].append(
            {
                "source_episode": source,
                "prepared_episode": entry["prepared_episode"],
                "split": entry["split"],
                "source_parquet_sha256": entry["source_sha256"],
                "source_video_sha256": sha(video),
                "source_video": str(video),
                "source_frames": entry["frames"],
                "start_frame": 0,
                "end_frame_exclusive": frames,
                "frames": frames,
                "duration_s": frames / 50,
                "video": clip.name,
                "video_sha256": sha(clip),
                "sheet": contact.name,
                "sheet_sha256": sha(contact),
                "sample_times_s": samples,
                "role": role,
            }
        )
        cards.append(
            f"<article><h2>Source {source}: {role.replace('_', ' ')}</h2>"
            f"<p>Whole-source split: {entry['split']}. Original frames 0–{frames - 1}; "
            "50 fps. Candidate segment cuts remain unaccepted.</p>"
            f'<video controls preload="metadata" src="{clip.name}" poster="{contact.name}"></video>'
            '<p class="position"></p><button class="back">Previous frame</button> '
            '<button class="forward">Next frame</button> '
            '<label>Original frame <input type="number" min="0" class="seek"></label>'
            f'<details><summary>Sampled contact sheet</summary><img src="{contact.name}"></details></article>'
        )
    (output / "review-clips.json").write_text(json.dumps(metadata, indent=2) + "\n")
    page = (
        """<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>G1 bottle subskill review</title><style>
body{font:16px system-ui;margin:24px auto;max-width:960px;background:#101721;color:#e5edf7;padding:0 16px}
article{margin:28px 0;padding:18px;background:#1c2837;border-radius:8px}video,img{width:100%;max-width:800px}
button,input{font:inherit;margin:5px;padding:5px}a{color:#95c7ff}input{width:110px}h2{font-size:20px}
</style><h1>Review the bottle subskills</h1><p>These are real recorded demonstrations, not simulated executions of new skills.
Find an empty-hand start and a steady held-bottle finish above the source table for picking; placement starts from that held pose and finishes after release on the stool.
Require at least two seconds of stable terminal behavior. Source 5 and 18 stay reserved for evaluation of a future checkpoint.</p>
<p>The user confirmed that picking must stop above the source table. Source 52's late hold near the stool does not qualify as that finish.
Source 78 is a placement candidate. Neither has accepted subskill labels.
Use frame stepping while paused and check the full recording before accepting boundaries.</p>"""
        + "\n".join(cards)
        + """
<script>document.querySelectorAll('article').forEach(a=>{const v=a.querySelector('video'),p=a.querySelector('.position'),i=a.querySelector('.seek');
function show(){p.textContent=`Original time ${v.currentTime.toFixed(2)} s | source frame ${Math.round(v.currentTime*50)}`;}
v.addEventListener('timeupdate',show);v.addEventListener('loadedmetadata',()=>{i.max=Math.round(v.duration*50)-1;show();});
function step(n){v.pause();v.currentTime=Math.max(0,Math.min(v.duration-.02,(Math.round(v.currentTime*50)+n)/50));}
a.querySelector('.back').onclick=()=>step(-1);a.querySelector('.forward').onclick=()=>step(1);
i.onchange=()=>{v.pause();v.currentTime=Math.max(0,Math.min(v.duration-.02,Number(i.value)/50));};});</script></html>"""
    )
    (output / "review.html").write_text(page + "\n")
    print(
        json.dumps(
            {
                "clips": len(metadata["clips"]),
                "total_frames": sum(c["frames"] for c in metadata["clips"]),
                "accepted_labels": 0,
            }
        )
    )


if __name__ == "__main__":
    main()
