"""Make timestamped review sheets; never alter dataset labels or videos."""

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--audit', type=Path, required=True)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    episodes = [json.loads(line) for line in args.audit.read_text().splitlines()]
    selected = [e for e in episodes if e['stationary_review_segments'] or e['source_episode'] in {5, 18}]
    records = []
    for episode in selected:
        index = episode['source_episode']
        video = args.dataset/'videos/chunk-000/observation.images.ego_view'/f'episode_{index:06d}.mp4'
        cap = cv2.VideoCapture(str(video))
        if not cap.isOpened():
            raise ValueError(f'Cannot open {video}')
        fps = cap.get(cv2.CAP_PROP_FPS)
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = (count-1)/fps
        overview = sorted(set([*np.linspace(0, duration, 12),
                               *np.linspace(max(0, duration-3), duration, 4)]))
        groups = [('overview', overview)]
        for i, segment in enumerate(episode['stationary_review_segments']):
            start, end = segment['start_s'], min(duration, segment['end_s'])
            # Include interval boundaries and surroundings. Long flags get <=15s gaps.
            times = sorted(set([max(0, start-2), *np.linspace(start, end, max(6, int(np.ceil((end-start)/15))+1)),
                                min(duration, end+2)]))
            groups.append((f'flag-{i}', times))
        sheets = []
        for label, times in groups:
            for offset in range(0, len(times), 16):
                subset = times[offset:offset+16]
                canvas = np.full((50+4*265, 4*320, 3), 245, np.uint8)
                title = f'Source {index} | {label} | sampled frames, not an outcome label'
                cv2.putText(canvas, title, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, .65, (20,20,20), 1, cv2.LINE_AA)
                frames = []
                for tile, requested in enumerate(subset):
                    frame_index = min(count-1, max(0, round(requested*fps)))
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                    ok, frame = cap.read()
                    if not ok:
                        raise ValueError(f'Frame decode failed: {video}, {frame_index}')
                    actual = frame_index/fps
                    x, y = tile % 4 * 320, 50+tile//4*265
                    h, w = frame.shape[:2]
                    ratio = min(320/w, 240/h)
                    resized = cv2.resize(frame, (round(w*ratio), round(h*ratio)))
                    canvas[y:y+resized.shape[0], x:x+resized.shape[1]] = resized
                    cv2.putText(canvas, f'{actual:.2f}s / frame {frame_index}', (x+4,y+258),
                                cv2.FONT_HERSHEY_SIMPLEX, .45, (20,20,20), 1, cv2.LINE_AA)
                    frames.append(dict(requested_s=float(requested), actual_s=actual, frame_index=frame_index))
                name = f'episode_{index:06d}_{label}_{offset//16}.jpg'
                if not cv2.imwrite(str(args.output/name), canvas, [cv2.IMWRITE_JPEG_QUALITY, 92]):
                    raise ValueError(f'Cannot save {name}')
                sheets.append(dict(path=name, kind=label, frames=frames))
        cap.release()
        records.append(dict(source_episode=index, prepared_episode=episode['prepared_episode'],
                            duration_s=duration, fps=fps, frame_count=count,
                            video=str(video), video_sha256=hashlib.sha256(video.read_bytes()).hexdigest(),
                            flags=episode['stationary_review_segments'], sheets=sheets))
        print(json.dumps(dict(source_episode=index, sheets=len(sheets))))
    info = args.dataset/'meta/info.json'
    (args.output/'source-info.json').write_bytes(info.read_bytes())
    (args.output/'index.json').write_text(json.dumps(dict(
        method='Sampled overview, last three seconds, and flagged intervals. Brief events between frames can be missed.',
        labels_modified=False, training_performed=False,
        source_info_sha256=hashlib.sha256(info.read_bytes()).hexdigest(),
        audit_sha256=hashlib.sha256(args.audit.read_bytes()).hexdigest(), episodes=records), indent=2)+'\n')


if __name__ == '__main__':
    main()
