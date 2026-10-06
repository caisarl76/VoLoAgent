"""Read all retained bottle episodes; flag numerical and stationary-data issues.

These are review priorities, not outcome labels or instructions to drop data.
The dataset and the training checkpoint are never modified.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


def ranges(mask):
    boundaries = np.flatnonzero(np.diff(np.r_[False, mask, False].astype(int)))
    return list(zip(boundaries[::2].tolist(), boundaries[1::2].tolist()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--mapping', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    entries = json.loads(args.mapping.read_text())
    fps, rows, files = 50, [], {}
    for entry in entries:
        episode = entry['source_episode_index']
        path = args.source/f'data/chunk-{episode//1000:03d}/episode_{episode:06d}.parquet'
        table = pq.read_table(path, columns=['observation.state','action.motion_token','teleop.left_hand_joints','teleop.right_hand_joints'])
        q = np.asarray(table['observation.state'].to_pylist(),dtype=float)
        token = np.asarray(table['action.motion_token'].to_pylist(),dtype=float)
        hands = np.concatenate([np.asarray(table[k].to_pylist(),dtype=float) for k in ['teleop.left_hand_joints','teleop.right_hand_joints']],axis=1)
        assert q.shape == (entry['length'],43) and token.shape == (entry['length'],64) and hands.shape == (entry['length'],14)
        finite = np.isfinite(q).all(1)&np.isfinite(token).all(1)&np.isfinite(hands).all(1)
        legacy_delta = np.max(np.abs(q[:,24:26]-q[:,22:24]),axis=1)
        stationary = np.zeros(len(q),dtype=bool)
        if len(q)>fps:
            # One-second endpoints: this can miss a motion that returns to its
            # start within that second. It only nominates video segments to inspect.
            stationary[fps:] = ((np.max(np.abs(q[fps:]-q[:-fps]),axis=1)<0.03)
                                &(np.max(np.abs(token[fps:]-token[:-fps]),axis=1)<0.01)
                                &(np.max(np.abs(hands[fps:]-hands[:-fps]),axis=1)<0.03))
        segments = [{'start_s': start/fps, 'end_s': end/fps, 'duration_s': (end-start)/fps}
                    for start,end in ranges(stationary) if end-start>=5*fps]
        row = dict(source_episode=episode,prepared_episode=entry['episode_index'],frames=len(q),
                   nonfinite_frames=int((~finite).sum()), token_bound_exceeded_frames=int((np.abs(token)>1.25).any(1).sum()),
                   maximum_absolute_motion_token=float(np.max(np.abs(token))),
                   legacy_left_finger_max_delta_rad=float(np.max(legacy_delta)),
                   legacy_left_finger_frames_over_0_1_rad=int((legacy_delta>0.1).sum()),
                   yields_40_frame_window=len(q)>=40,stationary_review_segments=segments,
                   stationary_flagged_frames=sum(round(s['duration_s']*fps) for s in segments),
                   outcome='unreviewed',stationary_flags_are_outcome_labels=False)
        rows.append(row)
        files[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    summary = dict(retained_episodes=len(rows),retained_frames=sum(r['frames'] for r in rows),
                   nonfinite_frames=sum(r['nonfinite_frames'] for r in rows),
                   token_bound_exceeded_frames=sum(r['token_bound_exceeded_frames'] for r in rows),
                   legacy_left_finger_max_delta_rad=max(r['legacy_left_finger_max_delta_rad'] for r in rows),
                   legacy_left_finger_frames_over_0_1_rad=sum(r['legacy_left_finger_frames_over_0_1_rad'] for r in rows),
                   shorter_than_action_horizon=[r['source_episode'] for r in rows if not r['yields_40_frame_window']],
                   episodes_with_stationary_review_segments=sum(bool(r['stationary_review_segments']) for r in rows),
                   stationary_flagged_frames=sum(r['stationary_flagged_frames'] for r in rows),
                   stationary_review_priorities=sorted(rows,key=lambda r:r['stationary_flagged_frames'],reverse=True)[:10],
                   visual_outcomes_reviewed=0,dataset_modified=False,training_performed=False,
                   method='One-second endpoint displacement proxy, thresholds: state max <0.03 rad, token max <0.01, hand-action max <0.03 rad; contiguous flags >=5s. Not proof of idle, failure, or training sampling weight.',
                   sources={**files,str(args.mapping):hashlib.sha256(args.mapping.read_bytes()).hexdigest()})
    (args.output/'episodes.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps({k:v for k,v in summary.items() if k not in ['sources','stationary_review_priorities']}))


if __name__ == '__main__':
    main()
