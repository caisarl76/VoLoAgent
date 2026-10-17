"""Prepare an outcome-review queue without changing any dataset or label."""

import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    paths = [args.source/"meta/episodes.jsonl", args.source/"meta/info.json",
             args.prepared/"source_mapping.json", args.prepared/"meta/episodes.jsonl"]
    info = json.loads(paths[1].read_text())
    raw = [json.loads(s) for s in paths[0].read_text().splitlines()]
    mapping = {r['source_episode_index']: r for r in json.loads(paths[2].read_text())}
    prepared = {r['episode_index']: r for r in [json.loads(s) for s in paths[3].read_text().splitlines()]}
    assert set(mapping) == {r['episode_index'] for r in raw} - set(info['discarded_episode_indices'])
    total_frames = sum(row['length'] for row in mapping.values())
    retained_prompts = sorted({task for row in prepared.values() for task in row['tasks']})
    review = []
    for episode in raw:
        source_id = episode['episode_index']
        entry = mapping.get(source_id)
        video = args.source / info['video_path'].format(
            episode_chunk=source_id // info['chunks_size'], episode_index=source_id,
            video_key='observation.images.ego_view')
        row = {'source_episode': source_id, 'prepared_episode': None if entry is None else entry['episode_index'],
               'retained': entry is not None, 'frames': episode['length'],
               'duration_s': episode['length']/info['fps'],
               'retained_frame_share': 0 if entry is None else episode['length']/total_frames,
               'video': str(video), 'outcome': 'unreviewed', 'label_source': None,
               'segments_to_review': [], 'notes': []}
        if entry is not None:
            assert entry['length'] == episode['length'] == prepared[entry['episode_index']]['length']
            row['trained_prompt'] = prepared[entry['episode_index']]['tasks']
        if source_id == 5:
            row.update(outcome='failed_without_recovery', label_source='user confirmation in this session',
                       video_sha256=hashlib.sha256(video.read_bytes()).hexdigest())
        if source_id == 18:
            row.update(outcome='successful_placement', label_source='session visual inspection at 37–40s; user confirmed green-stool destination',
                       video_sha256=hashlib.sha256(video.read_bytes()).hexdigest())
        if source_id == 14:
            row['notes'].append('20-second-spaced session inspection showed many stationary floor/table views; review idle periods and task coverage before any decision.')
        review.append(row)
    (args.output/'episodes.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in review))
    retained = [row for row in review if row['retained']]
    summary = {'source_episodes': len(raw), 'retained_episodes': len(retained), 'retained_frames': total_frames,
               'retained_prompts': retained_prompts,
               'retained_unreviewed': sum(row['outcome']=='unreviewed' for row in retained),
               'longest_retained': sorted(retained, key=lambda row: row['frames'], reverse=True)[:5],
               'allowed_outcomes': ['unreviewed', 'successful_placement', 'failed_without_recovery', 'failed_then_recovered', 'unclear'],
               'sources': {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths},
               'dataset_modified': False, 'training_performed': False,
               'sampling_or_checkpoint_causality_established': False}
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps({key:summary[key] for key in ('source_episodes', 'retained_episodes', 'retained_frames', 'retained_unreviewed')}))


if __name__ == '__main__':
    main()
