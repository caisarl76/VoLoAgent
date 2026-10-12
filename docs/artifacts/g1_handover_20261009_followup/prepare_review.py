"""Create proposed review clips, preserving all original videos and labels."""
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).parent
SOURCE = Path('/home/jihun/work/GR00T-WholeBodyControl/outputs/handover_bottle_260930')
CUTS = [
    dict(source_episode=15, start_frame=62, end_frame_exclusive=376,
         role='proposed_pick_to_offer', sampled_offer_frames=[365,375],
         user_episode_outcome='success', baseline_training_episode=10),
    dict(source_episode=30, start_frame=60, end_frame_exclusive=391,
         role='proposed_pick_to_offer', sampled_offer_frames=[370,390],
         user_episode_outcome='success', baseline_training_episode=19),
    dict(source_episode=16, start_frame=300, end_frame_exclusive=387,
         role='failure_roll_and_human_rescue', sampled_failure_frames=[335,350,360],
         user_episode_outcome='failure', baseline_training_episode=None),
]

if __name__ == '__main__':
    info = json.loads((SOURCE/'meta/info.json').read_text())
    out = ROOT/'review-clips'
    out.mkdir(exist_ok=True)
    report = dict(frame_interval='zero-based [start,end), at original 50 fps',
        source_data_modified=False, accepted_positive_training_cuts=0,
        scope='Proposed visual cuts only; not a training dataset or held-out policy test.', cuts=[])
    for cut in CUTS:
        ep = cut['source_episode']
        video = SOURCE/info['video_path'].format(episode_chunk=ep//info['chunks_size'],
            episode_index=ep, video_key='observation.images.ego_view')
        target = out/f'source_{ep:06d}_{cut["role"]}.mp4'
        start, end = cut['start_frame'], cut['end_frame_exclusive']
        if not target.exists():
            subprocess.run(['ffmpeg','-v','error','-i',str(video),'-an','-vf',
                f'trim=start_frame={start}:end_frame={end},setpts=PTS-STARTPTS',
                '-c:v','libx264','-crf','18','-pix_fmt','yuv420p',str(target)],check=True)
        metadata = json.loads(subprocess.check_output(['ffprobe','-v','error',
            '-select_streams','v:0','-show_entries','stream=nb_frames,r_frame_rate',
            '-of','json',str(target)]))['streams'][0]
        assert int(metadata['nb_frames']) == end-start and metadata['r_frame_rate']=='50/1'
        report['cuts'].append({**cut, 'accepted_for_training':False,
            'source_video':str(video),'source_sha256':hashlib.sha256(video.read_bytes()).hexdigest(),
            'clip':str(target.relative_to(ROOT)),'clip_sha256':hashlib.sha256(target.read_bytes()).hexdigest(),
            'frames':end-start,'duration_s':(end-start)/info['fps']})
    (ROOT/'review-cut-manifest.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(clips=len(report['cuts']),source_data_modified=False,
                         accepted_positive_training_cuts=0)))
