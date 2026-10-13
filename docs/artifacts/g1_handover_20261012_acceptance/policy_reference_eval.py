"""Recorded training-reference comparison; never opens a control connection."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
from gr00t.data.dataset.sharded_single_step_dataset import extract_step_data
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.eval.open_loop_eval import parse_observation_gr00t
from gr00t.policy.gr00t_policy import Gr00tPolicy

from offline_policy_smoke import COSMOS_REVISION, PROMPT, offline_cosmos_identity

SAMPLES = [(15, 10, f) for f in (62, 120, 200, 246, 300, 325, 365)] + [
    (30, 19, f) for f in (60, 120, 200, 260, 320, 370)]
WIDTHS = {'motion_token': 64, 'left_hand_joints': 7, 'right_hand_joints': 7}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((args.dataset/'meta/subset_manifest.json').read_text())
    assert manifest['task_prompt'] == PROMPT
    assert manifest['old_to_new']['15'] == 10 and manifest['old_to_new']['30'] == 19
    assert 16 in manifest['removed_episode_indices']
    config = json.loads((args.checkpoint/'processor_config.json').read_text())['processor_kwargs']
    tag = EmbodimentTag.UNITREE_G1_SONIC
    action_cfg = config['modality_configs'][tag.value]['action']
    assert all(c['rep'] == 'ABSOLUTE' for c in action_cfg['action_configs'])
    statistics = json.loads((args.checkpoint/'statistics.json').read_text())[tag.value]['action']
    report = dict(robot_connection=False, physical_success_verified=False, held_out=False,
        prompt=PROMPT, cosmos_revision=COSMOS_REVISION, checkpoint=str(args.checkpoint),
        checkpoint_index_sha256=hashlib.sha256((args.checkpoint/'model.safetensors.index.json').read_bytes()).hexdigest(),
        action_representation=action_cfg, normalization={k:config[k] for k in (
            'use_percentiles', 'use_mean_std', 'clip_outliers', 'use_relative_action')},
        hand_statistics={k:statistics[k] for k in WIDTHS if 'hand' in k}, samples=[])
    started = time.monotonic()
    with offline_cosmos_identity():
        policy = Gr00tPolicy(tag, str(args.checkpoint), device='cuda:0')
    report['model_load_s'] = time.monotonic()-started
    modality = policy.get_modality_config()
    assert modality['action'].delta_indices == list(range(40))
    loader = LeRobotEpisodeLoader(args.dataset, modality, video_backend='torchcodec')
    for source, subset, frame in SAMPLES:
        sample = extract_step_data(loader[subset], frame, modality, tag)
        assert sample.text == PROMPT
        obs = {f'state.{k}':v for k,v in sample.states.items()}
        obs.update({f'video.{k}':np.asarray(v) for k,v in sample.images.items()})
        obs.update({k:sample.text for k in modality['language'].modality_keys})
        torch.manual_seed(0)
        started = time.monotonic()
        action, _ = policy.get_action(parse_observation_gr00t(obs, modality))
        row = dict(source_episode=source, subset_episode=subset, source_frame=frame,
            latency_s=time.monotonic()-started, actions={})
        arrays = {}
        for key, width in WIDTHS.items():
            pred = np.asarray(action[key])[0]
            truth = np.asarray(sample.actions[key])
            assert pred.shape == truth.shape == (40, width)
            assert np.isfinite(pred).all() and np.isfinite(truth).all()
            difference = pred.astype(np.float64)-truth
            row['actions'][key] = dict(shape=list(pred.shape), finite=True,
                unit='dimensionless latent' if key == 'motion_token' else 'radians; native Dex3 command order',
                mae=float(np.abs(difference).mean()), rmse=float(np.sqrt(np.square(difference).mean())),
                max_abs_error=float(np.abs(difference).max()),
                first_prediction=pred[0].tolist(), first_reference=truth[0].tolist())
            arrays['prediction_'+key], arrays['reference_'+key] = pred, truth
        row['within_native_motion_bound'] = bool(np.abs(action['motion_token']).max() <= 1.25)
        np.savez_compressed(args.output/f'source{source}_frame{frame}.npz', **arrays)
        report['samples'].append(row)
        (args.output/'result.json').write_text(json.dumps(report, indent=2)+'\n')
    report['completed'] = True
    report['samples_compared'] = len(SAMPLES)
    (args.output/'result.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(dict(completed=True, samples_compared=len(SAMPLES), held_out=False,
        physical_success_verified=False)), flush=True)


if __name__ == '__main__':
    main()
