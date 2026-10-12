"""Direct inference on recorded training references; opens no control connection.

Run in the pinned Isaac-GR00T environment with an explicitly allocated GPU.
This checks loading and action shape/finiteness, not physical handover success.
"""
import argparse
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import time
from types import SimpleNamespace

import huggingface_hub
from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
from gr00t.data.dataset.sharded_single_step_dataset import extract_step_data
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.eval.open_loop_eval import parse_observation_gr00t
from gr00t.policy.gr00t_policy import Gr00tPolicy
import numpy as np
import torch

PROMPT = 'pick drink bottle from the table and handover'
COSMOS_REVISION = '9ce19a195e423419c349abfc86fd07178b230561'


@contextmanager
def offline_cosmos_identity():
    # Same model-info shim used by the existing pinned offline serving wrapper.
    snapshot = Path.home()/'.cache/huggingface/hub/models--nvidia--Cosmos-Reason2-2B/snapshots'/COSMOS_REVISION
    config = json.loads((snapshot/'config.json').read_text())
    assert config['model_type']=='qwen3_vl'
    assert config['architectures']==['Qwen3VLForConditionalGeneration']
    original = huggingface_hub.model_info

    def info(repo_id, *args, **kwargs):
        if repo_id!='nvidia/Cosmos-Reason2-2B' or args or kwargs:
            raise RuntimeError('Offline model-info only supports the pinned Cosmos repository')
        return SimpleNamespace(tags=['qwen3_vl'])

    huggingface_hub.model_info = info
    try:
        yield
    finally:
        huggingface_hub.model_info = original


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--preflight-only', action='store_true')
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Choose a new output directory')
    manifest = json.loads((args.dataset/'meta/subset_manifest.json').read_text())
    assert manifest['task_prompt'] == PROMPT
    assert manifest['old_to_new']['15'] == 10 and manifest['old_to_new']['30'] == 19
    assert 16 in manifest['removed_episode_indices']
    index = json.loads((args.checkpoint/'model.safetensors.index.json').read_text())
    shards = sorted(set(index['weight_map'].values()))
    assert all((args.checkpoint/s).is_file() for s in shards)
    assert (args.checkpoint/'processor_config.json').is_file()
    report = dict(robot_connection=False, network_service=False, physical_success_verified=False,
        policy_inference_verified=False, samples_are_training_references=True,
        checkpoint=str(args.checkpoint), dataset=str(args.dataset), prompt=PROMPT,
        checkpoint_metadata_sha256=hashlib.sha256(
            (args.checkpoint/'model.safetensors.index.json').read_bytes()).hexdigest(),samples=[])
    args.output.mkdir(parents=True)
    result_path = args.output/'result.json'
    result_path.write_text(json.dumps(report,indent=2)+'\n')
    if args.preflight_only:
        print(json.dumps(report))
        return
    torch.manual_seed(0)
    started = time.monotonic()
    with offline_cosmos_identity():
        policy = Gr00tPolicy(EmbodimentTag.UNITREE_G1_SONIC, str(args.checkpoint), device=args.device)
    report['cosmos_revision'] = COSMOS_REVISION
    report['model_load_s'] = time.monotonic()-started
    modality = policy.get_modality_config()
    loader = LeRobotEpisodeLoader(args.dataset, modality, video_backend='torchcodec')
    observed = {key:value for key,value in modality.items() if key!='action'}
    for source, subset, frame in [(15,10,62),(15,10,365),(30,19,60),(30,19,370)]:
        sample = extract_step_data(loader[subset], frame, observed, EmbodimentTag.UNITREE_G1_SONIC)
        assert sample.text == PROMPT
        observation = {f'state.{key}':value for key,value in sample.states.items()}
        observation.update({f'video.{key}':np.asarray(value) for key,value in sample.images.items()})
        observation.update({key:sample.text for key in modality['language'].modality_keys})
        started = time.monotonic()
        action,_ = policy.get_action(parse_observation_gr00t(observation,modality))
        record = dict(source_episode=source,subset_episode=subset,source_frame=frame,
                      latency_s=time.monotonic()-started,actions={})
        arrays = {}
        for key,width in [('motion_token',64),('left_hand_joints',7),('right_hand_joints',7)]:
            values = np.asarray(action[key])
            assert values.shape == (1,40,width) and np.isfinite(values).all(), (key,values.shape)
            arrays[key] = values
            record['actions'][key] = dict(shape=list(values.shape),
                max_abs=float(np.max(np.abs(values))),finite=True)
        np.savez_compressed(args.output/f'source{source}_frame{frame}_actions.npz',**arrays)
        report['samples'].append(record)
        result_path.write_text(json.dumps(report,indent=2)+'\n')
    report['policy_inference_verified'] = True
    result_path.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(result=str(result_path),recorded_samples=4,robot_connection=False)))


if __name__ == '__main__':
    main()
