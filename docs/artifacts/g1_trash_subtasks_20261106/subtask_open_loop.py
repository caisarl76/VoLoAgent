"""Open-loop action error per subtask prompt, mirroring gr00t/eval/open_loop_eval.py.

usage: subtask_open_loop.py MODEL DATASET OUT.jsonl EPISODE...
Each chunk is attributed to the prompt the policy saw when predicting it.
"""

from copy import deepcopy
import json
import sys

import numpy as np
import torch

from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
from gr00t.data.dataset.sharded_single_step_dataset import extract_step_data
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.eval.open_loop_eval import parse_action_gr00t, parse_observation_gr00t
from gr00t.policy.gr00t_policy import Gr00tPolicy

HORIZON = 16  # Same as the earlier ckpt-6000/20000 open-loop runs.

model, dataset_path, out_path, *episodes = sys.argv[1:]
tag = EmbodimentTag.resolve("unitree_g1_sonic")
policy = Gr00tPolicy(embodiment_tag=tag, model_path=model, device="cuda" if torch.cuda.is_available() else "cpu")
loader = LeRobotEpisodeLoader(dataset_path=dataset_path, modality_configs=policy.get_modality_config(),
                              video_backend="torchcodec", video_backend_kwargs=None)
action_keys = loader.modality_configs["action"].modality_keys
obs_configs = deepcopy(loader.modality_configs)
obs_configs.pop("action")

with open(out_path, "a") as out:
    for episode in map(int, episodes):
        traj = loader[episode]
        gt = {k: np.vstack(list(traj[f"action.{k}"])) for k in action_keys}
        pred = {k: [] for k in action_keys}
        prompts = []
        for step in range(0, len(traj), HORIZON):
            point = extract_step_data(traj, step, obs_configs, tag)
            obs = {f"state.{k}": v for k, v in point.states.items()}
            obs |= {f"video.{k}": np.array(v) for k, v in point.images.items()}
            for key in loader.modality_configs["language"].modality_keys:
                obs[key] = point.text
            chunk = parse_action_gr00t(policy.get_action(parse_observation_gr00t(obs, loader.modality_configs))[0])
            for k in action_keys:
                pred[k].extend(np.atleast_2d(chunk[f"action.{k}"])[:HORIZON])
            prompts += [str(point.text)] * HORIZON
        n = len(traj)
        err = {k: np.abs(np.asarray(pred[k])[:n] - gt[k]).mean(axis=1) for k in action_keys}
        prompts = np.asarray(prompts[:n])
        for prompt in dict.fromkeys(prompts):
            mask = prompts == prompt
            row = dict(model=model, episode=episode, prompt=prompt, frames=int(mask.sum()))
            row |= {f"mae_{k}": float(err[k][mask].mean()) for k in action_keys}
            out.write(json.dumps(row) + "\n")
        out.flush()
        print(episode, "done", flush=True)
