"""Summarize saved policy replies and render an input-comparison figure."""

import argparse
import itertools
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy-output", type=Path, required=True)
    parser.add_argument("--fixed-inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    original = json.loads((args.policy_output/'result.json').read_text())
    fixed = json.loads((args.fixed_inputs/'inputs.json').read_text())
    actions = np.load(args.policy_output/'actions.npz')
    inputs = np.load(args.policy_output/'inputs.npz')
    repeats = original['same_query_repeat_count']
    pairs = [('Same input repeats', 'recorded-10s', 'recorded-10s'),
             ('Legacy native preparation', 'recorded-10s', 'recorded-10s-native-preparation'),
             ('Change image only', 'recorded-10s', 'recorded-10s-state-sim-image'),
             ('Change state only', 'recorded-10s', 'sim-state-recorded-10s-image'),
             ('Change image + state', 'recorded-10s', 'sim-saved-native')]
    comparisons = []
    for label, a, b in pairs:
        indices = list(itertools.combinations(range(repeats), 2)) if a == b else list(itertools.product(range(repeats), repeat=2))
        metrics = {}
        for key in ['motion_token', 'left_hand_joints', 'right_hand_joints']:
            distances = [float(np.abs(actions[f'{a}__{i}__{key}']-actions[f'{b}__{j}__{key}']).mean()) for i,j in indices]
            metrics[key] = dict(mean_mae=float(np.mean(distances)), min_mae=min(distances), max_mae=max(distances), values=distances)
        comparisons.append(dict(label=label, case_a=a, case_b=b, metrics=metrics))
    target_errors = {}
    for key in ['motion_token', 'left_hand_joints', 'right_hand_joints']:
        values = [row['actions'][key]['recorded_target_mae'] for row in original['queries'] if 'recorded_target_mae' in row['actions'][key]]
        target_errors[key] = dict(mean_mae=float(np.mean(values)), min_mae=min(values), max_mae=max(values), evaluated_chunks=len(values))
    summary = dict(queries=len(original['queries']), recorded_episode=18, state_before=original['state_parity'],
                   state_after=fixed['state_parity'], comparisons=comparisons, recorded_target_errors=target_errors,
                   sim_state_frame_skew_s=original['sim_state_frame_skew_s'],
                   scope='Exploratory 3-repeat comparisons on one in-training episode and one saved simulation image; not a success-rate or significance test.')
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')

    fig = plt.figure(figsize=(15, 9), constrained_layout=True)
    grid = fig.add_gridspec(2, 6, height_ratios=[1.0, 1.05])
    for i, (case, title) in enumerate([('recorded-10s', 'Recorded training episode · 10s'),
                                     ('recorded-15s', 'Recorded training episode · 15s'),
                                     ('sim-saved-native', 'Failed simulation · first saved frame')]):
        ax = fig.add_subplot(grid[0, 2*i:2*i+2])
        ax.imshow(inputs[f'{case}__image'][0,0])
        ax.set_title(title, fontsize=12)
        ax.axis('off')
    ax = fig.add_subplot(grid[1, :3])
    values = [row['metrics']['motion_token']['mean_mae'] for row in comparisons]
    ax.barh(range(len(values)), values, color=['#8796A5', '#8796A5', '#E19736', '#E19736', '#BF542E'])
    ax.set_yticks(range(len(values)), [row['label'] for row in comparisons])
    ax.invert_yaxis()
    for i,v in enumerate(values):
        ax.text(v+0.002, i, f'{v:.3f}', va='center')
    ax.set_xlim(0,max(values)*1.2)
    ax.set_xlabel('Mean absolute motion-token difference (dimensionless)')
    ax.set_title('Policy response to changed inputs · 3 replies per case')
    ax.spines[['top','right']].set_visible(False)
    ax = fig.add_subplot(grid[1, 3:])
    times = [row['seconds'] for row in original['state_parity']]
    before = [np.degrees(row['max_absolute_state_delta']['left_hand']) for row in original['state_parity']]
    after = [np.degrees(row['max_absolute_state_delta']['left_hand']) for row in fixed['state_parity']]
    x = np.arange(len(times))
    ax.bar(x-.18,before,.36,label='Before: overwritten middle fingers',color='#BF542E')
    ax.bar(x+.18,after,.36,label='After: measured fingers retained',color='#2E8B6D')
    ax.set_xticks(x,[f'{t}s' for t in times])
    ax.set_ylabel('Largest left-finger input error (degrees)')
    ax.set_title('Training/native input parity · after fix: zero in all 8 groups')
    ax.legend(frameon=False,fontsize=9)
    ax.spines[['top','right']].set_visible(False)
    fig.suptitle('G1 bottle investigation: input mismatch fixed; scene mismatch still under investigation',fontsize=16)
    fig.savefig(args.output/'input-comparison.png',dpi=170)
    fig.savefig(args.output/'input-comparison.pdf')
    plt.close(fig)
    print(json.dumps(dict(queries=summary['queries'], motion_token_comparisons={row['label']:row['metrics']['motion_token']['mean_mae'] for row in comparisons}, recorded_target_errors=target_errors)))


if __name__ == '__main__':
    main()
