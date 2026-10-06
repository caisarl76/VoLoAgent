"""Plot saved native planner, decoder target and measured heading evidence."""

import argparse
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def rows(path):
    return [json.loads(s) for s in path.read_text().splitlines()]


def wrap(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


def yaw(quaternion):
    w, x, y, z = np.asarray(quaternion)/np.linalg.norm(quaternion)
    return math.atan2(2*(w*z+x*y), 1-2*(y*y+z*z))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    summaries = []
    names = ['negative-zero', 'positive-zero', 'negative-wrap', 'positive-wrap']
    for name, ax in zip(names, axes.flat):
        folder = args.root/'native-turns'/name
        result = json.loads((folder/'result.json').read_text())
        trace = rows(folder/'native-trace.jsonl')
        turn = [r for r in trace if r['event'] == 'turn_sample']
        first, last = turn[0], turn[-1]
        goal = first['goal_yaw_rad']
        origin = wrap(goal-math.radians(result['turn_angle_deg']))
        times = [r['at']-first['at'] for r in turn]
        measured = [math.degrees(wrap(r['measured_yaw_rad']-origin)) for r in turn]
        reference = [math.degrees(wrap(r['reference_yaw_rad']-origin)) for r in turn]
        target = [math.degrees(wrap(yaw(r['controller_target_quaternion'])-origin)) for r in turn]
        truth = [r for r in rows(folder/'ground-truth.jsonl') if first['at'] <= r['at'] <= last['at']]
        ax.axhline(result['turn_angle_deg'], color='black', linewidth=1, linestyle=':', label='Measured goal')
        ax.fill_between(times, result['turn_angle_deg']-3, result['turn_angle_deg']+3, color='#30aa75', alpha=.1)
        ax.plot(times, reference, color='#3266b5', label='Planner reference')
        ax.plot(times, target, color='#c0871b', label='Decoded base target', alpha=.8)
        ax.plot(times, measured, color='#b82e45', label='Controller measured yaw')
        ax.scatter([r['at']-first['at'] for r in truth],
                   [math.degrees(wrap(yaw(r['floating_base_pose'][3:])-origin)) for r in truth],
                   s=5, color='black', alpha=.25, label='Saved physics yaw (10 Hz)')
        ax.set_title(f'{name}: '+('turn completed' if result['turn_success'] else 'deadline interruption'))
        ax.set_xlabel('Seconds from first saved turn tick')
        ax.set_ylabel('Yaw relative to measured entry (degrees)')
        ax.grid(alpha=.2)
        dwell_start = None
        advancing = []
        previous_index = None
        for row in turn:
            index = row['telemetry_index']
            if previous_index is not None and index <= previous_index:
                continue
            previous_index = index
            if abs(wrap(goal-row['measured_yaw_rad'])) <= math.radians(3):
                dwell_start = row['at'] if dwell_start is None else dwell_start
                advancing.append(index)
            else:
                dwell_start, advancing = None, []
        error = abs(math.degrees(wrap(goal-last['measured_yaw_rad'])))
        rate = [abs(math.degrees(wrap(r['reference_yaw_rad']-r['previous_reference_yaw_rad']))/r['dt_s'])
                for r in turn if r['dt_s'] > 0]
        summary = dict(case=name,turn_success=result['turn_success'],control_checks=result['outcome'],
                       final_error_deg=error,elapsed_s=result['turn_elapsed_s'],
                       entry_yaw_measured_deg=math.degrees(origin),
                       final_in_tolerance_span_s=0 if dwell_start is None else last['at']-dwell_start,
                       advancing_indices_in_final_span=len(advancing),
                       max_saved_reference_rate_deg_s=max(rate),
                       max_saved_reference_lead_deg=max(abs(math.degrees(r['lead_rad'])) for r in turn),
                       nominal_reference_reached=any(abs(wrap(goal-r['reference_yaw_rad'])) < 1e-6 for r in turn),
                       hold_ack_elapsed_s=result.get('hold_ack_elapsed_s'),
                       post_interrupt_stationary_packets=result.get('post_interrupt_stationary_packets'),
                       coordinator_loss=result.get('coordinator_loss',{}).get('outcome'),
                       latent_wire_count=result['latent_wire_count'],
                       final_reason=result['turn_status']['reason'])
        assert summary['max_saved_reference_rate_deg_s'] <= 10.00001
        assert summary['max_saved_reference_lead_deg'] <= 5.00001
        if result['turn_success']:
            assert error <= 3 and summary['final_in_tolerance_span_s'] >= .5 and len(advancing) >= 2
        summaries.append(summary)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='outside lower center', ncol=3, fontsize=9)
    fig.suptitle('Native VLA-loop SONIC controls: 2 turns passed; 2 interrupted and held', fontsize=15)
    for suffix in ['png', 'pdf']:
        fig.savefig(args.root/f'native-turns.{suffix}', dpi=160)
    (args.root/'native-turn-analysis.json').write_text(json.dumps(dict(
        cases=summaries, limits_changed=False, hardware_acceptance=False,
        physics_plot_scope='Saved 10 Hz pose samples; no independent sub-tick settling acceptance.'),indent=2)+'\n')
    print(json.dumps(summaries))


if __name__ == '__main__':
    main()
