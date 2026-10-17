"""Offline analysis of the one instrumented loopback turn; no model/robot calls."""

import csv
import json
import math
from pathlib import Path
import statistics

from trace_validation import validate_controller_trace


ROOT = Path(__file__).resolve().parent


def yaw(q):
    w, x, y, z = q
    return math.degrees(math.atan2(2*(w*z+x*y), w*w+x*x-y*y-z*z))


def facing(v):
    return math.degrees(math.atan2(v[1], v[0]))


def wrap(degrees):
    return (degrees+180) % 360-180


def load(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def stats(values):
    return dict(min=min(values), median=statistics.median(values), max=max(values))


def main():
    run = ROOT/'native-negative-zero-run'
    metadata = validate_controller_trace(run/'planner-trace.jsonl')
    assert all(metadata[k+'_dropped'] == 0 for k in ['plan', 'merge', 'control'])
    records = load(run/'planner-trace.jsonl')
    native = load(run/'native-trace.jsonl')
    events = load(run/'events.jsonl')
    samples = [r for r in native if r['event'] == 'turn_sample']
    start, end = samples[0]['at'], samples[-1]['at']
    goal = math.degrees(samples[0]['goal_yaw_rad'])
    controls_all = [r for r in records if r['event'] == 'control']
    controls = [r for r in controls_all if start <= r['begin'] <= end]
    plans = [r for r in records if r['event'] == 'plan' and start <= r['begin'] <= end]
    merges = {r['generation']: r for r in records if r['event'] == 'merge'}
    wire = [r for r in events if r['event'] == 'action_wire' and r['topic'] == 'planner']
    assert all(not r['trajectory_truncated'] for r in plans + list(merges.values()))
    plan_rows = []
    for plan in plans:
        selected = [c for c in controls if c['generation'] == plan['generation']]
        assert selected
        heading = yaw(selected[0]['heading_quat'])
        reference_facing = facing(plan['facing'])
        raw_yaw = [yaw(plan['raw_qpos'][i+3:i+7]) for i in range(0, len(plan['raw_qpos']), 36)]
        resampled = [yaw(q) for q in plan['resampled_quat']]
        raw_hits = [i for i, y in enumerate(raw_yaw) if abs(wrap(y-reference_facing)) <= 1]
        hits = [i for i, y in enumerate(resampled) if abs(wrap(y-reference_facing)) <= 1]
        merge = merges[plan['generation']]
        generation_offset = merge['used_generation_frame'] - merge['previous_frame']
        packets = [r for r in wire if abs(r['at']-plan['begin']) < .2]
        wire_gap = min(max(abs(a-b) for a, b in zip(plan['facing'], r['fields']['facing'])) for r in packets)
        world_facing = wrap(reference_facing + heading)
        nearby = [s for s in samples if s['phase'] == 'TURNING' and 0 <= plan['begin']-s['at'] < .2]
        world_gap = min(abs(wrap(world_facing-math.degrees(s['reference_yaw_rad']))) for s in nearby)
        plan_rows.append(dict(generation=plan['generation'], t_s=plan['begin']-start,
                              facing_world_deg=world_facing,
                              raw_end_world_deg=wrap(raw_yaw[-1]+heading),
                              raw_min_world_deg=min(wrap(y+heading) for y in raw_yaw),
                              raw_max_world_deg=max(wrap(y+heading) for y in raw_yaw),
                              raw_closest_facing_gap_deg=min(abs(wrap(y-reference_facing)) for y in raw_yaw),
                              raw_first_within_1deg_s=raw_hits[0]/30 if raw_hits else None,
                              resampled_first_within_1deg_s=hits[0]/50 if hits else None,
                              generation_offset_frames=generation_offset,
                              unblended_hit_aligned_to_merged_s=(hits[0]+generation_offset)/50 if hits else None,
                              max_active_frame=max(c['frame'] for c in selected),
                              input_wire_max_component_gap=wire_gap,
                              recovered_world_facing_gap_deg=world_gap,
                              planning_ms=1000*(plan['end']-plan['begin']),
                              inference_ms=plan['inference_us']/1000,
                              merge_after_plan_end_ms=1000*(merges[plan['generation']]['at']-plan['end'])))
    control_rows = [dict(t_s=c['begin']-start, generation=c['generation'], tick=c['tick'],
                         frame=c['frame'], observation_window_frames=c['observation_window_frames'],
                         target_world_deg=yaw(c['target_quat']), measured_world_deg=yaw(c['measured_quat']),
                         heading_correction_deg=yaw(c['heading_quat']),
                         target_measured_gap_deg=abs(wrap(yaw(c['target_quat'])-yaw(c['measured_quat']))))
                    for c in controls]
    telemetry_gaps, tick_offsets = [], []
    for sample in samples:
        nearby = [c for c in controls_all if abs(c['telemetry_end']-sample['at']) < .1]
        pair = min(nearby, key=lambda c: max(abs(a-b) for a, b in zip(c['target_quat'], sample['controller_target_quaternion'])))
        telemetry_gaps.append(max(abs(a-b) for a, b in zip(pair['target_quat'], sample['controller_target_quaternion'])))
        tick_offsets.append(pair['tick']-sample['telemetry_index'])
    reached = [r for r in plan_rows if r['resampled_first_within_1deg_s'] is not None]
    summary = dict(
        scope='One native-loop simulation diagnostic, no checkpoint queries or hardware commands.',
        turn_window_s=end-start, goal_world_deg=goal, turn_samples=len(samples),
        turn_plan_count=len(plans), turn_control_count=len(controls), metadata=metadata,
        received_effective_facing_max_component_gap=max(max(abs(a-b) for a,b in zip(p['received_facing'],p['facing'])) for p in plans),
        wire_effective_facing_max_component_gap=max(r['input_wire_max_component_gap'] for r in plan_rows),
        recovered_world_facing_max_gap_deg=max(r['recovered_world_facing_gap_deg'] for r in plan_rows),
        selected_merge_max_quat_component_gap=max(max(abs(a-b) for a,b in zip(c['active_quat'],merges[c['generation']]['quat'][c['frame']])) for c in controls),
        native_telemetry_max_quat_component_gap=max(telemetry_gaps),
        observed_control_tick_minus_telemetry_index=sorted(set(tick_offsets)),
        generation_frame_mismatches=sum(m['published_generation_frame']!=m['used_generation_frame'] for m in merges.values() if not m['first']),
        raw_future_within_1deg_count=sum(r['raw_first_within_1deg_s'] is not None for r in plan_rows),
        raw_future_first_within_1deg_s=stats([r['raw_first_within_1deg_s'] for r in plan_rows if r['raw_first_within_1deg_s'] is not None]),
        resampled_future_within_1deg_count=len(reached),
        resampled_first_hit_beyond_max_active_frame_count=sum(r['unblended_hit_aligned_to_merged_s']>r['max_active_frame']/50 for r in reached),
        generated_to_merged_offset_frames=sorted(set(r['generation_offset_frames'] for r in plan_rows)),
        max_active_frame=max(c['frame'] for c in controls),
        observation_window_frames=sorted(set(c['observation_window_frames'] for c in controls)),
        raw_frames=sorted(set(p['raw_frames'] for p in plans)),
        resampled_frames=sorted(set(p['resampled_frames'] for p in plans)),
        merged_frames=sorted(set(m['frames'] for m in merges.values() if start <= m['at'] <= end)),
        best_active_target_goal_error_deg=min(abs(wrap(r['target_world_deg']-goal)) for r in control_rows),
        best_measured_goal_error_deg=min(abs(wrap(r['measured_world_deg']-goal)) for r in control_rows),
        target_measured_gap_deg=stats([r['target_measured_gap_deg'] for r in control_rows]),
        planning_ms=stats([r['planning_ms'] for r in plan_rows]),
        inference_ms=stats([r['inference_ms'] for r in plan_rows]),
        merge_after_plan_end_ms=stats([r['merge_after_plan_end_ms'] for r in plan_rows]),
        terminal_reason=samples[-1]['reason'],
        terminal_old_scalar_reference_lead_deg=math.degrees(samples[-1]['lead_rad']),
        terminal_required_retraction_rate_deg_s=math.degrees(samples[-1]['required_retraction_rate_rps']),
        limitations=['Future quaternions are predictions, not executed poses.',
                     'Encoder observes a lookahead window; active-frame counts do not describe all encoder input.',
                     'Generated hit indices are aligned by the recorded merge offset; blending may alter the actual merged hit.',
                     'Movement buffer timestamp is a local receipt/update identity; packets contain no sender sequence.',
                     'Initial phase timings are unavailable; planning/model durations bracket host operations, not CUDA kernel-only time.',
                     'telemetry_end brackets state publication, not the separate 500 Hz DDS motor writer.',
                     'One trace does not isolate generator/context, activation/blending and lower-controller causality.'])
    (ROOT/'findings.json').write_text(json.dumps(summary,indent=2)+'\n')
    for name, rows in [('plans.csv',plan_rows), ('controls.csv',control_rows)]:
        with (ROOT/name).open('w',newline='') as out:
            writer=csv.DictWriter(out,fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
    plot(plan_rows, control_rows, samples, start, goal)
    print(json.dumps(summary,indent=2))


def plot(plans, controls, samples, start, goal):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes=plt.subplots(3,1,figsize=(10,9),sharex=True,layout='constrained')
    t=[r['t_s'] for r in controls]
    for key,color,label in [('target_world_deg','#2563eb','Active root target'),('measured_world_deg','#111827','Measured heading')]:
        axes[0].plot(t,[r[key] for r in controls],color=color,label=label)
    axes[0].plot([s['at']-start for s in samples if s['phase']=='TURNING'],
                 [math.degrees(s['reference_yaw_rad']) for s in samples if s['phase']=='TURNING'],
                 color='#dc2626',label='Bounded requested facing')
    axes[0].axhline(goal,color='#6b7280',ls='--',label='Task goal (-15 deg from entry)')
    axes[0].set_title('SONIC turn: requested, selected and measured headings')
    axes[0].set_ylabel('World yaw (deg)'); axes[0].legend(fontsize=8,ncol=2)
    pt=[r['t_s'] for r in plans]
    axes[1].fill_between(pt,[r['raw_min_world_deg'] for r in plans],[r['raw_max_world_deg'] for r in plans],color='#f59e0b',alpha=.2,label='Range of future predicted headings')
    axes[1].plot(pt,[r['raw_end_world_deg'] for r in plans],color='#d97706',label='End of each future prediction')
    axes[1].plot(pt,[r['facing_world_deg'] for r in plans],color='#dc2626',label='Effective facing, mapped to world')
    axes[1].set_ylabel('World yaw (deg)'); axes[1].set_title('Future predictions at each replan (these poses are not current poses)')
    axes[1].legend(fontsize=8)
    hits=[r for r in plans if r['resampled_first_within_1deg_s'] is not None]
    axes[2].scatter([r['t_s'] for r in hits],[r['unblended_hit_aligned_to_merged_s'] for r in hits],s=13,color='#059669',label='Future 1 deg hit + merge offset (before blending)')
    axes[2].plot(pt,[r['max_active_frame']/50 for r in plans],color='#2563eb',label='Furthest active frame before replacement')
    axes[2].axhline(.9,color='#6b7280',ls=':',label='Recorded encoder window span: 45/50 s')
    axes[2].set_ylabel('Trajectory time (s)'); axes[2].set_xlabel('Seconds since first turn sample')
    axes[2].set_title('Motion playback and the farther-ahead target frames')
    axes[2].legend(fontsize=8)
    for ax in axes:
        ax.grid(alpha=.2); ax.axvline(samples[-1]['at']-start,color='#b45309',ls='--',alpha=.65)
    fig.savefig(ROOT/'turn-stages.png',dpi=180)
    fig.savefig(ROOT/'turn-stages.pdf')
    plt.close(fig)


if __name__=='__main__':
    main()
