"""Independent saved-array and measured-physics checks; never sends commands."""
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np
import yaml

os.environ.setdefault('MPLCONFIGDIR','/tmp/handover-acceptance-mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent
sys.path.insert(0,'/home/jihun/work/.enpire-runtime/g1-bottle-sim-20261006')
from grounded_startup import grounded_window


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def references():
    folder = ROOT/'reference-eval'
    report = json.loads((folder/'result.json').read_text())
    assert report['completed'] and report['samples_compared'] == 13 and not report['held_out']
    checks = []
    for row in report['samples']:
        path = folder/f"source{row['source_episode']}_frame{row['source_frame']}.npz"
        z = np.load(path,allow_pickle=False)
        metrics = {}
        for key,width in [('motion_token',64),('left_hand_joints',7),('right_hand_joints',7)]:
            pred,truth = z['prediction_'+key],z['reference_'+key]
            assert pred.shape == truth.shape == (40,width)
            assert np.isfinite(pred).all() and np.isfinite(truth).all()
            error = pred-truth
            expected = row['actions'][key]
            for name,value in [('mae',np.mean(np.abs(error))),('rmse',np.sqrt(np.mean(error**2))),
                               ('max_abs_error',np.max(np.abs(error)))]:
                assert np.isclose(value,expected[name],atol=1e-6,rtol=1e-5),(path,key,name)
            np.testing.assert_allclose(pred[0],expected['first_prediction'],atol=1e-7)
            np.testing.assert_allclose(truth[0],expected['first_reference'],atol=1e-7)
            metrics[key] = {f'prefix_{n}_rmse':float(np.sqrt(np.mean(error[:n]**2))) for n in (1,5,10,40)}
        assert np.abs(z['prediction_motion_token']).max() <= 1.25
        checks.append(dict(file=path.name,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            source_episode=row['source_episode'],source_frame=row['source_frame'],metrics=metrics))
    fig,axes = plt.subplots(2,1,figsize=(9,6),sharex=True)
    transitions=[]
    for ax,(ep,frame) in zip(axes,[(15,325),(30,320)]):
        z = np.load(folder/f'source{ep}_frame{frame}.npz',allow_pickle=False)
        t = np.arange(40)/50
        pred = np.max(np.abs(z['prediction_right_hand_joints']),axis=1)
        truth = np.max(np.abs(z['reference_right_hand_joints']),axis=1)
        ax.step(t,truth,where='post',label='Recorded finger target',color='black')
        ax.plot(t,pred,label='VLA prediction',color='#2878b5')
        ax.set_ylabel('Max finger bend (rad)');ax.set_title(f'Source {ep}, query frame {frame} — training reference')
        ax.grid(alpha=.25);ax.legend()
        first_open=lambda v: None if not np.any(v<.1) else int(np.flatnonzero(v<.1)[0])
        transitions.append(dict(source_episode=ep,query_frame=frame,
            reference_first_open_future_offset=first_open(truth),prediction_first_open_future_offset=first_open(pred),
            threshold_max_abs_joint_rad=.1))
    axes[-1].set_xlabel('Forecast time from query (s at 50 Hz)')
    fig.suptitle('Saved forecasts only: no executed release or physical success measured',fontsize=11)
    fig.tight_layout();fig.savefig(ROOT/'finger-opening-comparison.png',dpi=150);plt.close(fig)
    result=dict(samples_checked=len(checks),arrays_checked=len(checks)*6,all_finite=True,
        shapes_and_report_metrics_verified=True,held_out=False,physical_success_verified=False,
        prefix_interpretation='Example forecast prefixes; not measured executed chunks or release timing.',
        transitions=transitions,artifacts=checks)
    (folder/'artifact-verification.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


def physics():
    run = ROOT/'ready-040-fixture'
    result = json.loads((run/'result.json').read_text())
    assert result['completed'] and result['measured_ready_settling_passed'] and result['owned_processes_stopped']
    events = read_rows(run/'events.jsonl')
    begin = next(r['at'] for r in events if r['event']=='ready_begin')
    terminal = next(r['at'] for r in events if r['event']=='ready_terminal')
    samples = [r for r in read_rows(run/'ground-truth.jsonl') if begin<=r['received_monotonic']<=terminal+.8]
    trace = read_rows(run/'native-trace.jsonl')
    commands = [r for r in trace if r['event']=='planner_conversion' and begin<=r['at']<=terminal+.8]
    assert commands and not any(r['event']=='unexpected_latent_publication_attempt' for r in trace)
    assert all(c['world']['speed']==0 and c['world']['movement']==[0,0,0] for c in commands)
    target = np.array(list(yaml.safe_load((run/'simulation-profile.yaml').read_text())['handover']['ready_right_arm_joints'].values()))
    measured = np.array([r['body_q'][22:29] for r in samples])
    error = np.max(np.abs(measured-target),axis=1)
    stamps = np.array([r['received_monotonic'] for r in samples])
    tail = (stamps>=terminal-.5) & (stamps<=terminal)
    # Independent ground truth uses separate 20 Hz samples; it does not replace native telemetry's dwell decision.
    assert len(samples)>=20 and tail.sum()>=10
    ground = grounded_window(samples)
    assert ground['ready'],ground
    clearance = min(samples,key=lambda r:r['clearance']['distance_m'])
    qpos = np.array([r['qpos'] for r in samples])
    report = dict(scope='Actual SONIC/native ready-return component from default standing; no VLA pickup/offer or person removal.',
        ready_duration_s=terminal-begin,terminal_phase=result['terminal_status']['phase'],
        native_settling_confirmed=True,ground_truth_tail_within_0p05_rad=bool(np.all(error[tail]<=.05)),
        max_right_arm_error_last_0p5_s_rad=float(error[tail].max()),
        final_right_arm_max_error_rad=float(error[-1]),
        minimum_geometric_clearance=clearance['clearance'],
        clearance_screen_passed=bool(clearance['clearance']['distance_m']>0),
        desk_contact_or_overlap_samples=sum(r['clearance']['distance_m']<=0 for r in samples),
        final_geometric_clearance=samples[-1]['clearance'],
        base_horizontal_displacement_from_reset_start_m=float(np.max(np.linalg.norm(qpos[:,:2]-qpos[0,:2],axis=1))),
        grounded_standing_screen=ground,stationary_planner_packets=len(commands),
        latent_action_publications=0,scripted_prewarm_only=True,readiness_gates_fixture_only=True,
        hardware_profile_changed=False,hardware_acceptance=False,calibrated_scene=False,
        limitations=['Starts from default standing, not a recorded post-handover pose.',
            'Scene distance and bottle geometry provisional; cylinder remains on the desk.',
            'Person proxy has no arms/hands; this cannot establish human clearance.',
            'Source hand model explicitly warns of simulation instability.'])
    (run/'independent-verification.json').write_text(json.dumps(report,indent=2)+'\n')
    fig,ax = plt.subplots(figsize=(9,4))
    ax.plot(stamps-begin,error,label='Measured right-arm maximum error')
    ax.axhline(.05,color='black',linestyle='--',label='0.05 rad tolerance')
    ax.axvline(terminal-begin,color='green',linestyle=':',label='Native COMPLETED')
    ax.set_xlabel('Time since ready request (s)');ax.set_ylabel('Joint error (rad)')
    ax.set_title('Actual SONIC ready return — simulation fixture, no VLA manipulation')
    ax.grid(alpha=.25);ax.legend();fig.tight_layout();fig.savefig(ROOT/'ready-tracking.png',dpi=150);plt.close(fig)
    return report


if __name__ == '__main__':
    refs = references()
    ready = physics()
    print(json.dumps(dict(reference_samples=refs['samples_checked'],reference_arrays=refs['arrays_checked'],
        forecast_transitions=refs['transitions'],ready=ready),indent=2))
