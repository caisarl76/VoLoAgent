"""Independent comparison of saved direct and staged native physics samples."""
import json
import os
from pathlib import Path
import sys

import mujoco
import numpy as np
import yaml

from geometry import closest_bounds_gap

ROOT=Path(__file__).parent
sys.path.insert(0,'/home/jihun/work/.enpire-runtime/g1-bottle-sim-20261006')
from grounded_startup import grounded_window
os.environ.setdefault('MPLCONFIGDIR','/tmp/handover-lift-mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def analyze(run):
    result=json.loads((run/'result.json').read_text())
    events=[json.loads(s) for s in (run/'events.jsonl').read_text().splitlines()]
    begin=next(r['at'] for r in events if r['event']=='ready_begin')
    terminal=next(r['at'] for r in events if r['event']=='ready_terminal')
    samples=[json.loads(s) for s in (run/'ground-truth.jsonl').read_text().splitlines()]
    samples=[r for r in samples if begin<=r['received_monotonic']<=terminal+.8]
    model=mujoco.MjModel.from_xml_path(str(run/'scene.xml'));data=mujoco.MjData(model)
    obstacles=['desk_top',*(f'desk_leg_{i}' for i in range(4))]
    target=np.array(list(yaml.safe_load((run/'simulation-profile.yaml').read_text())['handover']['ready_right_arm_joints'].values()))
    rows=[]
    for sample in samples:
        data.qpos[:]=sample['qpos'];data.qvel[:]=sample['qvel'];mujoco.mj_forward(model,data)
        contacts=[]
        for c in data.contact:
            a,b=int(c.geom1),int(c.geom2)
            for hand,obstacle in ((a,b),(b,a)):
                body=model.body(int(model.geom_bodyid[hand])).name or ''
                name=model.geom(obstacle).name
                if body.startswith(('right_hand','right_wrist','right_elbow')) and name in obstacles:
                    contacts.append(dict(actor=body,obstacle=name,distance_m=float(c.dist)))
        rows.append(dict(time_s=sample['received_monotonic']-begin,
            conservative_gap=closest_bounds_gap(model,data,obstacles),
            simulated_contacts=contacts,right_arm_max_error_rad=float(np.max(np.abs(np.array(sample['body_q'][22:29])-target)))))
    last_half=[r for r in rows if terminal-begin-.5<=r['time_s']<=terminal-begin]
    assert len(last_half)>=8
    trace=[json.loads(s) for s in (run/'native-trace.jsonl').read_text().splitlines()]
    commands=[r for r in trace if r['event']=='planner_conversion' and begin<=r['at']<=terminal+.8]
    assert commands and all(r['world']['speed']==0 and r['world']['movement']==[0,0,0] for r in commands)
    assert not any(r['event']=='unexpected_latent_publication_attempt' for r in trace)
    report=dict(terminal_phase=result['terminal_status']['phase'],duration_s=terminal-begin,
        final_ready_settling_passed=result['measured_ready_settling_passed'],
        last_pre_completion_0p5_s_max_right_arm_error_rad=max(r['right_arm_max_error_rad'] for r in last_half),
        minimum_conservative_sampled_desk_gap=min(rows,key=lambda r:r['conservative_gap']['gap_m'])['conservative_gap'],
        sampled_hand_or_forearm_desk_contact_states=sum(bool(r['simulated_contacts']) for r in rows),
        physics_samples=len(rows),grounded_standing_screen=grounded_window(samples),
        zero_speed_planner_packets=len(commands),latent_action_publications=0,
        owned_processes_stopped=result['owned_processes_stopped'],hardware_acceptance=False,
        scope='Saved measured simulation snapshots, about 20 Hz; no continuous collision certificate or physical qualification.',
        samples=rows)
    return report


def main():
    direct=analyze(ROOT.parent/'g1_handover_20261012_acceptance/ready-040-fixture')
    staged=analyze(ROOT/'lift-first-native')
    assert staged['final_ready_settling_passed'] and staged['grounded_standing_screen']['ready']
    assert staged['last_pre_completion_0p5_s_max_right_arm_error_rad']<=.05
    assert staged['minimum_conservative_sampled_desk_gap']['gap_m']>0
    assert staged['sampled_hand_or_forearm_desk_contact_states']==0
    stages=[json.loads(s) for s in (ROOT/'lift-first-native/ready-stages.jsonl').read_text().splitlines()]
    assert [r['stage'] for r in stages]==[0,1]
    report=dict(direct=direct,staged=staged,stages=stages,hardware_profile_changed=False,
        production_native_reset_changed=False,simulation_prototype_only=True)
    (ROOT/'native-comparison.json').write_text(json.dumps(report,indent=2)+'\n')
    fig,axes=plt.subplots(2,1,figsize=(10,7))
    for label,r in (('Previous direct return',direct),('Lift-first prototype',staged)):
        time=[s['time_s'] for s in r['samples']]
        axes[0].plot(time,[s['conservative_gap']['gap_m']*1000 for s in r['samples']],label=label)
        axes[1].plot(time,[s['right_arm_max_error_rad'] for s in r['samples']],label=label)
    axes[0].axhline(0,color='black',linestyle='--');axes[0].set_ylabel('Conservative sampled desk gap (mm)')
    axes[1].axhline(.05,color='black',linestyle='--',label='Final 0.05 rad tolerance')
    axes[1].set_ylabel('Error to final ready target (rad)');axes[1].set_xlabel('Time since ready request (s)')
    for ax in axes:ax.grid(alpha=.25);ax.legend()
    fig.suptitle('Actual SONIC/MuJoCo samples — provisional scene, simulation-only staged prototype')
    fig.tight_layout();fig.savefig(ROOT/'native-lift-comparison.png',dpi=150);plt.close(fig)
    print(json.dumps({name:{k:v for k,v in r.items() if k!='samples'} for name,r in (('direct',direct),('staged',staged))},indent=2))


if __name__=='__main__':main()
