"""Measured approach FK and ideal-tracking route comparison; no robot commands."""
import json
from itertools import product
import os
from pathlib import Path
import sys

import mujoco
import numpy as np
import yaml

ROOT = Path(__file__).parent
PREVIOUS = ROOT.parent/'g1_handover_20261012_acceptance'
NATIVE = Path('/home/jihun/work/VoLoAgent/.worktrees/g1-harness-native')
sys.path[:0] = [str(NATIVE),str(PREVIOUS)]
from gear_sonic.utils.inference.standing_reset import StandingReset
from gear_sonic.utils.teleop.xr_upperbody_bridge import G1_UPPER_BODY_JOINT_INDICES
from scene import closest_clearance
from geometry import closest_bounds_gap

os.environ.setdefault('MPLCONFIGDIR','/tmp/handover-lift-mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    z = np.load(ROOT.parent/'g1_handover_20261009_followup/palm-diagnostics/measured-extract.npz')
    model = mujoco.MjModel.from_xml_path(str(PREVIOUS/'ready-040-fixture/scene.xml'))
    data = mujoco.MjData(model)
    names = z['names'].tolist()
    addresses = np.array([model.joint(n).qposadr[0] for n in names])
    arm_addresses = addresses[29:36]
    hand_order = ('thumb_0','thumb_1','thumb_2','index_0','index_1','middle_0','middle_1')
    hand_addresses = {side:[model.joint(f'{side}_hand_{n}_joint').qposadr[0] for n in hand_order]
        for side in ('left','right')}
    body_addresses = np.concatenate([addresses[:22],addresses[29:36]])
    hand_geoms = []
    for g in range(model.ngeom):
        body = model.body(int(model.geom_bodyid[g])).name or ''
        if model.geom_contype[g] and (body.startswith('right_hand') or body=='right_wrist_yaw_link'):
            if model.geom_type[g]==mujoco.mjtGeom.mjGEOM_MESH:
                mid = model.geom_dataid[g]
                vertices = model.mesh_vert[model.mesh_vertadr[mid]:model.mesh_vertadr[mid]+model.mesh_vertnum[mid]]
            else:
                assert model.geom_type[g]==mujoco.mjtGeom.mjGEOM_BOX
                vertices = np.array(list(product((-1,1),repeat=3)))*model.geom_size[g]
            hand_geoms.append((g,vertices.copy()))
    assert hand_geoms

    def geometry():
        mujoco.mj_forward(model,data)
        points = np.concatenate([v@data.geom_xmat[g].reshape(3,3).T+data.geom_xpos[g] for g,v in hand_geoms])
        return dict(wrist_xyz_m=data.xpos[model.body('right_wrist_yaw_link').id].tolist(),
            hand_lowest_z_m=float(points[:,2].min()),hand_farthest_x_m=float(points[:,0].max()),
            desk_top_bounds_gap=closest_bounds_gap(model,data,['desk_top']),
            desk_top_clearance=closest_clearance(model,data,['desk_top']))

    approaches = {}
    for ep in (15,30,16):
        rows = []
        for frame in range(201):
            data.qpos[:] = model.qpos0
            data.qpos[:3] = [0,0,.787]
            data.qpos[3:7] = [1,0,0,0]
            data.qpos[addresses] = z[f'{ep}/observation.state'][frame]
            rows.append(dict(frame=frame,time_s=frame/50,
                right_arm_rad=z[f'{ep}/observation.state'][frame,29:36].tolist(),**geometry()))
        first_height = next((r['frame'] for r in rows if r['hand_lowest_z_m']>=.88),None)
        first_edge = next((r['frame'] for r in rows if r['hand_farthest_x_m']>=.40),None)
        approaches[str(ep)] = dict(assumed_pelvis_height_m=.787,neutral_root_orientation=True,
            whole_hand_first_3cm_above_assumed_desk_frame=first_height,
            hand_first_passes_assumed_desk_front_frame=first_edge,samples=rows)

    events=[json.loads(s) for s in (PREVIOUS/'ready-040-fixture/events.jsonl').read_text().splitlines()]
    begin=next(r['at'] for r in events if r['event']=='ready_begin')
    initial=next(json.loads(s) for s in (PREVIOUS/'ready-040-fixture/ground-truth.jsonl').read_text().splitlines()
        if json.loads(s)['received_monotonic']>=begin)
    initial_qpos=np.array(initial['qpos'])
    initial_state=dict(body_q=np.array(initial['body_q']),left_hand_q=np.array(initial['left_hand_q']),
        right_hand_q=np.array(initial['right_hand_q']),base_quat=np.array(initial_qpos[3:7]))
    final = list(yaml.safe_load((PREVIOUS/'ready-040-fixture/simulation-profile.yaml').read_text())['handover']['ready_right_arm_joints'].values())
    candidates=[('direct',None,None)]
    for ep in (15,30):
        for frame in ([70,80,90] if ep==15 else [55,60,65]):
            candidates.append((f'source{ep}_f{frame}',ep,frame))
    routes=[]
    for label,ep,frame in candidates:
        state={k:v.copy() for k,v in initial_state.items()}
        waypoints=[] if ep is None else [z[f'{ep}/observation.state'][frame,29:36].tolist()]
        target_list=waypoints+[final]
        rows=[];t=0.
        for stage,target in enumerate(target_list):
            hand_target=initial_state['right_hand_q'] if stage<len(waypoints) else np.zeros(7)
            reset=StandingReset(state,state['left_hand_q'],hand_target,right_arm_target=target)
            for _ in range(1000):
                command=reset.advance(state,.02)
                state['body_q'][G1_UPPER_BODY_JOINT_INDICES]=command.upper_body_position
                state['left_hand_q']=np.array(command.left_hand_position)
                state['right_hand_q']=np.array(command.right_hand_position)
                data.qpos[:]=initial_qpos
                data.qpos[body_addresses]=state['body_q']
                for side in ('left','right'):data.qpos[hand_addresses[side]]=state[f'{side}_hand_q']
                rows.append(dict(time_s=t,stage=stage,right_arm_rad=state['body_q'][22:29].tolist(),**geometry()))
                t+=.02
                if reset.is_settled(state,.00001,.01):break
            else:raise RuntimeError('Ideal reference did not reach waypoint')
        minimum=min(rows,key=lambda r:r['desk_top_clearance']['distance_m'])
        bound_min=min(rows,key=lambda r:r['desk_top_bounds_gap']['gap_m'])
        routes.append(dict(label=label,source_episode=ep,source_frame=frame,
            right_arm_waypoints_rad=waypoints,ideal_reference_duration_s=t,
            minimum_desk_top_clearance=minimum['desk_top_clearance'],
            minimum_conservative_desk_gap=bound_min['desk_top_bounds_gap'],
            nonpositive_desk_distance_samples=sum(r['desk_top_clearance']['distance_m']<=0 for r in rows),
            samples=rows))
    report=dict(scope='Source approach FK plus ideal measured-tracking comparison, no new dynamics or physical clearance qualification.',
        desk_height_assumed_m=.85,desk_front_assumed_m=.40,
        source_world_translation_measured=False,source_root_orientation_assumed_neutral=True,
        approaches=approaches,routes=routes,hand_model_unstable_warning=True,hardware_acceptance=False)
    (ROOT/'lift-analysis.json').write_text(json.dumps(report,indent=2)+'\n')
    fig,axes=plt.subplots(2,2,figsize=(11,8))
    for ep,color in ((15,'#2878b5'),(30,'#e08025')):
        rows=approaches[str(ep)]['samples'];time=[r['time_s'] for r in rows]
        axes[0,0].plot(time,[r['hand_lowest_z_m'] for r in rows],label=f'Source {ep}',color=color)
        axes[0,1].plot(time,[r['hand_farthest_x_m'] for r in rows],label=f'Source {ep}',color=color)
    axes[0,0].axhline(.85,color='black',linestyle='--',label='Assumed tabletop')
    axes[0,0].axhline(.88,color='gray',linestyle=':',label='3 cm geometric margin')
    axes[0,1].axhline(.40,color='black',linestyle='--',label='Assumed near desk edge')
    axes[0,0].set_ylabel('Lowest hand mesh point (m)');axes[0,1].set_ylabel('Farthest hand mesh point, forward (m)')
    for r in routes:
        axes[1,0].plot([s['time_s'] for s in r['samples']],
            [s['desk_top_bounds_gap']['gap_m']*1000 for s in r['samples']],label=r['label'])
    axes[1,0].axhline(0,color='black',linestyle='--');axes[1,0].set_ylabel('Ideal path conservative desk gap (mm)')
    best=max(routes[1:],key=lambda r:r['minimum_conservative_desk_gap']['gap_m'])
    for r in (routes[0],best):
        axes[1,1].plot([s['hand_farthest_x_m'] for s in r['samples']],
            [s['hand_lowest_z_m'] for s in r['samples']],label=r['label'])
    axes[1,1].axhline(.85,color='black',linestyle='--');axes[1,1].axvline(.40,color='black',linestyle=':')
    axes[1,1].set_xlabel('Hand forward extent (m)');axes[1,1].set_ylabel('Hand lowest point (m)')
    for ax in axes.flat:ax.grid(alpha=.25);ax.legend(fontsize=8)
    for ax in (axes[0,0],axes[0,1],axes[1,0]):ax.set_xlabel('Time (s)')
    fig.suptitle('Recorded lift-before-reach pattern; ideal candidate screening, provisional scene')
    fig.tight_layout();fig.savefig(ROOT/'lift-comparison.png',dpi=150);plt.close(fig)
    candidate=dict(simulation_candidate_only=True,geometry_validated_for_hardware=False,
        source_episode=best['source_episode'],source_frame=best['source_frame'],
        right_arm_joint_names=names[29:36],right_arm_waypoints_rad=best['right_arm_waypoints_rad'],
        final_ready_rad=final,open_empty_hand_before_lift=False,
        strategy='Raise with measured entry fingers, then open while approaching final ready.',
        ideal_min_conservative_desk_gap_m=best['minimum_conservative_desk_gap']['gap_m'])
    (ROOT/'lift-candidate.json').write_text(json.dumps(candidate,indent=2)+'\n')
    print(json.dumps(dict(approaches={ep:{k:v for k,v in a.items() if k!='samples'} for ep,a in approaches.items()},
        candidates=[{k:v for k,v in r.items() if k not in ('samples','right_arm_waypoints_rad')} for r in routes]),indent=2))


if __name__=='__main__':main()
