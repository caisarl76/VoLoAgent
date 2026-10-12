"""CPU FK/reference diagnostics, never a controller or learned policy rollout."""
from pathlib import Path
import hashlib
import json
import os
import struct
import sys
import xml.etree.ElementTree as ET
import numpy as np
import mujoco

OUT = Path(__file__).parent
G = Path('/home/jihun/work/VoLoAgent/.worktrees/g1-harness-native')
MODEL = G / 'gear_sonic_deploy/g1/g1_29dof_with_hand.xml'
READY = OUT.parents[1] / 'g1_handover_20261009/ready-pose-candidate.json'
z = np.load(OUT / 'measured-extract.npz')
names = z['names'].tolist()
m = mujoco.MjModel.from_xml_path(str(MODEL))
d = mujoco.MjData(m)
jointids = np.array([m.joint(n).id for n in names])
addresses = m.jnt_qposadr[jointids]
assert len(set(addresses)) == 43
body = m.body('right_wrist_yaw_link').id
def angle(x):
    return float(np.degrees(np.arccos(np.clip(x, -1, 1))))
def fk(q, root):
    d.qpos[:] = m.qpos0
    d.qpos[:3] = 0  # root-centered frame, no measured root translation exists
    d.qpos[3:7] = root / np.linalg.norm(root)
    d.qpos[addresses] = q
    mujoco.mj_forward(m, d)
    R = d.xmat[body].reshape(3, 3).copy()
    return R, d.xpos[body].copy(), d.xmat[m.body('pelvis').id].reshape(3,3).copy()

rootxml = ET.parse(MODEL).getroot()
hashes = json.loads((OUT / 'source-hashes.json').read_text())
for p in [MODEL, READY, G/'gear_sonic/utils/inference/standing_reset.py', G/'gear_sonic/utils/teleop/xr_upperbody_bridge.py']:
    hashes[str(p)] = hashlib.sha256(p.read_bytes()).hexdigest()
for asset in rootxml.findall('asset/mesh'):
    p = MODEL.parent / 'meshes' / asset.attrib['file']
    hashes[str(p)] = hashlib.sha256(p.read_bytes()).hexdigest()
(OUT/'source-hashes.json').write_text(json.dumps(hashes,indent=2)+'\n')
palm = MODEL.parent/'meshes/right_hand_palm_link.STL'
raw = palm.read_bytes()
ntri = struct.unpack_from('<I', raw, 80)[0]
tri = np.frombuffer(raw, dtype=np.dtype([('normal','<f4',(3,)),('vertices','<f4',(3,3)),('attr','<u2')]), offset=84, count=ntri)
verts=np.unique(tri['vertices'].reshape(-1,3),axis=0)
eigs,evec=np.linalg.eigh(np.cov(verts.T))
mapping = [{'state_index': i,'joint':n,'qpos_address':int(addresses[i]),'range_rad':m.jnt_range[jointids[i]].tolist()} for i,n in enumerate(names)]
samples={15:[300,325,350,365,375,385,395,405,415,420,425,431],16:[246,260,275,290,305,320,335,340,345,350,355,360,370,386],30:[280,300,320,340,360,370,380,390,400,410,415,420]}
records=[]
curves={}
for ep in (15,16,30):
    states=z[f'{ep}/observation.state']
    roots=z[f'{ep}/observation.root_orientation']
    grav=z[f'{ep}/observation.projected_gravity']
    series=[]
    for frame,(q,root,g) in enumerate(zip(states,roots,grav)):
        R,pos,Rbase=fk(q,root)
        roll=float(q[names.index('right_wrist_roll_joint')])
        # +Y is a mesh/hinge-supported semantic hypothesis, not calibrated contact normal.
        normal=R[:,1]
        predicted_g=Rbase.T@np.array([0.,0.,-1.])
        gravity_error=angle(np.dot(predicted_g,g/np.linalg.norm(g)))
        series.append([frame/50,angle(normal[2]),angle(np.sin(roll)),float(roll*180/np.pi),gravity_error])
        if frame in samples[ep]:
            records.append({'episode':ep,'source_frame':frame,'time_s':frame/50,
                'right_arm_and_hand_radians':dict(zip(names[29:],q[29:].tolist())),
                'wrist_position_root_origin_world_axes_m':pos.tolist(),
                'wrist_rotation_world_from_link':R.tolist(),
                'candidate_plus_y_normal_world':normal.tolist(),
                'candidate_plus_y_tilt_deg':angle(normal[2]),
                'candidate_minus_y_tilt_deg':angle(-normal[2]),
                'candidate_unsigned_plane_inclination_deg':angle(abs(normal[2])),
                'wrist_roll_only_tilt_deg':angle(np.sin(roll)),
                'projected_gravity_agreement_deg':gravity_error})
    curves[ep]=np.array(series)

ready=json.loads(READY.read_text())['joints_radians']
ready_checks=[]
paths=[]
for ep,frame in [(15,415),(16,386),(30,420)]:
    start=z[f'{ep}/observation.state'][frame].copy()
    root=z[f'{ep}/observation.root_orientation'][frame]
    target=start.copy()
    for name,value in ready.items(): target[names.index(name)]=value
    delta=target-start
    T=np.max(np.abs(delta))/.5
    trajectory=[]
    # Intended named-joint reference only; not the defective ready implementation.
    for t in np.arange(0,T+.52,.02):
        q=start+np.sign(delta)*np.minimum(np.abs(delta),.5*t)
        R,pos,_=fk(q,root)
        trajectory.append([float(t),*pos.tolist(),angle(R[2,1])])
    paths.append({'episode':ep,'start_frame':frame,'reference_time_wrist_xyz_and_tilt':trajectory})
    ready_checks.append({'episode':ep,'source_frame':frame,'empty_hand_confirmed':ep==15,
        'max_named_joint_displacement_rad':float(np.max(np.abs(delta))),
        'ideal_0p5_radps_reference_duration_s':float(T),
        'ideal_duration_plus_dwell_s':float(T+.5),
        'max_wrist_displacement_m':float(np.linalg.norm(np.array(trajectory)[:,1:4]-np.array(trajectory)[0,1:4],axis=1).max()),
        'ready_candidate_tilt_deg':trajectory[-1][-1]})

result={'artifact_type':'CPU MuJoCo measured-joint forward kinematics and named-joint return references; no dynamics, VLA, hardware or scene contact validation',
 'mujoco_version':mujoco.__version__, 'coordinate_convention':'SI units; metadata wxyz root quaternion normalized; root translation fixed at origin, world gravity -Z; full 43 joints mapped by exact name',
 'contact_plane_calibrated':False,
 'palm_plane_hypothesis':'+Y of right_wrist_yaw_link normal to XZ palm plane; palm mesh translation [0.0415,-0.003,0], identity rotation in XML; finger hinges +Z curl +X fingers toward +Y. Requires hardware contact-surface validation.',
 'mesh_evidence':{'raw_vertices_bounds_m':[verts.min(0).tolist(),verts.max(0).tolist()], 'extent_m':np.ptp(verts,axis=0).tolist(),'covariance_eigenvalues':eigs.tolist(),'smallest_eigenvector_unsigned':evec[:,0].tolist()},
 'joint_mapping':mapping, 'samples':records,
 'gravity_orientation_check':{ep:{'max_deg':float(c[:,-1].max()),'median_deg':float(np.median(c[:,-1]))} for ep,c in curves.items()},
 'ready_target_limits':[{ 'joint':n,'value':v,'within_xml_limits':bool(m.jnt_range[m.joint(n).id][0]<=v<=m.jnt_range[m.joint(n).id][1])} for n,v in ready.items()],
 'ready_reference_checks':ready_checks, 'ready_reference_paths':paths,
 'limitations':['No calibrated hand contact-plane transform, bottle/contact/friction model, root translation, desk or human geometry.', 'Successful reference episodes and incident bracket are manually sampled, not exact release/contact onset.', 'Measured angles are not motor targets; synthetic ideal tracking does not verify SONIC settling.', 'Ep16 final and ep30 final references only test hypothetical return geometry and cannot authorize return during human removal.']}
(OUT/'evidence.json').write_text(json.dumps(result,indent=2)+'\n')
os.environ.setdefault('MPLCONFIGDIR','/tmp/handover-palm-mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig,axs=plt.subplots(3,1,figsize=(10,9),sharex=True)
for ax,ep in zip(axs,(15,16,30)):
    c=curves[ep]
    ax.plot(c[:,0],c[:,1],label='Full FK: hypothesized +Y palm normal')
    ax.plot(c[:,0],c[:,2],'--',label='Wrist roll alone (all other rotation ignored)')
    if ep==16: ax.axvspan(335/50,361/50,color='red',alpha=.15,label='User-confirmed roll/rescue bracket')
    ax.set_ylabel('Tilt from +Z (deg)');ax.set_title(f'Episode {ep}: measured reference FK, no calibrated contact plane');ax.grid(alpha=.3);ax.legend(fontsize=8)
axs[-1].set_xlabel('Source time (s at 50 Hz)');axs[-1].set_xlim(4.8,8.65)
fig.tight_layout();fig.savefig(OUT/'palm-angle-comparison.png',dpi=160)
print(json.dumps({'mesh':result['mesh_evidence'],'gravity':result['gravity_orientation_check'],'selected':[{k:r[k] for k in ['episode','source_frame','candidate_plus_y_tilt_deg','wrist_roll_only_tilt_deg']} for r in records if r['source_frame'] in [335,350,365,375,415]],'ready':ready_checks},indent=2))
