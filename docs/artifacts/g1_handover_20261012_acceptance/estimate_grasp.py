"""Estimate grasp reach from recorded joints; no bottle pose or robot commands."""
import hashlib
import json
from pathlib import Path
import struct

import mujoco
import numpy as np

ROOT = Path(__file__).parent
BASE = ROOT.parent/'g1_handover_20261009_followup/palm-diagnostics'
MODEL = Path('/home/jihun/work/VoLoAgent/.worktrees/g1-harness-native/gear_sonic_deploy/g1/g1_29dof_with_hand.xml')


def main():
    z = np.load(BASE/'measured-extract.npz',allow_pickle=False)
    model = mujoco.MjModel.from_xml_path(str(MODEL))
    data = mujoco.MjData(model)
    addresses = np.array([model.joint(str(n)).qposadr[0] for n in z['names']])
    assert len(set(addresses)) == 43
    wrist = model.body('right_wrist_yaw_link').id
    torso = model.body('torso_link').id
    mesh = MODEL.parent/'meshes/right_hand_palm_link.STL'
    raw = mesh.read_bytes()
    triangles = np.frombuffer(raw,dtype=np.dtype([('n','<f4',(3,)),('v','<f4',(3,3)),('a','<u2')]),
        offset=84,count=struct.unpack_from('<I',raw,80)[0])
    vertices = triangles['v'].reshape(-1,3)
    bbox_center = (vertices.min(0)+vertices.max(0))/2
    # XML mesh mounting offset; this is a mesh centre, not measured bottle contact.
    local_center = np.array([.0415,-.003,0])+bbox_center
    rows, max_error, grasp_max_error = [], 0.0, 0.0
    for ep,interval in [(15,(180,201)),(30,(168,201))]:
        samples = []
        for frame,q in enumerate(z[f'{ep}/observation.state']):
            data.qpos[:] = model.qpos0
            data.qpos[:3] = 0
            data.qpos[3:7] = [1,0,0,0]
            data.qpos[addresses] = q
            mujoco.mj_forward(model,data)
            wp = data.xpos[wrist].copy()
            error = np.linalg.norm(wp-z[f'{ep}/observation.eef_state'][frame,7:10])
            max_error = max(max_error,float(error))
            if interval[0] <= frame < interval[1]:
                grasp_max_error = max(grasp_max_error,float(error))
                palm = wp+data.xmat[wrist].reshape(3,3)@local_center
                rt = data.xmat[torso].reshape(3,3).T
                samples.append(dict(frame=frame,wrist_pelvis_m=wp.tolist(),
                    wrist_torso_origin_m=(rt@(wp-data.xpos[torso])).tolist(),
                    palm_mesh_center_torso_origin_m=(rt@(palm-data.xpos[torso])).tolist()))
        summary = {}
        for key in ('wrist_pelvis_m','wrist_torso_origin_m','palm_mesh_center_torso_origin_m'):
            values = np.array([s[key] for s in samples])
            summary[key] = dict(median_xyz_m=np.median(values,0).tolist(),
                min_xyz_m=values.min(0).tolist(),max_xyz_m=values.max(0).tolist(),
                median_horizontal_distance_m=float(np.median(np.linalg.norm(values[:,:2],axis=1))))
        rows.append(dict(source_episode=ep,visual_grasp_bracket_half_open=list(interval),
            exact_contact_onset_labelled=False,summary=summary,samples=samples))
    assert grasp_max_error < 1e-5, f'Grasp-bracket FK disagrees with recorded EEF: {grasp_max_error}'
    report = dict(scope='Read-only grasp-bracket FK; useful scene prior, not measured bottle pose.',
        frame_convention='Pelvis/model torso-origin axes: +X forward, +Y left, +Z up; neutral floating base. Torso origin is not chest skin surface.',
        root_translation_available=False,world_bottle_position_measured=False,
        bottle_size_measured=False,bottle_mass_measured=False,
        max_all_frame_recorded_wrist_fk_error_m=max_error,
        max_grasp_bracket_recorded_wrist_fk_error_m=grasp_max_error,
        palm_proxy='Centre of mounted palm STL bounding box; bottle-to-hand offset unknown.',
        episodes=rows,inputs_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (BASE/'measured-extract.npz',MODEL,mesh)},
        limitations=['Grasp brackets selected from ego RGB; no exact tactile/contact annotation.',
            'Wrist positions are already FK-derived, so agreement verifies mapping, not independent physical metrology.',
            'Elsewhere in these episodes stored EEF and named-joint FK differ by up to 3.2 cm; their alignment is not established for the whole episode.',
            'Recorded images contain no depth stream; camera mounting and bottle contact transform remain uncalibrated.',
            'A grasp locates the hand; it does not uniquely locate bottle centre, desk edge, or bottle weight.'])
    (ROOT/'grasp-reach.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(max_fk_error_m=max_error,episodes=[dict(source_episode=r['source_episode'],
        summary=r['summary']) for r in rows]),indent=2))


if __name__ == '__main__':
    main()
