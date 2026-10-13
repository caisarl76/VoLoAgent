"""Relocate provisional desk around saved states; geometry only, no resimulation."""
import json
from pathlib import Path

import mujoco
import numpy as np

from scene import build_scene, closest_clearance

ROOT = Path(__file__).parent


def main():
    run = ROOT/'ready-040-fixture'
    events = [json.loads(s) for s in (run/'events.jsonl').read_text().splitlines()]
    begin = next(r['at'] for r in events if r['event']=='ready_begin')
    end = next(r['at'] for r in events if r['event']=='ready_terminal')+.8
    measured = [json.loads(s) for s in (run/'ground-truth.jsonl').read_text().splitlines()]
    measured = [r for r in measured if begin<=r['received_monotonic']<=end]
    z = np.load(ROOT.parent/'g1_handover_20261009_followup/palm-diagnostics/measured-extract.npz')
    folder = ROOT/'geometry-sweep'
    folder.mkdir(exist_ok=False)
    rows=[]
    for front in (.15,.20,.25,.30,.35,.40,.45,.50):
        path = folder/f'scene-{int(front*100):03d}.xml'
        build_scene(path,desk_front_m=front)
        model = mujoco.MjModel.from_xml_path(str(path))
        data = mujoco.MjData(model)
        names = ['desk_top',*(f'desk_leg_{i}' for i in range(4)),'person_proxy_geom']
        clearances=[]
        for sample in measured:
            assert len(sample['qpos']) == model.nq
            data.qpos[:] = sample['qpos'];mujoco.mj_forward(model,data)
            clearances.append(closest_clearance(model,data,names))
        source_checks=[]
        addresses=[model.joint(str(n)).qposadr[0] for n in z['names']]
        for ep,frame in ((15,191),(15,200),(15,365),(15,415),(30,168),(30,200),(30,370)):
            data.qpos[:] = model.qpos0
            data.qpos[3:7] = [1,0,0,0]
            data.qpos[addresses] = z[f'{ep}/observation.state'][frame]
            mujoco.mj_forward(model,data)
            source_checks.append(dict(source_episode=ep,source_frame=frame,
                root_height_assumed_m=float(model.qpos0[2]),
                clearance=closest_clearance(model,data,names)))
        rows.append(dict(desk_front_m=front,
            saved_ready_states_minimum=min(clearances,key=lambda c:c['distance_m']),
            saved_ready_states_contact_or_overlap_samples=sum(c['distance_m']<=0 for c in clearances),
            saved_ready_states_samples=len(clearances),source_pose_checks=source_checks))
    report=dict(scope='Geometric sensitivity screen only; desk moved around the same recorded trajectory.',
        new_physics_trials=0,calibrated_scene=False,hardware_acceptance=False,
        source_pose_root_translation_measured=False,source_pose_root_orientation_assumed_neutral=True,
        limitation='Saved trajectory belongs to the 0.40 m scene; relocated scenes cannot predict dynamics, bottle contact or human-arm clearance.',
        rows=rows)
    (ROOT/'clearance-sweep.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps([dict(desk_front_m=r['desk_front_m'],minimum_m=r['saved_ready_states_minimum']['distance_m'],
        contact_samples=r['saved_ready_states_contact_or_overlap_samples']) for r in rows],indent=2))


if __name__ == '__main__':
    main()
