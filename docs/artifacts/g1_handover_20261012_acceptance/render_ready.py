"""Render saved actual simulation states at wall-time cadence; no new physics."""
import bisect
import json
from pathlib import Path
import subprocess

import cv2
import mujoco
import numpy as np

ROOT = Path(__file__).parent
RUN_NAME = 'ready-040-fixture'
MOTION_LABEL = 'Actual SONIC ready component'
CLEARANCE_LABEL = 'signed hand clearance'
CLEARANCE_FUNCTION = None


def main():
    run = ROOT/RUN_NAME
    events = [json.loads(s) for s in (run/'events.jsonl').read_text().splitlines()]
    begin = next(r['at'] for r in events if r['event']=='ready_begin')
    end = next(r['at'] for r in events if r['event']=='ready_terminal')+.8
    rows = [json.loads(s) for s in (run/'ground-truth.jsonl').read_text().splitlines()]
    times = [r['received_monotonic'] for r in rows]
    stamps = np.arange(begin,min(end,times[-1]),1/20)
    indices = [max(0,bisect.bisect_right(times,t)-1) for t in stamps]
    assert indices and np.all(np.diff(times)>0)
    model = mujoco.MjModel.from_xml_path(str(run/'scene.xml'))
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model,height=360,width=640)
    path = ROOT/'ready-return.avi'
    writer = cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*'MJPG'),20,(1280,360))
    assert writer.isOpened()
    selected = []
    try:
        for i,(stamp,index) in enumerate(zip(stamps,indices)):
            row = rows[index]
            data.qpos[:] = row['qpos'];data.qvel[:] = row['qvel']
            mujoco.mj_forward(model,data)
            panels = []
            for camera in ('handover_overview','handover_closeup'):
                renderer.update_scene(data,camera=camera)
                panels.append(cv2.cvtColor(renderer.render(),cv2.COLOR_RGB2BGR))
            frame = np.concatenate(panels,axis=1)
            cv2.rectangle(frame,(0,0),(1280,60),(20,20,20),-1)
            label = f'{MOTION_LABEL} | t={stamp-begin:.2f}s | no VLA manipulation'
            cv2.putText(frame,label,(10,23),cv2.FONT_HERSHEY_SIMPLEX,.58,(255,255,255),1,cv2.LINE_AA)
            clearance = (row['clearance']['distance_m'] if CLEARANCE_FUNCTION is None
                else CLEARANCE_FUNCTION(model,data))*1000
            label = f'Provisional 85 cm desk | {CLEARANCE_LABEL} {clearance:.2f} mm | support OFF'
            cv2.putText(frame,label,(10,47),cv2.FONT_HERSHEY_SIMPLEX,.52,
                (80,130,255) if clearance<=0 else (100,230,140),1,cv2.LINE_AA)
            writer.write(frame)
            if i in (0,len(stamps)//3,2*len(stamps)//3,len(stamps)-1):selected.append(frame)
    finally:
        renderer.close();writer.release()
    subprocess.run(['ffmpeg','-v','error','-y','-i',str(path),'-c:v','libx264','-crf','22',
        '-pix_fmt','yuv420p','-movflags','+faststart',str(ROOT/'ready-return.mp4')],check=True)
    cv2.imwrite(str(ROOT/'ready-return-preview.jpg'),np.concatenate(selected,axis=0))
    report = dict(frames=len(stamps),fps=20,duration_s=len(stamps)/20,
        rendering='Post-run mj_forward of saved measured qpos; zero-order hold at wall-time cadence, no resimulation.',
        max_sample_age_s=float(max(t-times[i] for t,i in zip(stamps,indices))),
        source_run=RUN_NAME,robot_connection=False,virtual_support_enabled=False,
        full_handover_demonstrated=False,hardware_acceptance=False)
    report['clearance_overlay']=CLEARANCE_LABEL
    (ROOT/'render.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
