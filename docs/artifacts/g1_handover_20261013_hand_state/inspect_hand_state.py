"""Recorded measured-joint classification and Qwen comparison; no API calls."""

import hashlib
import json
from pathlib import Path
import statistics

import numpy as np

from vlm_orchestrator.harness.g1.hand_pose import classify_right_hand, RIGHT_HAND_JOINT_NAMES

HERE = Path(__file__).resolve().parent
CACHE = HERE.parent / 'g1_handover_20261009_followup/palm-diagnostics/measured-extract.npz'


def main():
    z = np.load(CACHE, allow_pickle=False)
    columns = [z['names'].tolist().index(name) for name in RIGHT_HAND_JOINT_NAMES]
    phases = json.loads((HERE/'qwen-offer-cues/result.json').read_text())
    baseline = json.loads((HERE.parent/'g1_handover_20261013_qwen/phase-cases/result.json').read_text())
    diagnostic = json.loads((HERE/'qwen-phase-cases/result.json').read_text())
    rows = []
    for ep, frame in [(15,246), (15,300), (15,350), (15,365), (15,375),
                      (30,300), (30,360), (30,370), (30,390), (16,335), (16,350)]:
        q = z[f'{ep}/observation.state'][frame, columns]
        rows.append(dict(episode=ep, frame=frame, right_hand_rad=q.tolist(),
                         state=classify_right_hand(q)))
    assert all(r['state']=='open' for r in rows if (r['episode'], r['frame'])
               in {(15,365),(15,375),(30,370),(30,390)})
    assert all(r['state']=='closed' for r in rows if (r['episode'], r['frame'])
               in {(15,246),(15,300),(30,300)})
    for current, old in zip(phases['cases'], baseline['cases']):
        assert current['id']==old['id']
        assert (HERE/'qwen-offer-cues'/current['image']).read_bytes() == (
            HERE.parent/'g1_handover_20261013_qwen/phase-cases'/old['image']).read_bytes()
    for episode, info in phases['episodes'].items():
        assert info['sha256']==baseline['episodes'][episode]['sha256']
        assert hashlib.sha256(Path(info['video']).read_bytes()).hexdigest()==info['sha256']
    calls = [c['provider_call']['latency_s'] for c in phases['cases'] if c.get('provider_call')]
    output = dict(cache_sha256=hashlib.sha256(CACHE.read_bytes()).hexdigest(),
        native_joint_names=RIGHT_HAND_JOINT_NAMES, source_columns=columns,
        recorded_rows=rows, thresholds_are_provisional=True, robot_connection=False,
        synthetic_receipt_times=True, source_videos_and_current_jpegs_rechecked=True,
        held_out_positive_test=False,
        qwen_no_cue_summary=baseline['summary'],
        qwen_all_phase_cue_diagnostic_summary=diagnostic['summary'],
        qwen_offer_only_cue_summary=phases['summary'],
        qwen_offer_only_provider_latency_s=dict(min=min(calls),median=statistics.median(calls),max=max(calls)))
    (HERE/'classification-analysis.json').write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps({k:v for k,v in output.items() if k!='recorded_rows'},indent=2))


if __name__=='__main__':
    main()
