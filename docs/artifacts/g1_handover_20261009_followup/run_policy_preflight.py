"""Run only CPU metadata preflight in the existing training container over SSH."""
import json
from pathlib import Path
import shlex
import subprocess
import uuid

ROOT = Path(__file__).parent
CONTAINER = 'jihun_gr00t_n17_handover_bottle_261002_20k_gpu6_20261006'

if __name__ == '__main__':
    output = '/tmp/volo-handover-metadata-preflight-' + uuid.uuid4().hex
    command = ['docker','exec','-i',CONTAINER,'/workspace/.venv/bin/python','-',
        '--checkpoint','/outputs/train/handover-bottle-261002-20k-gpu6/checkpoint-20000',
        '--dataset','/outputs/dataset_view','--output',output,'--preflight-only']
    result = subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10',
        'h100',shlex.join(command)], input=(ROOT/'offline_policy_smoke.py').read_text(),
        capture_output=True,text=True,check=True,timeout=60)
    report = json.loads(result.stdout.splitlines()[-1])
    report.update(host='h100',container=CONTAINER,remote_temporary_output=output,
        preflight_only=True,checkpoint_loaded=False)
    (ROOT/'policy-metadata-preflight.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(preflight_only=True,checkpoint_loaded=False,robot_connection=False)))
