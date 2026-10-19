"""Bounded reference evaluation on h100 GPU 0; no server or robot connection."""
import argparse
import io
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import uuid

ROOT = Path(__file__).parent
PREVIOUS = ROOT.parent/'g1_handover_20261009_followup'
sys.path.insert(0, str(PREVIOUS))
from run_policy_smoke_h100 import DATASET, IMAGE, RUN, ssh


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'reference-eval')
    output = parser.parse_args().output.resolve()
    output.mkdir(exist_ok=False)
    gpu = subprocess.check_output(ssh(['nvidia-smi', '-i', '0', '--query-gpu=uuid,memory.free',
        '--format=csv,noheader,nounits']), text=True).strip().split(',')
    gpu_uuid, free = gpu[0].strip(), int(gpu[1])
    if free < 24000:
        raise RuntimeError('Allocated GPU has insufficient free memory; nothing launched')
    files = {'offline_policy_smoke.py':(PREVIOUS/'offline_policy_smoke.py').read_text(),
        'policy_reference_eval.py':(ROOT/'policy_reference_eval.py').read_text()}
    stage_code = ('import pathlib,tempfile,json,sys; p=pathlib.Path(tempfile.mkdtemp(prefix="volo-handover-reference-")); '
        '[(p/k).write_text(v) for k,v in json.load(sys.stdin).items()]; print(p)')
    remote = subprocess.check_output(ssh(['python3','-c',stage_code]), input=json.dumps(files), text=True).strip()
    name = 'volo-handover-reference-' + uuid.uuid4().hex[:12]
    command = ['docker','run','--rm','--name',name,'--network','none','--gpus',f'device={gpu_uuid}',
        '--shm-size','2g','--mount',f'type=bind,src={remote},dst=/eval',
        '--mount',f'type=bind,src={RUN},dst=/outputs,readonly',
        '--mount',f'type=bind,src={DATASET},dst=/dataset,readonly',
        '--mount','type=bind,src=/mnt/data01/huggingface,dst=/root/.cache/huggingface,readonly']
    for flag in ('HF_HUB_OFFLINE','TRANSFORMERS_OFFLINE','HF_DATASETS_OFFLINE','NO_ALBUMENTATIONS_UPDATE','PYTHONUNBUFFERED'):
        command += ['-e',flag+'=1']
    command += ['-e','OMP_NUM_THREADS=4','--entrypoint','/workspace/.venv/bin/python',IMAGE,
        '/eval/policy_reference_eval.py','--checkpoint',
        '/outputs/train/handover-bottle-261002-20k-gpu6/checkpoint-20000',
        '--dataset','/outputs/dataset_view','--output','/eval/output']
    invocation = dict(host='h100', gpu=0, image=IMAGE, container=name, remote_stage=remote,
        free_mib=free, command=command, robot_connection=False)
    (output/'invocation.json').write_text(json.dumps(invocation,indent=2)+'\n')
    print(json.dumps(dict(status='starting',gpu=0,log=str(output/'inference.log'))),flush=True)
    try:
        with (output/'inference.log').open('w') as log:
            result = subprocess.run(ssh(command),stdout=log,stderr=subprocess.STDOUT,timeout=600)
    except (subprocess.TimeoutExpired,KeyboardInterrupt):
        subprocess.run(ssh(['docker','stop','--time','10',name]),capture_output=True,timeout=30)
        raise
    invocation['exit_code'] = result.returncode
    (output/'invocation.json').write_text(json.dumps(invocation,indent=2)+'\n')
    if result.returncode:
        raise RuntimeError(f'Reference evaluation exited {result.returncode}; see {output / "inference.log"}')
    bundle = subprocess.check_output(ssh(['tar','-C',remote+'/output','-cf','-','.']))
    with tarfile.open(fileobj=io.BytesIO(bundle)) as archive:
        for member in archive.getmembers():
            if member.isdir():
                continue
            relative = Path(member.name)
            if not member.isfile() or len(relative.parts)!=1 or relative.suffix not in ('.json','.npz'):
                raise RuntimeError('Unexpected evaluation artifact')
            with archive.extractfile(member) as data:
                (output/relative.name).write_bytes(data.read())
    report = json.loads((output/'result.json').read_text())
    assert report['completed'] and report['samples_compared']==13
    print(json.dumps(dict(completed=True, samples_compared=13, result=str(output/'result.json'))),flush=True)


if __name__ == '__main__':
    main()
