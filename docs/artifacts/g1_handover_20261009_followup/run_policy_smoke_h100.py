"""Run four disconnected reference inferences on the user-allocated h100 GPU 0."""
import argparse
import io
import json
import shlex
import subprocess
import tarfile
import uuid
from pathlib import Path

ROOT = Path(__file__).parent
IMAGE = 'sha256:3d3fb48f355e59780d86201b61ea7e0251029166b61403bc70d4c6839a2b6c01'
RUN = '/mnt/data01/jhkim/gr00t_runs/handover_bottle_260930_subset_261002_n17_20k_gpu6_20261006'
DATASET = '/mnt/data01/jhkim/gr00t_runs/handover_bottle_260930_subset_261002_n17_gpu2_20261002/dataset'


def ssh(command):
    return ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10', 'h100', shlex.join(command)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    gpu = subprocess.check_output(ssh(['nvidia-smi', '-i', '0',
        '--query-gpu=uuid,memory.free,utilization.gpu', '--format=csv,noheader,nounits']), text=True).strip()
    gpu_uuid, memory_free, utilization = [part.strip() for part in gpu.split(',')]
    if int(memory_free) < 24000:
        raise RuntimeError(f'GPU 0 has only {memory_free} MiB free; no inference launched')
    stage_code = ('import pathlib,tempfile,sys; '
        'p=pathlib.Path(tempfile.mkdtemp(prefix="volo-handover-smoke-gpu0-")); '
        '(p/"offline_policy_smoke.py").write_text(sys.stdin.read()); print(p)')
    remote = subprocess.check_output(ssh(['python3', '-c', stage_code]),
        input=(ROOT/'offline_policy_smoke.py').read_text(), text=True).strip()
    name = 'volo-handover-smoke-gpu0-' + uuid.uuid4().hex[:12]
    invocation = dict(host='h100', physical_gpu=0, gpu_uuid=gpu_uuid,
        initial_free_mib=int(memory_free), initial_utilization_percent=int(utilization),
        image=IMAGE, container=name, remote_stage=remote, network='none',
        robot_connection=False, samples_are_training_references=True)
    (args.output/'invocation.json').write_text(json.dumps(invocation, indent=2)+'\n')
    command = ['docker', 'run', '--rm', '--name', name, '--network', 'none',
        '--gpus', f'device={gpu_uuid}', '--shm-size', '2g',
        '--mount', f'type=bind,src={remote},dst=/smoke',
        '--mount', f'type=bind,src={RUN},dst=/outputs,readonly',
        '--mount', f'type=bind,src={DATASET},dst=/dataset,readonly',
        '--mount', 'type=bind,src=/mnt/data01/huggingface,dst=/root/.cache/huggingface,readonly',
        '-e', 'HF_HUB_OFFLINE=1', '-e', 'TRANSFORMERS_OFFLINE=1', '-e', 'HF_DATASETS_OFFLINE=1',
        '-e', 'NO_ALBUMENTATIONS_UPDATE=1', '-e', 'PYTHONUNBUFFERED=1', '-e', 'OMP_NUM_THREADS=4',
        '--entrypoint', '/workspace/.venv/bin/python', IMAGE, '/smoke/offline_policy_smoke.py',
        '--checkpoint', '/outputs/train/handover-bottle-261002-20k-gpu6/checkpoint-20000',
        '--dataset', '/outputs/dataset_view', '--output', '/smoke/output', '--device', 'cuda:0']
    invocation['command'] = command
    (args.output/'invocation.json').write_text(json.dumps(invocation, indent=2)+'\n')
    print(json.dumps(dict(status='starting', host='h100', physical_gpu=0,
        log=str(args.output/'inference.log'), robot_connection=False)), flush=True)
    try:
        with (args.output/'inference.log').open('w') as log:
            result = subprocess.run(ssh(command), stdout=log, stderr=subprocess.STDOUT, timeout=600)
    except (subprocess.TimeoutExpired, KeyboardInterrupt):
        subprocess.run(ssh(['docker', 'stop', '--time', '10', name]), capture_output=True, timeout=30)
        raise
    invocation['exit_code'] = result.returncode
    (args.output/'invocation.json').write_text(json.dumps(invocation, indent=2)+'\n')
    if result.returncode:
        raise RuntimeError(f'Inference exited {result.returncode}; see {args.output / "inference.log"}')
    bundle = subprocess.check_output(ssh(['tar', '-C', remote+'/output', '-cf', '-', '.']))
    with tarfile.open(fileobj=io.BytesIO(bundle)) as archive:
        for member in archive.getmembers():
            if member.isdir():
                continue
            relative = Path(member.name)
            if not member.isfile() or len(relative.parts) != 1 or relative.suffix not in ('.json', '.npz'):
                raise RuntimeError(f'Unexpected smoke artifact: {member.name}')
            with archive.extractfile(member) as data:
                (args.output/relative.name).write_bytes(data.read())
    report = json.loads((args.output/'result.json').read_text())
    assert report['policy_inference_verified'] and len(report['samples']) == 4
    print(json.dumps(dict(status='passed', recorded_samples=4, result=str(args.output/'result.json'),
        physical_success_verified=False, robot_connection=False)), flush=True)


if __name__ == '__main__':
    main()
