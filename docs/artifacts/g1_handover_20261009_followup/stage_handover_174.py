"""Stage the pinned image and minimal inference inputs between user-owned hosts."""
import argparse
import json
from pathlib import Path
import shlex
import subprocess
import time

ROOT = Path(__file__).parent
IMAGE = 'sha256:3d3fb48f355e59780d86201b61ea7e0251029166b61403bc70d4c6839a2b6c01'
RUN = '/mnt/data01/jhkim/gr00t_runs/handover_bottle_260930_subset_261002_n17_20k_gpu6_20261006'
STAGE = '/mnt/data01/jhkim/model_weight/volo-handover-smoke-20261012'


def ssh(host, command):
    return ['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10',host,shlex.join(command)]


def stream(source, target, log):
    with log.open('w') as output:
        sender = subprocess.Popen(source,stdout=subprocess.PIPE,stderr=output)
        receiver = subprocess.Popen(target,stdin=sender.stdout,stdout=output,stderr=output)
        sender.stdout.close()
        while receiver.poll() is None:
            print(json.dumps(dict(stage=log.stem,status='running')),flush=True)
            time.sleep(10)
        receiver_code = receiver.returncode
        sender_code = sender.wait()
    if sender_code or receiver_code:
        raise RuntimeError(f'Transfer failed: sender={sender_code}, receiver={receiver_code}; see {log}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('part',choices=['image','inputs'])
    args = parser.parse_args()
    if args.part=='image':
        # A completed Docker load can be reused after interruption.
        probe = subprocess.run(ssh('h100_174',['docker','image','inspect',IMAGE]),capture_output=True)
        if probe.returncode:
            stream(ssh('h100',['docker','save',IMAGE]),ssh('h100_174',['docker','load']),ROOT/'stage-image.log')
        result = subprocess.check_output(ssh('h100_174',['docker','image','inspect',IMAGE,'--format','{{.Id}}']),text=True).strip()
        assert result==IMAGE
        report = dict(image=IMAGE,verified=True)
    else:
        # Refuse an existing destination: partial files require explicit diagnosis.
        created = subprocess.run(ssh('h100_174',['mkdir',STAGE]),capture_output=True,text=True)
        if created.returncode:
            if 'Permission denied' not in created.stderr:
                raise RuntimeError(created.stderr)
            uid = int(subprocess.check_output(ssh('h100_174',['id','-u']),text=True))
            gid = int(subprocess.check_output(ssh('h100_174',['id','-g']),text=True))
            name = Path(STAGE).name
            # Create only this new directory on the existing Docker-managed disk.
            create = f'mkdir /stage/{name} && chown {uid}:{gid} /stage/{name}'
            subprocess.run(ssh('h100_174',['docker','run','--rm','--network','none',
                '--mount','type=bind,src=/mnt/data01/jhkim/model_weight,dst=/stage',
                '--entrypoint','/bin/bash','sha256:4cbe2a3f7fc68dbf24eafa2e861e00db5c2d5eda8330df6af6122bbcce46370d',
                '-c',create]),check=True)
        checkpoint='train/handover-bottle-261002-20k-gpu6/checkpoint-20000/'
        files=[checkpoint+n for n in ['config.json','processor_config.json','statistics.json',
            'embodiment_id.json','model.safetensors.index.json',
            'model-00001-of-00003.safetensors','model-00002-of-00003.safetensors',
            'model-00003-of-00003.safetensors']]
        files+=['dataset_view/meta']
        for ep in (10,19):
            files += [f'dataset_view/data/chunk-000/episode_{ep:06d}.parquet',
                f'dataset_view/videos/chunk-000/observation.images.ego_view/episode_{ep:06d}.mp4']
        stream(ssh('h100',['tar','--dereference','-C',RUN,'-cf','-',*files]),
               ssh('h100_174',['tar','-C',STAGE,'-xf','-']),ROOT/'stage-inputs.log')
        report = dict(stage=STAGE,files=files,source_host='h100',target_host='h100_174',
            inference_only=True,optimizer_or_secrets_copied=False)
    (ROOT/f'staged-{args.part}.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)
