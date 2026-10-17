"""Historical image relay; use transfer_handover.sh for direct input transfers."""
import argparse
import json
import shlex
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).parent
IMAGE = 'sha256:3d3fb48f355e59780d86201b61ea7e0251029166b61403bc70d4c6839a2b6c01'


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
        # The user took ownership of the direct, resumable rsync transfer.
        # The old host-side tar sources were container-only /dataset symlinks.
        parser.error('Input relay retired: run transfer_handover.sh on h100 (.173)')
    (ROOT/f'staged-{args.part}.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)
