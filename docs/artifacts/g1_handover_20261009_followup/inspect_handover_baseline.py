"""Read-only SSH audit of the existing training output, not model inference."""
import hashlib
import json
from pathlib import Path
import shlex
import subprocess

REMOTE = r'''
import hashlib,json,pathlib
r=pathlib.Path('/mnt/data01/jhkim/gr00t_runs/handover_bottle_260930_subset_261002_n17_20k_gpu6_20261006')
def read(name): return json.loads((r/name).read_text())
info=read('dataset_view/meta/info.json')
train=r/'train/handover-bottle-261002-20k-gpu6'
checkpoint=train/'checkpoint-20000'
result=dict(host='h100',run=str(r),checkpoint=str(checkpoint),
    run_manifest=read('evidence/run_manifest.json'),
    loader_probe=read('evidence/native-loader-probe.json'),
    tasks=[json.loads(x) for x in (r/'dataset_view/meta/tasks.jsonl').read_text().splitlines() if x],
    subset_manifest=read('dataset_view/meta/subset_manifest.json'),
    dataset_info={k:info.get(k) for k in ('total_episodes','total_frames','fps','splits')},
    exit_codes={name:(r/name).read_text().strip() for name in ('train/exit','train.exit')},
    checkpoint_files={p.name:p.stat().st_size for p in checkpoint.iterdir() if p.is_file()},
    metadata_hashes={},model_inference_verified=False,robot_connection=False)
for name in ['evidence/run_manifest.json','evidence/native-loader-probe.json',
             'dataset_view/meta/tasks.jsonl','dataset_view/meta/subset_manifest.json']:
    result['metadata_hashes'][name]=hashlib.sha256((r/name).read_bytes()).hexdigest()
for p in checkpoint.glob('*.json'):
    result['metadata_hashes'][str(p.relative_to(r))]=hashlib.sha256(p.read_bytes()).hexdigest()
lines=(r/'train.log').read_text(errors='replace').splitlines()
result['training_summary']=[line for line in lines if 'train_runtime' in line or 'Training completed!' in line][-2:]
print(json.dumps(result))
'''

if __name__ == '__main__':
    result = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10',
                             'h100', 'python3 -c ' + shlex.quote(REMOTE)],
                            capture_output=True, text=True, check=True, timeout=60)
    report = json.loads(result.stdout)
    report['audit_script_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    target = Path(__file__).with_name('baseline-training-audit.json')
    target.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(dict(report=str(target), checkpoint=report['checkpoint'],
                         prompt=report['tasks'][0]['task'],
                         source16_excluded=16 in report['subset_manifest']['removed_episode_indices'],
                         model_inference_verified=False)))
