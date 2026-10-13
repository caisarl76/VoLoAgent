"""Verify saved acceptance evidence, source hashes and disabled hardware profile."""
import ast
import hashlib
import json
from pathlib import Path
import re
import subprocess

import yaml

ROOT = Path(__file__).parent
REPO = ROOT.parents[2]


def main():
    syntax = []
    for path in ROOT.glob('*.py'):
        ast.parse(path.read_text(),filename=str(path));syntax.append(path.name)
    profile = REPO/'configs/g1/handover.example.yaml'
    p = yaml.safe_load(profile.read_text())
    assert p['handover']['checkpoint_verified'] is False and p['handover']['ready_pose_reviewed'] is False
    for name in ('handover.example.yaml','workstation.yaml'):
        current = REPO/'configs/g1'/name
        assert subprocess.check_output(['git','show',f'742f820:configs/g1/{name}'],cwd=REPO)==current.read_bytes()
    previous = json.loads((ROOT.parent/'g1_handover_20261009/numeric-audit.json').read_text())
    src = Path(previous['dataset_path'])
    for name,value in previous['source_hashes'].items():
        assert hashlib.sha256((src/name).read_bytes()).hexdigest()==value,name
    volo_log = (ROOT/'volo-tests.log').read_text()
    scene_log = (ROOT/'scene-tests.log').read_text()
    assert '709 passed, 19 skipped, 46 warnings' in volo_log
    assert re.search(r'6 passed in [0-9.]+s',scene_log)
    refs = json.loads((ROOT/'reference-eval/artifact-verification.json').read_text())
    ready = json.loads((ROOT/'ready-040-fixture/independent-verification.json').read_text())
    report = dict(date='2026-10-13',base_commit='742f820',native_commit='44f9bba',
        scope='Disconnected checkpoint comparison, grasp reach estimates and provisional native-ready simulation.',
        hardware_commands_sent=False,hardware_profiles_unchanged=True,
        hardware_example_sha256=hashlib.sha256(profile.read_bytes()).hexdigest(),
        hardware_checkpoint_verified=False,hardware_ready_pose_reviewed=False,
        tests=[dict(command='env PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest -q --tb=short',
                    passed=709,skipped=19,warnings=46,exit_code=0,log='volo-tests.log'),
            dict(command='env PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /home/jihun/work/VoLoAgent/.worktrees/g1-harness-native/.venv/bin/python -m pytest docs/artifacts/g1_handover_20261012_acceptance/test_scene.py -q --tb=short',
                    passed=6,exit_code=0,log='scene-tests.log')],
        setup_failures=['Unrelated ROS pytest plugin missing lark; disabled plugin autoload.',
            'Sandbox ZMQ socket test aborted; successful rerun outside sandbox.',
            'Busy local GPU 0 planner OOM; successful startup on idle local GPU 1.',
            'Unverified fixture correctly rejected; ready component uses explicitly labelled simulation gates.',
            'OSMesa unavailable; successful offline EGL rendering on GPU 1.'],
        syntax_checked_files=syntax,relative_links_checked=True,
        reference_observations_checked=refs['samples_checked'],prediction_reference_arrays_checked=refs['arrays_checked'],
        policy_rollout_run=False,ready_tracking_passed=ready['native_settling_confirmed'],
        ready_clearance_screen_passed=ready['clearance_screen_passed'],scene_calibrated=False,
        robot_balance_qualified=False,stable_bottle_support_validated=False,full_handover_demonstrated=False,
        training_performed=False,source_files_rehashed=len(previous['source_hashes']),source_data_changed=False,
        assets_sha256={str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest()
            for f in ROOT.rglob('*') if f.is_file() and f.suffix in ('.mp4','.npz','.png','.jpg')},
        independent_review='pending')
    existing = ROOT/'verification.json'
    if existing.exists():
        report['independent_review'] = json.loads(existing.read_text()).get('independent_review','pending')
    existing.write_text(json.dumps(report,indent=2)+'\n')
    for path in (ROOT/'README.md',ROOT/'COLLECTION.md',ROOT/'WORK.md'):
        for target in re.findall(r'\]\(([^)]+)\)',path.read_text()):
            if not target.startswith(('http:','https:')):
                assert (path.parent/target).exists(),(path,target)
    print(json.dumps(dict(syntax_files=len(syntax),source_files=len(previous['source_hashes']),
        source_unchanged=True,hardware_profiles_unchanged=True,relative_links=True,
        saved_assets_hashed=len(report['assets_sha256'])),indent=2))


if __name__=='__main__':
    main()
