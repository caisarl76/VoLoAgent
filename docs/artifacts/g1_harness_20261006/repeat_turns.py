"""Four sequential loopback turning controls with unchanged planner limits."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--native', type=Path, required=True)
    parser.add_argument('--profile', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    probe = Path(__file__).parents[1]/'g1_harness_20261002/sonic_turn_shadow_probe.py'
    cases = [('negative-zero', 0, -15), ('positive-zero', 0, 15),
             ('negative-wrap', -170, -15), ('positive-wrap', 170, 15)]
    results = []
    for name, yaw, angle in cases:
        output = args.output/name
        command = [str(args.native/'.venv/bin/python'), str(probe), '--native',str(args.native.resolve()),
                   '--profile',str(args.profile.resolve()),'--output',str(output.resolve()),'--gpu','0',
                   '--sonic-policy','sonic_v1_1','--release-band','--prepare-standing',
                   '--initial-yaw-deg',str(yaw),'--turn-angle-deg',str(angle),'--turn-rate-deg-s','10']
        with (args.output/f'{name}-driver.log').open('w') as log:
            completed = subprocess.run(command, stdout=log, stderr=log,
                                       env={**os.environ,'PYTHONPATH':str(args.native.resolve())})
        result = json.loads((output/'result.json').read_text()) if (output/'result.json').exists() else {}
        row = dict(case=name,command=command,exit_code=completed.returncode,
                   result=result,not_a_native_main_loop_test=True)
        results.append(row)
        (args.output/'results.json').write_text(json.dumps(dict(robot_actuated=False,dds_interface='lo',
            limits_changed=False,probe_sha256=hashlib.sha256(probe.read_bytes()).hexdigest(),cases=results),indent=2)+'\n')
        print(json.dumps(dict(case=name,exit_code=completed.returncode,gate=result.get('gate'),
                              outcome=result.get('outcome'),reason=result.get('reason'))),flush=True)


if __name__ == '__main__':
    main()
