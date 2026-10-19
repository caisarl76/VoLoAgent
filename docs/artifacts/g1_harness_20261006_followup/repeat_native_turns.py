"""Complete three remaining native-loop turning controls sequentially."""

import argparse
import json
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--native', type=Path, required=True)
    parser.add_argument('--profile', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    driver = Path(__file__).with_name('native_locomotion_probe.py')
    cases = [('positive-zero', 0, 15), ('negative-wrap', -170, -15), ('positive-wrap', 170, 15)]
    results = []
    for name, yaw, angle in cases:
        command = [sys.executable, str(driver.resolve()), '--native', str(args.native.resolve()),
                   '--profile', str(args.profile.resolve()), '--output', str((args.output/name).resolve()),
                   '--initial-yaw-deg', str(yaw), '--turn-angle-deg', str(angle)]
        with (args.output/f'{name}-driver.log').open('w') as log:
            run = subprocess.run(command, stdout=log, stderr=log)
        path = args.output/name/'result.json'
        result = json.loads(path.read_text()) if path.exists() else dict(outcome='driver_error')
        results.append(dict(case=name,command=command,exit_code=run.returncode,result=result))
        passed = all(x['exit_code'] == 0 and x['result']['outcome'] == 'passed_control_checks' for x in results)
        (args.output/'batch-results.json').write_text(json.dumps(dict(
            robot_actuated=False, control_checks_passed=passed, cases=results),indent=2)+'\n')
        print(json.dumps(dict(case=name,outcome=result['outcome'],turning=result.get('turning'),
                              reason=result.get('reason'))),flush=True)
    return 0 if passed else 1


if __name__ == '__main__':
    raise SystemExit(main())
