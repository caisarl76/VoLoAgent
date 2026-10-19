"""Check the saved live false-success frames against the corrected criterion."""

import argparse
from dataclasses import asdict
import hashlib
import importlib.util
import json
from pathlib import Path
import shlex
import shutil
import time

from openai import OpenAI

from vlm_orchestrator.harness.g1.contract import Execution
from vlm_orchestrator.harness.g1.monitor import G1CompletionMonitor
from vlm_orchestrator.harness.g1.registry import load_profile
from vlm_orchestrator.vlm.api import chat_create, parse_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mission', type=Path, required=True)
    parser.add_argument('--profile', type=Path, required=True)
    parser.add_argument('--env-file', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    api_key = None
    for line in args.env_file.read_text().splitlines():
        name, separator, value = line.strip().removeprefix('export ').partition('=')
        if separator and name.strip() == 'GENON_API_KEY':
            api_key = shlex.split(value, comments=True)[0]
    if not api_key:
        raise ValueError('GENON_API_KEY required')
    api = OpenAI(api_key=api_key,base_url='https://api.genon.ai/v1',timeout=10,max_retries=0)
    original = Path(__file__).parents[1]/'g1_harness_20261001/evaluate_vision_cases.py'
    spec = importlib.util.spec_from_file_location('frame_eval', original)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    rows = [json.loads(s) for s in (args.mission/'events.jsonl').read_text().splitlines()]
    frames = [Path(row['path']) for row in rows if row['event']=='monitor_frame']
    def vision(system, content):
        return chat_create(api,model='openai/gpt-5.6-sol',temperature=0,
                           messages=[{'role':'system','content':system},{'role':'user','content':content}]).choices[0].message.content
    profile = load_profile(args.profile)
    monitor = G1CompletionMonitor(profile.require_skill('bottle_to_right_table'),vision,time.monotonic,profile.limits)
    execution = Execution('offline-live-replay','negative-source-table','bottle_to_right_table','MANIPULATING',1,time.monotonic(),None)
    monitor.begin(execution,helper.snapshot(frames[0].read_bytes(),execution,0))
    report = {'profile_sha256':profile.registry_sha256,'criterion':monitor.skill.completion_criteria,
              'robot_connection':False,'model':'openai/gpt-5.6-sol','base_url':'https://api.genon.ai/v1',
              'label_source':'Live MuJoCo bottle position and source_table contact; all three frames negative',
              'initial_reference':'First saved monitor frame; the original pre-start snapshot was not persisted',
              'cases':[]}
    for index,path in enumerate(frames):
        shutil.copy2(path,args.output/path.name)
        decision = monitor.check(helper.snapshot(path.read_bytes(),execution,index+1))
        response = parse_json(monitor.raw_response or '')
        claim = isinstance(response,dict) and response.get('status')=='complete' and response.get('action')=='next'
        case = {'frame':path.name,'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                'expected_complete':False,'frame_complete':claim,'decision':asdict(decision),'raw_response':monitor.raw_response}
        report['cases'].append(case)
        (args.output/'result.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps({'frame':index,'frame_complete':claim,'outcome':decision.outcome}),flush=True)
    report['summary'] = {'total':len(frames),'false_complete':sum(row['frame_complete'] for row in report['cases']),
                         'unavailable':sum(row['decision']['outcome']=='unavailable' for row in report['cases'])}
    (args.output/'result.json').write_text(json.dumps(report,indent=2)+'\n')
    return 1 if report['summary']['false_complete'] or report['summary']['unavailable'] else 0


if __name__ == '__main__':
    raise SystemExit(main())
