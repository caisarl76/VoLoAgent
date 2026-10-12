"""Recorded-image checks through HandoverMonitor; never connects to a robot."""

import argparse
import base64
from dataclasses import asdict, replace
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import time

from openai import OpenAI

from vlm_orchestrator.harness.g1.contract import Execution, ObservationSnapshot
from vlm_orchestrator.harness.g1.handover_monitor import HandoverMonitor
from vlm_orchestrator.harness.g1.registry import HarnessLimits
from vlm_orchestrator.vlm.api import chat_create, parse_json

SOURCE = Path('/home/jihun/work/GR00T-WholeBodyControl/outputs/handover_bottle_260930')


def extract(video, index, target):
    if not target.exists():
        subprocess.run(['ffmpeg', '-v', 'error', '-i', str(video), '-vf',
                        f'select=eq(n\\,{index})', '-frames:v', '1', str(target)], check=True)
    return target.read_bytes()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, default=Path(__file__).with_name('vision-cases.json'))
    parser.add_argument('--env-file', type=Path)
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--deadline', type=float, default=10.)
    parser.add_argument('--case', action='append')
    args = parser.parse_args()
    if args.deadline <= 0:
        parser.error('Deadline must be positive')
    manifest = json.loads(args.manifest.read_text())
    cases = manifest['cases']
    if args.case:
        if set(args.case) - {c['id'] for c in cases}:
            parser.error('Unknown case ID')
        cases = [c for c in cases if c['id'] in args.case]
    key = os.environ.get('GENON_API_KEY')
    if args.env_file:
        for line in args.env_file.read_text().splitlines():
            name, sep, value = line.strip().removeprefix('export ').partition('=')
            if sep and name.strip() == 'GENON_API_KEY':
                tokens = shlex.split(value, comments=True)
                if len(tokens) == 1:
                    key = tokens[0]
    if not args.prepare_only and not key:
        parser.error('GENON_API_KEY missing')
    args.output.mkdir(parents=True, exist_ok=True)
    frames = args.output / 'frames'
    frames.mkdir(exist_ok=True)
    report = dict(robot_connection=False, model='openai/gpt-5.6-sol',
                  base_url='https://api.genon.ai/v1', deadline_s=args.deadline,
                  production_deadline=args.deadline == 10.,
                  scope='Recorded timestamps; provider wall time is not real-time video replay.',
                  label_provenance=manifest['label_provenance'], episodes={}, cases=[])
    info = json.loads((SOURCE / 'meta/info.json').read_text())
    for episode in sorted({c['episode'] for c in cases}):
        video = SOURCE / info['video_path'].format(
            episode_chunk=episode // info['chunks_size'], episode_index=episode,
            video_key='observation.images.ego_view')
        report['episodes'][str(episode)] = dict(video=str(video),
            sha256=hashlib.sha256(video.read_bytes()).hexdigest())
    api = None if args.prepare_only else OpenAI(api_key=key, base_url=report['base_url'],
                                             timeout=args.deadline, max_retries=0)
    calls = []

    def provider(system, content):
        started = time.monotonic()
        record = dict(started_at=started)
        calls.append(record)
        try:
            result = chat_create(api, model=report['model'], messages=[
                dict(role='system', content=system), dict(role='user', content=content)])
            record['response_model'] = result.model
            record['usage'] = result.usage.model_dump() if result.usage else None
            return result.choices[0].message.content
        except Exception as exc:
            record['error_class'] = type(exc).__name__
            record['http_status'] = getattr(exc, 'status_code', None)
            body = getattr(exc, 'body', None)
            error = body.get('error', body) if isinstance(body, dict) else None
            if isinstance(error, dict) and isinstance(error.get('message'), str):
                # Save just the server message, never request headers or the key.
                record['error_message'] = error['message'].replace(key, '[REDACTED]')[:500]
            raise
        finally:
            record['latency_s'] = time.monotonic() - started

    sessions = {}
    for case in cases:
        episode = str(case['episode'])
        video = Path(report['episodes'][episode]['video'])
        paths = {i: frames / f'ep{episode}_f{i:06d}.jpg'
                 for i in (case['initial_frame'], case['frame'])}
        images = {i: extract(video, i, p) for i, p in paths.items()}
        record = {**case, 'image': str(paths[case['frame']].relative_to(args.output))}
        if api is not None:
            session = case.get('sequence', case['id'])
            if session not in sessions:
                execution = Execution('recorded-handover', session, 'bottle_handover',
                                      'MANIPULATING', 1, time.monotonic(), None)
                limits = replace(HarnessLimits(), monitor_deadline_s=args.deadline,
                                 decision_expiry_s=args.deadline)
                monitor = HandoverMonitor(provider, time.monotonic, limits)
                sessions[session] = execution, monitor
                new_session = True
            else:
                execution, monitor = sessions[session]
                new_session = False

            def snapshot(index):
                image = images[index]
                return ObservationSnapshot(execution.runtime_id, execution.execution_id,
                    execution.inference_epoch, 'ego_view', index / info['fps'],
                    hashlib.sha256(image).hexdigest(), time.monotonic(), 0.,
                    base64.b64encode(image).decode())

            if new_session:
                initial = snapshot(case['initial_frame'])
                monitor.begin_phase(case['phase'], 1, execution, initial, initial.received_at)
            elif monitor.phase != case['phase']:
                raise ValueError('A sequence must keep the same phase')
            if monitor.last_check is not None:
                time.sleep(max(0., monitor.limits.monitor_interval_s -
                               (time.monotonic() - monitor.last_check)))
            call_index = len(calls)
            decision = monitor.check(snapshot(case['frame']))
            invoked = len(calls) > call_index
            raw = monitor.raw_response if invoked else None
            parsed = parse_json(raw) if raw else None
            frame_outcome = parsed.get('outcome') if parsed else None
            record.update(decision=asdict(decision), raw_response=raw,
                          frame_outcome=frame_outcome, provider_called=invoked,
                          provider_call=calls[call_index] if invoked else None,
                          confirmation_streak=monitor.streak,
                          matches_label=(frame_outcome in case['acceptable_outcomes']
                              if frame_outcome is not None else
                              decision.outcome == 'failure' and 'failure' in case['acceptable_outcomes']),
                          latency_s=decision.decided_at - decision.captured_at)
        report['cases'].append(record)
        (args.output / 'result.json').write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps(dict(case=case['id'], decision=record.get('decision', {}).get('outcome'),
                              frame_outcome=record.get('frame_outcome'),
                              latency_s=record.get('latency_s'))), flush=True)
    if api is not None:
        report['summary'] = dict(total=len(cases),
            unavailable=sum(c['decision']['outcome'] == 'unavailable' for c in report['cases']),
            label_disagreements=sum(not c['matches_label'] for c in report['cases']),
            complete_decisions=sum(c['decision']['outcome'] == 'complete' for c in report['cases']),
            confirmed_phases=len({(c['episode'], c.get('sequence', c['id']), c['phase'])
                for c in report['cases'] if c['decision']['outcome'] == 'complete'}),
            false_positive_frames=sum(c['frame_outcome'] == 'yes' and
                                     'yes' not in c['acceptable_outcomes'] for c in report['cases']))
        (args.output / 'result.json').write_text(json.dumps(report, indent=2) + '\n')
    return 0 if args.prepare_only or not report['summary']['label_disagreements'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
