"""Summarize saved Qwen checks against historical Genon evidence; no API calls."""

import hashlib
import json
from pathlib import Path
import statistics


HERE = Path(__file__).resolve().parent
BASELINE = HERE.parent / 'g1_handover_20261009_followup'


def summarize(path):
    report = json.loads(path.read_text())
    cases = report['cases']
    times = [(c.get('provider_call') or {}).get('latency_s') for c in cases]
    times = [value for value in times if value is not None]
    confirmed = {(c['episode'], c.get('sequence', c['id']), c['phase'])
                 for c in cases if c['decision']['outcome'] == 'complete'}
    return {
        'file': str(path.relative_to(HERE.parent)),
        'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
        'model': report['model'],
        'request_options': report.get('request_options', {}),
        'cases': len(cases),
        'provider_calls': sum(c['provider_called'] for c in cases),
        'unavailable': sum(c['decision']['outcome'] == 'unavailable' for c in cases),
        'label_agreements': sum(c['matches_label'] for c in cases),
        'unique_confirmed_phases': len(confirmed),
        'false_positive_frames_against_review_labels': sum(
            c['frame_outcome'] == 'yes' and 'yes' not in c['acceptable_outcomes']
            for c in cases),
        'provider_latency_s': dict(min=min(times), median=statistics.median(times),
                                  max=max(times)) if times else None,
        'outcomes': {c['id']: dict(frame=c['frame_outcome'],
                                 decision=c['decision']['outcome'],
                                 matches_label=c['matches_label']) for c in cases},
    }


def main():
    files = {
        'qwen_phase': HERE / 'phase-cases/result.json',
        'genon_phase_historical': BASELINE / 'vision-compatible/result.json',
        'qwen_transfer': HERE / 'transfer-sequence/result.json',
        'genon_transfer_historical': BASELINE / 'transfer-after/result.json',
        'qwen_default_probe': HERE / 'default-probe/result.json',
        'qwen_capped_thinking_probe': HERE / 'capped-thinking-probe/result.json',
        'qwen_fast_probe': HERE / 'fast-probe/result.json',
    }
    output = {
        'robot_connection': False,
        'comparison_scope': 'Selected recorded frames; historical Genon timing, not an isolated concurrent benchmark.',
        'label_scope': 'User-confirmed episode outcomes with investigator-selected phase labels; frame 405 is an unresolved review boundary.',
        'reports': {name: summarize(path) for name, path in files.items()},
    }
    for qwen, genon in [('qwen_phase', 'genon_phase_historical'),
                        ('qwen_transfer', 'genon_transfer_historical')]:
        current, old = (json.loads(files[name].read_text()) for name in (qwen, genon))
        assert [(c['id'], c['phase'], c['initial_frame'], c['frame']) for c in current['cases']] == [
            (c['id'], c['phase'], c['initial_frame'], c['frame']) for c in old['cases']]
        assert current['deadline_s'] == old['deadline_s'] == 10.
        for episode, info in current['episodes'].items():
            assert info['sha256'] == old['episodes'][episode]['sha256']
            assert hashlib.sha256(Path(info['video']).read_bytes()).hexdigest() == info['sha256']
        for case in current['cases']:
            for index in [case['initial_frame'], case['frame']]:
                filename = f"ep{case['episode']}_f{index:06d}.jpg"
                assert (files[qwen].parent / 'frames' / filename).read_bytes() == (
                    files[genon].parent / 'frames' / filename).read_bytes()
    output['source_videos_and_frame_bytes_match'] = True
    (HERE / 'comparison.json').write_text(json.dumps(output, indent=2) + '\n')
    print(json.dumps({name: {key: value for key, value in summary.items()
                           if key not in {'outcomes', 'request_options'}}
                      for name, summary in output['reports'].items()}, indent=2))


if __name__ == '__main__':
    main()
