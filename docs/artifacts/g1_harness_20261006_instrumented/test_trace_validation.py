import json
import pytest
from trace_validation import validate_controller_trace


def write_trace(path, rows):
    path.write_text(''.join(json.dumps(row) + '\n' for row in rows))


def metadata(**overrides):
    return dict(event='metadata', schema=1, plan_count=1, merge_count=0,
                control_count=1, **overrides)


def test_complete_trace_is_counted(tmp_path):
    path = tmp_path/'trace.jsonl'
    write_trace(path, [metadata(), {'event': 'plan'}, {'event': 'control'}])
    assert validate_controller_trace(path)['control_count'] == 1


@pytest.mark.parametrize('rows', [
    [], [metadata()],
    [metadata(), {'event': 'plan'}],
    [metadata(), {'event': 'plan'}, {'event': 'plan'}],
    [metadata(), {'event': 'plan'}, {'event': 'unknown'}],
    [metadata(), {'event': []}, {'event': 'control'}],
    [metadata(), {'event': {}}, {'event': 'control'}],
    [metadata(), {'event': 'plan'}, {'event': 'control'}, {'event': 'control'}],
])
def test_incomplete_or_inconsistent_trace_is_rejected(tmp_path, rows):
    path = tmp_path/'trace.jsonl'
    write_trace(path, rows)
    with pytest.raises(ValueError):
        validate_controller_trace(path)


def test_partial_json_record_is_rejected(tmp_path):
    path = tmp_path/'trace.jsonl'
    write_trace(path, [metadata(), {'event': 'plan'}])
    with path.open('a') as out:
        out.write('{"event":"control"')
    with pytest.raises(ValueError):
        validate_controller_trace(path)
