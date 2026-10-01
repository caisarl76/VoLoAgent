import hashlib
import importlib
import json
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]


def modules():
    assert importlib.util.find_spec('vlm_orchestrator.harness'), 'G1 harness package is missing'
    return (importlib.import_module('vlm_orchestrator.harness.g1.registry'),
            importlib.import_module('vlm_orchestrator.harness.g1.contract'))


def test_profile_preserves_trained_prompt():
    r, _ = modules()
    p = r.load_profile(ROOT / 'configs/g1/workstation.yaml')
    assert p.require_skill('bottle_to_right_table').prompt == 'pick drink bottle and place it on the right table'
    assert (p.policy_port, p.action_horizon, p.publish_rate) == (15558, 40, 50)
    with pytest.raises(ValueError):
        p.require_skill('invented_grasp')


def test_registry_digest_matches_raw_bytes():
    r, _ = modules()
    path = ROOT / 'configs/g1/workstation.yaml'
    assert r.load_profile(path).registry_sha256 == hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize('key,value', [('lease_s', 2.01), ('feedback_max_age_s', float('nan')), ('walk_max_speed_mps', .201)])
def test_profile_rejects_unsafe_limits(tmp_path, key, value):
    r, _ = modules()
    data = yaml.safe_load((ROOT / 'configs/g1/workstation.yaml').read_text())
    data['limits'][key] = value
    path = tmp_path / 'profile.yaml'
    path.write_text(yaml.safe_dump(data))
    with pytest.raises(ValueError):
        r.load_profile(path)


@pytest.mark.parametrize('method,params', [('start_manipulation', {'skill_id': 'bottle_to_right_table', 'prompt': 'grasp'}), ('walk_for', {'direction': 'forward', 'duration_s': 5.01, 'speed_mps': .2}), ('turn_by', {'angle_rad': float('nan'), 'rate_rps': .1}), ('unknown', {})])
def test_rpc_schema_rejects_unknown_or_nonfinite_values(method, params):
    _, c = modules()
    req = dict(version=1, request_id='r', runtime_id='boot', session_id='session', lease_id='lease', method=method, params=params)
    with pytest.raises(ValueError):
        c.decode_request(json.dumps(req).encode())


def test_request_requires_owner_context():
    _, c = modules()
    req = dict(version=1, request_id='r', runtime_id='boot', session_id=None, lease_id=None, method='start_manipulation', params={'skill_id': 'bottle_to_right_table'})
    with pytest.raises(ValueError):
        c.decode_request(json.dumps(req).encode())


def test_rpc_golden_fixture():
    _, c = modules()
    fixtures = json.loads((ROOT / 'tests/fixtures/g1_rpc_v1.json').read_text())
    for raw in fixtures['valid_requests']:
        assert c.decode_request(json.dumps(raw).encode()).method == raw['method']
    for raw in fixtures['invalid_requests']:
        with pytest.raises(ValueError):
            c.decode_request(json.dumps(raw).encode())


def test_response_rejects_bad_result_types():
    _, c = modules()
    with pytest.raises(ValueError):
        c.decode_response(b'{"request_id":"r","runtime_id":"b","result":{},"error":null}')
