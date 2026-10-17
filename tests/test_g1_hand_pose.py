import importlib
from dataclasses import asdict
import json

import pytest
import numpy as np


@pytest.mark.parametrize('joints,expected', [
    ([0.028, -0.012, -0.010, -0.011, 0.020, -0.005, 0.007], 'open'),
    ([0.028, -0.341, -0.538, 0.533, 1.188, 0.653, 0.795], 'closed'),
    ([0.028, -0.014, -0.027, 0.009, 0.453, 0.016, 0.101], 'partial'),
    ([0.0, 0.0, 0.0, 0.6, 0.6, 0.6, 0.6], 'partial'),
    ([0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'partial'),
    ([], 'unknown'), (None, 'unknown'), ([0.0] * 6, 'unknown'),
    ([float('nan')] + [0.0] * 6, 'unknown'),
    ([float('inf')] + [0.0] * 6, 'unknown'),
    ([False] + [0.0] * 6, 'unknown'), (['0'] * 7, 'unknown'),
    ([[0.0]] * 7, 'unknown'),
    ({i / 1000: 0 for i in range(7)}, 'unknown'),
    (set(i / 1000 for i in range(7)), 'unknown'),
    ([10 ** 1000] + [0.] * 6, 'unknown'),
])
def test_measured_dex3_pose(joints, expected):
    module = importlib.import_module('vlm_orchestrator.harness.g1.hand_pose')
    assert module.classify_right_hand(joints) == expected


def test_overflowing_integer_joint_cannot_be_open():
    module = importlib.import_module('vlm_orchestrator.harness.g1.hand_pose')
    q = np.array([np.iinfo(np.int64).min, 0, 0, 0, 0, 0, 0], dtype=np.int64)
    assert module.classify_right_hand(q) == 'unknown'


def test_legacy_observation_decodes_as_unknown_hand():
    from test_g1_monitor import snapshot
    from vlm_orchestrator.harness.g1.contract import decode_response
    frame = asdict(snapshot('legacy'))
    for field in ['right_hand_state', 'right_hand_sample_at', 'right_hand_sample_index']:
        del frame[field]
    response = decode_response(json.dumps(dict(request_id='r', runtime_id='boot',
                                               result=frame, error=None)).encode())
    assert response.result.right_hand_state == 'unknown'
    assert response.result.right_hand_sample_at is None


@pytest.mark.parametrize('field,value', [('right_hand_state', []),
                                        ('right_hand_state', 'holding'),
                                        ('right_hand_sample_index', -1),
                                        ('right_hand_sample_at', float('nan')),
                                        ('right_hand_sample_at', 10 ** 1000)])
def test_malformed_hand_metadata_rejected_by_rpc(field, value):
    from test_g1_monitor import snapshot
    from vlm_orchestrator.harness.g1.contract import decode_response
    frame = asdict(snapshot('invalid'))
    frame[field] = value
    with pytest.raises(ValueError):
        decode_response(json.dumps(dict(request_id='r', runtime_id='boot',
                                        result=frame, error=None)).encode())
