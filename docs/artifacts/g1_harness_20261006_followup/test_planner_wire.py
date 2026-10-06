"""A malformed or translating command cannot certify a stationary wire window."""

import json
import math
import struct

import pytest

from planner_wire import planner_fields, stationary


def packet(*, speed=0., movement=(0., 0., 0.), mode=0, version=1):
    header = dict(v=version, endian='le', count=1, fields=[
        dict(name='mode', dtype='i32', shape=[1]), dict(name='movement', dtype='f32', shape=[3]),
        dict(name='facing', dtype='f32', shape=[3]), dict(name='speed', dtype='f32', shape=[1]),
        dict(name='height', dtype='f32', shape=[1])])
    return b'planner'+json.dumps(header).encode().ljust(1280, b'\0')+struct.pack('<i8f', mode, *movement, 1., 0., 0., speed, .8)


def test_zero_translation_is_stationary():
    assert stationary(planner_fields(packet()))


@pytest.mark.parametrize('params', [dict(speed=.2), dict(movement=(1., 0., 0.)), dict(mode=1)])
def test_translation_or_locomotion_mode_is_not_stationary(params):
    assert not stationary(planner_fields(packet(**params)))


@pytest.mark.parametrize('case', ['truncated', 'wrong_topic', 'wrong_version', 'nonfinite'])
def test_invalid_wire_is_rejected(case):
    wire = packet()
    if case == 'truncated':
        wire = wire[:-1]
    elif case == 'wrong_topic':
        wire = b'pose'+wire[7:]
    elif case == 'wrong_version':
        wire = packet(version=2)
    elif case == 'nonfinite':
        wire = packet(speed=math.nan)
    with pytest.raises(ValueError):
        planner_fields(wire)
