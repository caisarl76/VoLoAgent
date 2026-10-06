"""Read this experiment's stationary/translation fields from native wire bytes."""

import json
import math
import struct


def planner_fields(packet):
    if not packet.startswith(b'planner') or len(packet) < 7+1280+36:
        raise ValueError('Truncated or non-planner packet')
    header = json.loads(packet[7:7+1280].rstrip(b'\0'))
    expected = [dict(name='mode', dtype='i32', shape=[1]),
                dict(name='movement', dtype='f32', shape=[3]),
                dict(name='facing', dtype='f32', shape=[3]),
                dict(name='speed', dtype='f32', shape=[1]),
                dict(name='height', dtype='f32', shape=[1])]
    if (header.get('v') != 1 or header.get('endian') != 'le' or header.get('count') != 1
            or header.get('fields', [])[:5] != expected):
        raise ValueError('Unexpected native planner schema')
    values = struct.unpack_from('<i8f', packet, 7+1280)
    if not all(math.isfinite(v) for v in values):
        raise ValueError('Nonfinite planner fields')
    return dict(mode=values[0], movement=list(values[1:4]), facing=list(values[4:7]),
                speed=values[7], height=values[8])


def stationary(fields):
    return fields['mode'] == 0 and fields['speed'] == 0 and all(v == 0 for v in fields['movement'])
