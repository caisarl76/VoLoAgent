"""Reject partial controller traces before treating a diagnostic as captured."""

import json


def validate_controller_trace(path):
    with path.open() as stream:
        first = stream.readline()
        if not first:
            raise ValueError('Empty controller trace')
        metadata = json.loads(first)
        if not isinstance(metadata, dict) or metadata.get('event') != 'metadata' or metadata.get('schema') != 1:
            raise ValueError('Missing controller trace schema metadata')
        counts = dict(plan=0, merge=0, control=0)
        for line in stream:
            row = json.loads(line)
            event = row.get('event') if isinstance(row, dict) else None
            if event not in counts:
                raise ValueError('Unknown controller trace record')
            counts[event] += 1
        for event, actual in counts.items():
            expected = metadata.get(event + '_count')
            if type(expected) is not int or expected < 0 or actual != expected:
                raise ValueError(f'{event} records do not match metadata')
        return metadata
