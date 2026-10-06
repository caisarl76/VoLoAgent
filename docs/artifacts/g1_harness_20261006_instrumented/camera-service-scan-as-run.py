import json
from pathlib import Path

matches = []
for process in Path('/proc').iterdir():
    if not process.name.isdigit():
        continue
    try:
        argv = process.joinpath('cmdline').read_bytes().decode().strip('\0').split('\0')
        programs = [a for a in argv[:4] if not a.startswith('-') and
                    ('camera' in Path(a).name.lower() or 'image_server' in Path(a).name.lower())]
        if not programs:
            continue
        flags = {}
        for i, arg in enumerate(argv[:-1]):
            if arg in ['--width', '--height', '--fps', '--resolution', '--config', '--camera-config']:
                flags[arg] = argv[i+1]
        matches.append(dict(pid=int(process.name), programs=programs,
                            cwd=str(process.joinpath('cwd').resolve()), selected_flags=flags))
    except (OSError, UnicodeError):
        continue
print(json.dumps(dict(camera_start_commands_sent=False, robot_commands_sent=False,
                     matching_processes=matches), indent=2))
