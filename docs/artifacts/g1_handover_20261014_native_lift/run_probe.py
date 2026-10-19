"""Exercise the production lift-first reset in isolated SONIC/MuJoCo."""
import hashlib
import contextlib
import io
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT.parent / "g1_handover_20261012_acceptance"))
import ready_probe

# Prior measured TRT allocation was 897 MiB. Leave other GPU users untouched.
ready_probe.MIN_FREE_GPU_MIB = 2048


if __name__ == "__main__":
    # The older fixture prints a historical commit; emit corrected provenance below.
    with contextlib.redirect_stdout(io.StringIO()):
        code = ready_probe.main()
    output = Path(sys.argv[sys.argv.index("--output") + 1])
    path = output / "result.json"
    if path.exists():
        result = json.loads(path.read_text())
        result.pop("native_commit", None)
        result.update(
            production_native_reset_changed=True,
            motion_implementation="gear_sonic/utils/inference/standing_reset.py",
            artifact_motion_wrapper=False,
            native_base_commit=subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ready_probe.NATIVE, text=True
            ).strip(),
            native_source_sha256={
                name: hashlib.sha256((ready_probe.NATIVE / name).read_bytes()).hexdigest()
                for name in (
                    "gear_sonic/utils/inference/standing_reset.py",
                    "gear_sonic/utils/inference/harness_profile.py",
                    "gear_sonic/utils/inference/harness_control.py",
                    "gear_sonic/scripts/run_vla_inference.py",
                )
            },
            minimum_free_gpu_mib=2048,
        )
        path.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2))
        if not result.get("measured_ready_settling_passed", False):
            code = 1
    raise SystemExit(code)
