"""Run the paired-trace ready probe with the sim-side PD/torque recorder.

--gravity-ff-motors 15,16,... enables the simulation-only diagnostic feedforward.
"""
import hashlib
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys

ROOT = Path(__file__).parent
SIM = str(ROOT.parent / "g1_handover_20261012_acceptance" / "sim_worker.py")
WORKER = ROOT / "sim_torque_worker.py"
ORIGINAL_POPEN = subprocess.Popen


class Popen(ORIGINAL_POPEN):
    def __init__(self, args, *rest, **kwargs):
        if isinstance(args, list):
            args = [str(WORKER) if a == SIM else a for a in args]
        super().__init__(args, *rest, **kwargs)


if __name__ == "__main__":
    motors = ""
    if "--gravity-ff-motors" in sys.argv:
        index = sys.argv.index("--gravity-ff-motors")
        motors = sys.argv.pop(index + 1)
        sys.argv.pop(index)
    os.environ["G1_DIAG_GRAVITY_FF_MOTORS"] = motors
    output = Path(sys.argv[sys.argv.index("--output") + 1])
    source = WORKER.read_bytes()
    subprocess.Popen = Popen
    try:
        runpy.run_path(str(ROOT.parent / "g1_handover_20261014_tracking/run_probe.py"), run_name="__main__")
    finally:
        path = output / "result.json"
        if path.exists():
            result = json.loads(path.read_text())
            result.update(
                pd_torque_trace=True, sim_worker_sha256=hashlib.sha256(source).hexdigest(),
                diagnostic_gravity_feedforward_motors=[int(x) for x in motors.split(",") if x],
                physics_modified=bool(motors),
            )
            path.write_text(json.dumps(result, indent=2) + "\n")
            (output / "sim-worker-as-run.py").write_bytes(source)
