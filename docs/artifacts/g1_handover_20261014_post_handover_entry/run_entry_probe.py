"""Ready-return probe that starts from a recorded post-handover entry pose.

Usage: run_entry_probe.py --entry source30_f420 --return direct|profile --gpu 1 --desk-front-m 0.40 --output DIR
The first reset_ready is the fixture pre-pose (entry_worker.py) with the desk
away (sim_entry_worker.py). After it completes, the desk is restored only if the
hand clears it; pre-pose.json records both. The probe's own result covers the
second reset_ready, the return under test.
"""
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
import time

from vlm_orchestrator.harness.g1.client import G1Client

ROOT = Path(__file__).parent
TRACE_WORKER = str(ROOT.parent / "g1_handover_20261014_tracking" / "native_worker.py")
SIM = str(ROOT.parent / "g1_handover_20261012_acceptance" / "sim_worker.py")
WORKER, SIM_WORKER = ROOT / "entry_worker.py", ROOT / "sim_entry_worker.py"
ORIGINAL_POPEN, ORIGINAL_REQUEST = subprocess.Popen, G1Client.request


def pop_option(name):
    index = sys.argv.index(name)
    value = sys.argv.pop(index + 1)
    sys.argv.pop(index)
    return value


class Popen(ORIGINAL_POPEN):
    def __init__(self, args, *rest, **kwargs):
        if isinstance(args, list):
            args = [str(WORKER) if a == TRACE_WORKER else str(SIM_WORKER) if a == SIM else a for a in args]
        super().__init__(args, *rest, **kwargs)


if __name__ == "__main__":
    os.environ["G1_DIAG_ENTRY"] = pop_option("--entry")
    os.environ["G1_DIAG_RETURN"] = pop_option("--return")
    output = Path(sys.argv[sys.argv.index("--output") + 1])
    pre_pose = {}

    def request(self, method, params, **kwargs):
        if method != "reset_ready" or pre_pose:
            return ORIGINAL_REQUEST(self, method, params, **kwargs)
        started = time.monotonic()
        pre_pose["execution"] = asdict(ORIGINAL_REQUEST(self, method, params, **kwargs).result)
        while True:  # Keep the lease alive until the pre-pose settles.
            ORIGINAL_REQUEST(self, "heartbeat", {"sequence": time.monotonic_ns()}, **kwargs)
            status = self.get_status()
            if status.phase != "RESETTING" or time.monotonic() - started > 20:
                break
            time.sleep(0.05)
        pre_pose.update(phase=status.phase, reason=status.reason, duration_s=time.monotonic() - started)
        if status.phase == "COMPLETED":
            (output / "desk-restore").touch()
            while not (output / "desk-restore.json").exists() and time.monotonic() - started < 25:
                ORIGINAL_REQUEST(self, "heartbeat", {"sequence": time.monotonic_ns()}, **kwargs)
                time.sleep(0.05)
            path = output / "desk-restore.json"
            pre_pose["desk"] = json.loads(path.read_text()) if path.exists() else None
        (output / "pre-pose.json").write_text(json.dumps(pre_pose, indent=2, default=str) + "\n")
        if status.phase != "COMPLETED" or not (pre_pose["desk"] or {}).get("restored"):
            raise RuntimeError(f"Pre-pose not usable: {status.phase} {status.reason} desk={pre_pose.get('desk')}")
        return ORIGINAL_REQUEST(self, method, params, **kwargs)

    G1Client.request = request
    subprocess.Popen = Popen
    source, sim_source = WORKER.read_bytes(), SIM_WORKER.read_bytes()
    try:
        runpy.run_path(str(ROOT.parent / "g1_handover_20261014_tracking/run_probe.py"), run_name="__main__")
    finally:
        path = output / "result.json"
        if path.exists():
            result = json.loads(path.read_text())
            result.update(entry=os.environ["G1_DIAG_ENTRY"], return_mode=os.environ["G1_DIAG_RETURN"],
                          pre_pose=pre_pose, entry_worker_sha256=hashlib.sha256(source).hexdigest(),
                          sim_entry_worker_sha256=hashlib.sha256(sim_source).hexdigest())
            path.write_text(json.dumps(result, indent=2) + "\n")
            (output / "entry-worker-as-run.py").write_bytes(source)
            (output / "sim-entry-worker-as-run.py").write_bytes(sim_source)
