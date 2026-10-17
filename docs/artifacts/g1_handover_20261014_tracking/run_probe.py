"""Run the existing production-return probe with read-only paired tracing."""
import hashlib
import json
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT.parent / "g1_handover_20261012_acceptance"))
import ready_probe

ready_probe.NATIVE_WORKER = ROOT / "native_worker.py"


if __name__ == "__main__":
    if "--candidate-profile" in sys.argv:
        index = sys.argv.index("--candidate-profile")
        ready_probe.PROFILE_TEMPLATE = Path(sys.argv.pop(index + 1)).resolve()
        sys.argv.pop(index)
    output = Path(sys.argv[sys.argv.index("--output") + 1])
    worker_source = (ROOT / "native_worker.py").read_bytes()
    try:
        runpy.run_path(str(ROOT.parent / "g1_handover_20261014_native_lift/run_probe.py"),
                       run_name="__main__")
    finally:
        path = output / "result.json"
        if path.exists():
            result = json.loads(path.read_text())
            result.update(paired_reference_feedback_trace=True,
                          trace_worker_sha256=hashlib.sha256(worker_source).hexdigest())
            path.write_text(json.dumps(result, indent=2) + "\n")
            (output / "trace-worker-as-run.py").write_bytes(worker_source)
