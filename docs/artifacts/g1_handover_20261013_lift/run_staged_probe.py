"""Bounded loopback SONIC simulation with the lift-before-reach fixture."""
from pathlib import Path
import json
import hashlib
import sys

ROOT=Path(__file__).parent
sys.path.insert(0,str(ROOT.parent/'g1_handover_20261012_acceptance'))
import ready_probe

ready_probe.NATIVE_WORKER=ROOT/'native_staged_worker.py'
# Prior successful logs show 897 MiB TRT allocation; retain 2 GiB free preflight
# for this bounded fixture. Stop only its own processes if startup runs out.
ready_probe.MIN_FREE_GPU_MIB=2048

if __name__=='__main__':
    code=ready_probe.main()
    output=Path(sys.argv[sys.argv.index('--output')+1])
    path=output/'result.json'
    if path.exists():
        result=json.loads(path.read_text())
        candidate=json.loads((ROOT/'lift-candidate.json').read_text())
        result.update(native_ready_wrapper='staged_reset.py simulation prototype',
            production_native_reset_changed=False,lift_source_episode=candidate['source_episode'],lift_source_frame=candidate['source_frame'],
            minimum_free_gpu_mib=2048,
            prototype_sha256={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
                for name in ('staged_reset.py','native_staged_worker.py','lift-candidate.json')})
        path.write_text(json.dumps(result,indent=2)+'\n')
    raise SystemExit(code)
