"""Install raised-waypoint prototype only in the isolated simulation worker."""
import json
from pathlib import Path
import sys
import time

from gear_sonic.scripts import run_vla_inference as loop
from staged_reset import StagedReadyReset

ROOT=Path(__file__).parent
ORIGINAL=loop.StandingReset


def main():
    candidate=json.loads((ROOT/'lift-candidate.json').read_text())
    assert candidate['simulation_candidate_only'] and not candidate['geometry_validated_for_hardware']
    output=Path(sys.argv[sys.argv.index('--output')+1])
    def record(stage):
        with (output/'ready-stages.jsonl').open('a') as stream:
            stream.write(json.dumps(dict(at=time.monotonic(),stage=stage,
                label=['raise_with_current_fingers','final_ready_open_hand'][stage],
                simulation_prototype_only=True))+'\n')
    def factory(*a,**kw):
        if kw.get('right_arm_target') is None:return ORIGINAL(*a,**kw)
        record(0)
        return StagedReadyReset(*a,waypoints=candidate['right_arm_waypoints_rad'],on_stage=record,**kw)
    loop.StandingReset=factory
    sys.path.insert(0,str(ROOT.parent/'g1_harness_20261006_followup'))
    from native_locomotion_worker import main as native_main
    native_main()


if __name__=='__main__':main()
