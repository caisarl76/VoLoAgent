"""Native loopback simulation with fresh full bottle support/contact evidence."""

import argparse
import importlib.util
import json
from pathlib import Path
import sys
import time

from placement_truth import sample_bottle_truth


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scene", type=Path, required=True)
    args = parser.parse_args()
    original = Path(__file__).parents[1]/"g1_harness_20261001/sonic_sim_worker.py"
    spec = importlib.util.spec_from_file_location("original_sim_worker", original)
    worker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(worker)
    original_wrapper = worker.SimWrapper

    class SceneWrapper(original_wrapper):
        def __init__(self, robot, name, config, **kwargs):
            config["ROBOT_SCENE"] = str(args.scene)
            kwargs.update(offscreen=True, enable_image_publish=True)
            super().__init__(robot, name, config, **kwargs)
            env = self.sim.sim_env
            step = env.sim_step
            sampled_at = 0.0
            stream = (args.output/"bottle-ground-truth.jsonl").open("w")

            def observe_step():
                nonlocal sampled_at
                step()
                now = time.monotonic()
                if now < sampled_at:
                    return
                sampled_at = now+0.1
                stream.write(json.dumps(sample_bottle_truth(env.mj_model, env.mj_data, now=now))+'\n')
                stream.flush()

            env.sim_step = observe_step
            start = self.sim.start

            def live_start():
                self.sim.start_image_publish_subprocess(camera_port=11555)
                try:
                    start()
                finally:
                    stream.close()

            self.sim.start = live_start

    worker.SimWrapper = SceneWrapper
    sys.argv = [sys.argv[0], "--output", str(args.output)]
    worker.main()


if __name__ == '__main__':
    main()
