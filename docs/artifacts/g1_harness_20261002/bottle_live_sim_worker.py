"""Run the existing loopback simulator with the sample scene/live camera."""

import argparse
import importlib.util
import json
from pathlib import Path
import sys
import time

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
            bottle = env.mj_model.body("task_bottle").id
            geom = env.mj_model.geom("task_bottle_geom").id
            stream = (args.output/"bottle-ground-truth.jsonl").open("w")

            def observe_step():
                nonlocal sampled_at
                step()
                now = time.monotonic()
                if now < sampled_at:
                    return
                sampled_at = now + 0.1
                contacts = []
                for contact in env.mj_data.contact:
                    if geom in (contact.geom1, contact.geom2):
                        other = contact.geom2 if geom == contact.geom1 else contact.geom1
                        contacts.append(env.mj_model.geom(other).name or env.mj_model.body(env.mj_model.geom_bodyid[other]).name)
                stream.write(json.dumps({"at": now, "sim_time": env.mj_data.time,
                                         "bottle_position": env.mj_data.xpos[bottle].tolist(),
                                         "bottle_contacts": contacts,
                                         "robot_height": float(env.mj_data.qpos[2])})+'\n')
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
