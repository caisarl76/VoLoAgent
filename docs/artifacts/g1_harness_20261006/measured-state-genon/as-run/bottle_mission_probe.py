"""One real-checkpoint/sample-scene mission on DDS lo; no physical robot.

Native inference is the sole actuator publisher. The driver starts owned local
processes, uses the real coordinator/Genon monitor, and saves object truth.
Placement requires fresh stable top support without any robot contact.
This gate validates only the supplied simulation, not real-world calibration.
"""

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import pty
import shlex
import signal
import subprocess
import time
import uuid
import xml.etree.ElementTree as ET

from openai import OpenAI
import zmq

from vlm_orchestrator.harness.g1.client import G1Client
from vlm_orchestrator.harness.g1.contract import SkillCall
from vlm_orchestrator.harness.g1.monitor import G1CompletionMonitor
from vlm_orchestrator.harness.g1.registry import load_profile
from vlm_orchestrator.harness.g1.runner import HarnessRunner
from vlm_orchestrator.vlm.api import chat_create
from placement_truth import placement_confirmed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--native", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--env-file", type=Path, required=True)
    args = parser.parse_args()
    for name in ['native', 'profile', 'scene', 'output', 'env_file']:
        setattr(args, name, getattr(args, name).resolve())
    api_key = None
    for line in args.env_file.read_text().splitlines():
        name, separator, value = line.strip().removeprefix('export ').partition('=')
        if separator and name.strip() == 'GENON_API_KEY':
            api_key = shlex.split(value, comments=True)[0]
    if not api_key:
        raise ValueError('GENON_API_KEY required')
    args.output.mkdir(parents=True, exist_ok=False)
    snapshot = args.output/'as-run'
    snapshot.mkdir()
    for filename in ['bottle_mission_probe.py', 'bottle_live_sim_worker.py', 'placement_truth.py']:
        (snapshot/filename).write_bytes(Path(__file__).with_name(filename).read_bytes())
    (snapshot/'run_vla_inference.py').write_bytes((args.native/'gear_sonic/scripts/run_vla_inference.py').read_bytes())
    stool_geom = ET.parse(args.scene).getroot().find("worldbody/geom[@name='right_green_stool']")
    if stool_geom is None or stool_geom.get('type') != 'cylinder':
        raise ValueError('Scene must define the cylindrical right_green_stool destination')
    stool_pos = [float(v) for v in stool_geom.get('pos').split()]
    stool_size = [float(v) for v in stool_geom.get('size').split()]
    destination = dict(geom='right_green_stool', xy=stool_pos[:2], radius=stool_size[0], height=stool_pos[2]+stool_size[1])
    origin = Path('/home/jihun/work/GR00T-WholeBodyControl')
    profile = load_profile(args.profile)
    endpoint = f'ipc:///tmp/g1-bottle-mission-{uuid.uuid4().hex}.sock'
    result = {'outcome': 'failed', 'gate': 'initialization', 'robot_connection': False,
              'dds_interface': 'lo', 'controller_gpu': 0, 'policy_gpu': 1,
              'checkpoint': profile.checkpoint, 'profile_sha256': profile.registry_sha256,
              'scene_sha256': hashlib.sha256(args.scene.read_bytes()).hexdigest(),
              'native_runner_sha256': hashlib.sha256((snapshot/'run_vla_inference.py').read_bytes()).hexdigest(),
              'destination_geometry': destination,
              'scene_calibrated': False, 'placement_validated': False,
              'destination_contact_only': False,
              'monitor_model': 'openai/gpt-5.6-sol', 'monitor_base_url': 'https://api.genon.ai/v1'}
    processes, logs = [], []
    master, slave = pty.openpty()
    context = zmq.Context()
    keyboard = context.socket(zmq.PUB)
    keyboard.setsockopt(zmq.LINGER, 0)
    keyboard.bind('tcp://127.0.0.1:11580')
    client = None
    events = (args.output/'driver-events.jsonl').open('w')

    def launch(name, argv, gpu, path, stdin=None):
        log = (args.output/f'{name}.log').open('w')
        logs.append(log)
        process = subprocess.Popen(argv, cwd=args.native, stdout=log, stderr=log, stdin=stdin,
                                   start_new_session=True, env={**os.environ, 'PYTHONPATH': path,
                                   'CUDA_VISIBLE_DEVICES': str(gpu), 'MUJOCO_GL': 'egl',
                                   'PYTHONUNBUFFERED': '1',
                                   **({'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'} if name == 'policy' else {}),
                                   'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1',
                                   'NO_ALBUMENTATIONS_UPDATE': '1', 'TOKENIZERS_PARALLELISM': 'false'})
        processes.append((name,process))
        events.write(json.dumps({'at':time.monotonic(),'event':'launch','name':name,'pid':process.pid,'argv':argv,'gpu':gpu})+'\n')
        events.flush()

    def wait_for(predicate, timeout):
        deadline = time.monotonic()+timeout
        while time.monotonic()<deadline:
            for name, process in processes:
                if process.poll() is not None:
                    raise RuntimeError(f'{name} exited {process.returncode}')
            if predicate():
                return
            time.sleep(0.1)
        raise TimeoutError('Readiness deadline exceeded')

    try:
        result['gate'] = 'policy_server_startup'
        launch('policy', [str(Path('/home/jihun/work/Isaac-GR00T/.venv/bin/python')),
                         str(origin/'gear_sonic/scripts/serve_pnp_bottle_offline.py'),
                         '--model-path', profile.checkpoint, '--embodiment-tag', 'UNITREE_G1_SONIC',
                         '--device', 'cuda:0', '--host', '127.0.0.1', '--port', '15558'],
               1, '/home/jihun/work/Isaac-GR00T')
        # The worker's 10s prewarm deadline starts only after serving is ready.
        # Loading the model is a separate startup prerequisite, not a policy call.
        wait_for(lambda: 'Server is ready and listening' in (args.output/'policy.log').read_text(), 240)
        result['gate'] = 'native_preparation'
        launch('mujoco', [str(origin/'.venv_sim/bin/python'), str(Path(__file__).with_name('bottle_live_sim_worker.py')),
                          '--output', str(args.output), '--scene', str(args.scene)], 0, str(args.native))
        binary = args.native/'gear_sonic_deploy/target/release/g1_deploy_onnx_ref'
        policy = origin/'gear_sonic_deploy/policy/sonic_v1_1'
        result['controller_binary_sha256'] = hashlib.sha256(binary.read_bytes()).hexdigest()
        launch('sonic', [str(binary), 'lo', str(policy/'model_decoder.onnx'), str(origin/'gear_sonic_deploy/reference/example'),
                         '--encoder-file', str(policy/'model_encoder.onnx'), '--planner-file',
                         str(origin/'gear_sonic_deploy/planner/target_vel/V2/planner_sonic.onnx'),
                         '--obs-config', str(policy/'observation_config.yaml'), '--input-type', 'zmq_manager',
                         '--harness-planner-hold', '--output-type', 'zmq', '--disable-crc-check',
                         '--zmq-host', '127.0.0.1', '--zmq-port', '11556', '--zmq-out-port', '11557',
                         '--logs-dir', str(args.output/'controller')], 0, str(args.native), slave)
        launch('native', [str(args.native/'.venv/bin/python'), '-m', 'gear_sonic.scripts.run_vla_inference',
                          '--camera-host', '127.0.0.1', '--camera-port', '11555',
                          '--state-zmq-host', '127.0.0.1', '--state-zmq-port', '11557',
                          '--action-zmq-host', '127.0.0.1', '--action-zmq-port', '11556',
                          '--keyboard-zmq-host', '127.0.0.1', '--keyboard-zmq-port', '11580',
                          '--harness-endpoint', endpoint, '--harness-profile', str(args.profile)], 0,
               str(args.native) + ':/home/jihun/work/Isaac-GR00T')
        wait_for(lambda: Path(endpoint[6:]).exists() and 'Init Done' in (args.output/'sonic.log').read_text(), 60)
        client = G1Client(endpoint)
        keyboard.send_string('k')
        wait_for(lambda: client.get_status().controller_running and client.get_status().policy_ready, 30)
        (args.output/'release-band').touch()
        time.sleep(2)
        def factory():
            return G1Client(endpoint)
        preparation = HarnessRunner(profile, factory, None, args.output/'preparation')
        prepared = preparation.run(SkillCall('reset_standing', {'open_hands': False}))
        result['preparation'] = asdict(prepared)
        if prepared.outcome != 'completed':
            raise RuntimeError(f'Preparation failed: {prepared.reason}')
        result['gate'] = 'actual_checkpoint_bottle_mission'
        api = OpenAI(api_key=api_key, base_url=result['monitor_base_url'], timeout=10, max_retries=0)
        def vision(system, content):
            return chat_create(api, model=result['monitor_model'], temperature=0,
                               messages=[{'role':'system','content':system},{'role':'user','content':content}]).choices[0].message.content
        monitor = G1CompletionMonitor(profile.require_skill('bottle_to_right_table'), vision, time.monotonic, profile.limits)
        runner = HarnessRunner(profile, factory, monitor, args.output/'mission')
        mission = runner.run(SkillCall('bottle_to_right_table', {}))
        result['mission'] = asdict(mission)
        result['outcome'] = mission.outcome
        result['final_status'] = asdict(client.get_status())
        truth = [json.loads(s) for s in (args.output/'bottle-ground-truth.jsonl').read_text().splitlines(keepends=True)
                 if s.endswith('\n')]
        recent = truth[-10:]
        result['independent_last_object_samples'] = recent
        result['destination_contact_only'] = len(recent) == 10 and all(
            any(c['geom'] == 'right_green_stool' for c in row['contacts']) for row in recent)
        result['truth_evaluated_at'] = time.monotonic()
        result['placement_validated'] = placement_confirmed(truth, destination, now=result['truth_evaluated_at'])
        result['outcome'] = 'completed' if mission.outcome == 'completed' and result['placement_validated'] else 'failed'
        result['reason'] = ('Coordinator and independent simulation placement both passed' if result['outcome'] == 'completed'
                            else 'Independent object truth does not establish placement' if not result['placement_validated']
                            else f'Coordinator did not complete: {mission.reason}')
    except Exception as exc:
        result['reason'] = str(exc)
    finally:
        if client is not None:
            client.close()
        for name, process in reversed(processes):
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGINT if name in {'native','mujoco','policy'} else signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    process.wait(5)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    process.wait()
        for log in logs:
            log.close()
        events.close()
        keyboard.close()
        context.term()
        os.close(master)
        os.close(slave)
        result['owned_process_exit_codes'] = {name:process.returncode for name,process in processes}
        (args.output/'result.json').write_text(json.dumps(result,indent=2,default=str)+'\n')
        print(json.dumps(result,default=str))
    return 0 if result['outcome']=='completed' and result['placement_validated'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
