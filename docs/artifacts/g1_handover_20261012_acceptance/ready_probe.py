"""Actual native ready-return probe; scripted prewarm never publishes VLA actions."""
import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import pty
import signal
import socket
import subprocess
import time
import uuid

import yaml
import zmq

from vlm_orchestrator.harness.g1.client import G1Client
from vlm_orchestrator.harness.g1.registry import load_profile

ROOT=Path(__file__).parent
NATIVE=Path('/home/jihun/work/VoLoAgent/.worktrees/g1-harness-native')
ORIGIN=Path('/home/jihun/work/GR00T-WholeBodyControl')
NATIVE_WORKER=ROOT.parent/'g1_harness_20261006_followup/native_locomotion_worker.py'
MIN_FREE_GPU_MIB=3072
PROFILE_TEMPLATE=ROOT.parents[2]/'configs/g1/handover.example.yaml'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--desk-front-m',type=float,default=0.40)
    parser.add_argument('--desk-height-m',type=float,default=0.85)
    parser.add_argument('--gpu',type=int,default=0)
    args=parser.parse_args()
    gpu_check=subprocess.run(['nvidia-smi','--id',str(args.gpu),
        '--query-gpu=index,name,memory.free','--format=csv,noheader,nounits'],
        check=True,capture_output=True,text=True).stdout.strip()
    if int(gpu_check.rsplit(',',1)[1])<MIN_FREE_GPU_MIB:
        raise RuntimeError(f'SONIC needs at least {MIN_FREE_GPU_MIB} MiB free before startup: {gpu_check}')
    output=args.output.resolve()
    for port in (11556,11557,11580):
        with socket.socket() as probe:
            probe.bind(('127.0.0.1',port))
    for proc in Path('/proc').glob('[0-9]*'):
        try:
            argv=[x.decode() for x in (proc/'cmdline').read_bytes().split(b'\0') if x]
            if any(Path(x).name in ('g1_deploy_onnx_ref','run_sim_loop.py','sonic_sim_worker.py',
                'bottle_sim_worker.py','bottle_live_sim_worker.py') for x in argv):
                raise RuntimeError(f'Existing controller/simulator PID {proc.name}; left untouched')
        except (OSError,UnicodeError):
            pass
    output.mkdir(parents=True,exist_ok=False)
    # The geometry builder needs MuJoCo; use the existing simulation environment.
    scene_code='from scene import build_scene; import sys; build_scene(sys.argv[1],desk_front_m=float(sys.argv[2]),desk_height_m=float(sys.argv[3]))'
    subprocess.run([str(ORIGIN/'.venv_sim/bin/python'),'-c',scene_code,str(output/'scene.xml'),str(args.desk_front_m),str(args.desk_height_m)],
        cwd=ROOT,check=True)
    template=PROFILE_TEMPLATE
    data=yaml.safe_load(template.read_text())
    # Test-fixture gates for the ready component only; no checkpoint qualification.
    data['handover']['ready_pose_reviewed']=True
    data['handover']['checkpoint_verified']=True
    path=output/'simulation-profile.yaml'
    path.write_text('# SIMULATION TEST FIXTURE ONLY: gates do not qualify a checkpoint or physical return.\n'
        +yaml.safe_dump(data,sort_keys=False))
    profile=load_profile(path)
    endpoint='ipc:///tmp/g1-handover-ready-'+uuid.uuid4().hex+'.sock'
    result=dict(completed=False,robot_connection=False,native_main_loop=True,
        policy_inference_used_for_motion=False,scripted_prewarm_only=True,desk_front_m=args.desk_front_m,desk_height_m=args.desk_height_m,
        hardware_profile_changed=False,hardware_acceptance=False,scene_calibrated=False,
        native_commit='44f9bba',endpoint=endpoint)
    result['readiness_gates_fixture_only']=True
    result['checkpoint_qualified']=False
    result['gpu_preflight']=gpu_check
    result['cuda_visible_devices']=str(args.gpu)
    context=zmq.Context()
    keyboard=context.socket(zmq.PUB)
    keyboard.setsockopt(zmq.LINGER,0)
    keyboard.bind('tcp://127.0.0.1:11580')
    master,slave=pty.openpty()
    processes,logs=[],[]
    client,lease=None,None
    heartbeat_at=0.0
    worker=NATIVE_WORKER
    events=(output/'events.jsonl').open('x')

    def record(event,**fields):
        events.write(json.dumps(dict(at=time.monotonic(),event=event,**fields))+'\n');events.flush()

    def launch(command,name,stdin=None):
        log=(output/(name+'.log')).open('x');logs.append(log)
        process=subprocess.Popen(command,cwd=NATIVE,stdin=stdin,stdout=log,stderr=log,
            start_new_session=True,env={**os.environ,'PYTHONPATH':str(NATIVE),'CUDA_VISIBLE_DEVICES':str(args.gpu)})
        processes.append((name,process));record('launch',name=name,command=command,pid=process.pid)

    def request(method,params):
        reply=client.request(method,params,session_id='handover-ready-sim',
            lease_id=None if lease is None else lease.lease_id)
        record('rpc',method=method,result=asdict(reply.result));return reply.result

    def pump(seconds,predicate=None):
        nonlocal heartbeat_at
        deadline=time.monotonic()+seconds;last=None
        while time.monotonic()<deadline:
            for name,process in processes:
                if process.poll() is not None:raise RuntimeError(f'{name} exited {process.returncode}')
            now=time.monotonic()
            if lease is not None and now>=heartbeat_at:
                request('heartbeat',{'sequence':time.monotonic_ns()});heartbeat_at=now+0.25
            last=client.get_status();record('status',**asdict(last))
            if last.phase=='FAULT':raise RuntimeError(last.reason)
            if predicate and predicate(last):return last
            time.sleep(0.05)
        if predicate:raise TimeoutError(f'Condition not met: {last}')
        return last

    try:
        policy=ORIGIN/'gear_sonic_deploy/policy/sonic_v1_1'
        binary=NATIVE/'gear_sonic_deploy/target/release/g1_deploy_onnx_ref'
        result['binary_sha256']=hashlib.sha256(binary.read_bytes()).hexdigest()
        launch([str(ORIGIN/'.venv_sim/bin/python'),str(ROOT/'sim_worker.py'),'--output',str(output),
            '--scene',str(output/'scene.xml')],'mujoco')
        launch([str(binary),'lo',str(policy/'model_decoder.onnx'),str(ORIGIN/'gear_sonic_deploy/reference/example'),
            '--encoder-file',str(policy/'model_encoder.onnx'),'--planner-file',
            str(ORIGIN/'gear_sonic_deploy/planner/target_vel/V2/planner_sonic.onnx'),
            '--obs-config',str(policy/'observation_config.yaml'),'--input-type','zmq_manager',
            '--harness-planner-hold','--output-type','zmq','--disable-crc-check',
            '--zmq-host','127.0.0.1','--zmq-port','11556','--zmq-out-port','11557',
            '--logs-dir',str(output/'controller')],'sonic',slave)
        launch([str(NATIVE/'.venv/bin/python'),str(worker),'--output',str(output),
            '--profile',str(path),'--endpoint',endpoint],'native')
        deadline=time.monotonic()+60
        while not Path(endpoint[6:]).exists() or 'Init Done' not in (output/'sonic.log').read_text():
            if any(p.poll() is not None for _,p in processes):raise RuntimeError('Startup child exited; inspect logs')
            if time.monotonic()>deadline:raise TimeoutError('Controller initialization deadline')
            time.sleep(0.1)
        client=G1Client(endpoint)
        time.sleep(0.3);keyboard.send_string('k')
        pump(30,lambda s:s.controller_running and s.policy_ready and s.telemetry_age_s is not None and s.telemetry_age_s<0.5)
        (output/'startup-request').touch()
        pump(2.5)
        assert (output/'startup-receipt.json').exists(),'Grounded startup was not applied'
        lease=request('claim_control',dict(registry_sha256=profile.registry_sha256))
        result['initial_status']=asdict(client.get_status())
        record('ready_begin')
        execution=request('reset_ready',dict(execution_id=''))
        result['ready_execution']=asdict(execution)
        terminal=pump(16,lambda s:s.phase in ('COMPLETED','INTERRUPTED'))
        record('ready_terminal',phase=terminal.phase,reason=terminal.reason)
        result['terminal_status']=asdict(terminal)
        pump(1.0)
        result['completed']=True
        result['measured_ready_settling_passed']=terminal.phase=='COMPLETED'
    except Exception as error:
        result['error']=f'{type(error).__name__}: {error}'
    finally:
        if client is not None:client.close()
        for name,process in reversed(processes):
            if process.poll() is None:
                os.killpg(process.pid,signal.SIGINT if name=='native' else signal.SIGTERM)
                try:process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid,signal.SIGKILL);process.wait()
        for log in logs:log.close()
        keyboard.close();context.term();os.close(master);os.close(slave);events.close()
        result['owned_process_exit_codes']={name:p.returncode for name,p in processes}
        result['owned_processes_stopped']=all(p.poll() is not None for _,p in processes)
        (output/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
    return 0 if result['completed'] else 1


if __name__=='__main__':
    raise SystemExit(main())
