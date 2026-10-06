"""Actual-main-loop locomotion probe: simulated DDS lo, scripted prewarm.

The native VLA main loop is the only action publisher. This driver publishes
keyboard input, calls local RPC and observes wire traffic; no latent is sent.
"""

import argparse
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import pty
import signal
import subprocess
import time
import uuid

import zmq

from vlm_orchestrator.harness.g1.client import G1Client
from vlm_orchestrator.harness.g1.registry import load_profile
from planner_wire import planner_fields, stationary
from trace_validation import validate_controller_trace


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--native", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--initial-yaw-deg", type=float, default=0.0)
    parser.add_argument("--turn-angle-deg", type=float, default=-15.0)
    parser.add_argument("--post-walk-hold-s", type=float, default=0.0)
    parser.add_argument("--controller-trace", action="store_true")
    args = parser.parse_args()
    if not math.isfinite(args.post_walk_hold_s) or not 0 <= args.post_walk_hold_s <= 3:
        parser.error("--post-walk-hold-s must be between 0 and 3 seconds")
    for name in ["native", "profile", "output"]:
        setattr(args, name, getattr(args, name).resolve())
    args.output.mkdir(parents=True, exist_ok=False)
    snapshot = args.output/"as-run"
    snapshot.mkdir()
    for name in ["native_locomotion_probe.py", "native_locomotion_worker.py", "planner_wire.py", "trace_validation.py"]:
        (snapshot/name).write_bytes(Path(__file__).with_name(name).read_bytes())
    for name in ["scripts/run_vla_inference.py", "utils/inference/bounded_planner.py", "utils/inference/harness_control.py"]:
        (snapshot/Path(name).name).write_bytes((args.native/'gear_sonic'/name).read_bytes())
    controller_source = args.native/'gear_sonic_deploy/src/g1/g1_deploy_onnx_ref'
    if args.controller_trace:
        for name in ['src/g1_deploy_onnx_ref.cpp', 'include/planner_trace.hpp', 'include/localmotion_kplanner.hpp']:
            (snapshot/Path(name).name).write_bytes((controller_source/name).read_bytes())
    origin = Path("/home/jihun/work/GR00T-WholeBodyControl")
    previous = Path(__file__).parents[1] / "g1_harness_20261001"
    endpoint = f"ipc:///tmp/g1-native-fault-{uuid.uuid4().hex}.sock"
    profile = load_profile(args.profile)
    result = {"outcome": "failed", "gate": "initialization", "robot_connection": False,
              "dds_interface": "lo", "gpu": 0, "action_port": 11556,
              "state_port": 11557, "keyboard_port": 11580,
              "native_main_loop": True, "scripted_camera": True, "scripted_policy": True,
              "initial_yaw_deg": args.initial_yaw_deg, "turn_angle_deg": args.turn_angle_deg,
              "post_walk_hold_s": args.post_walk_hold_s,
              "controller_trace": args.controller_trace,
              "profile_sha256": profile.registry_sha256, "endpoint": endpoint}
    events = (args.output / "events.jsonl").open("w")
    context = zmq.Context()
    keyboard = context.socket(zmq.PUB)
    keyboard.setsockopt(zmq.LINGER, 0)
    keyboard.bind("tcp://127.0.0.1:11580")
    wire = context.socket(zmq.SUB)
    wire.setsockopt(zmq.LINGER, 0)
    wire.setsockopt(zmq.SUBSCRIBE, b"")
    wire.connect("tcp://127.0.0.1:11556")
    master, slave = pty.openpty()
    processes, logs, client, lease = [], [], None, None
    session = "native-fault-probe"
    heartbeat_at = 0.0
    wire_counts = {}
    wire_events = []
    keep_heartbeats = True

    def record(event, **fields):
        events.write(json.dumps({"at": time.monotonic(), "event": event, **fields}) + "\n")
        events.flush()

    def launch(argv, name, python_path, stdin=None):
        log = (args.output / f"{name}.log").open("w")
        logs.append(log)
        process = subprocess.Popen(argv, cwd=args.native, stdin=stdin,
                                   stdout=log, stderr=log, start_new_session=True,
                                   env={**os.environ, "PYTHONPATH": python_path,
                                        "CUDA_VISIBLE_DEVICES": "0"})
        processes.append((name, process))
        record("launch", name=name, argv=argv, pid=process.pid)

    def observe_wire():
        while True:
            try:
                payload = wire.recv(zmq.NOBLOCK)
            except zmq.Again:
                break
            topic = next((t for t in (b"pose", b"planner", b"command") if payload.startswith(t)), None)
            if topic is None:
                raise AssertionError("Unknown action wire topic")
            header = json.loads(payload[len(topic):len(topic)+1280].rstrip(b"\0"))
            name = topic.decode() + ":v" + str(header["v"])
            wire_counts[name] = wire_counts.get(name, 0) + 1
            fields = planner_fields(payload) if topic == b"planner" else None
            row = dict(at=time.monotonic(), topic=topic.decode(), version=header["v"],
                       size=len(payload), sha256=hashlib.sha256(payload).hexdigest(), fields=fields)
            wire_events.append(row)
            record("action_wire", **row)
            if name == "pose:v4":
                raise AssertionError("Scripted result reached the controller wire")

    def rpc(method, params):
        response = client.request(method, params, session_id=session,
                                  lease_id=None if lease is None else lease.lease_id)
        record("rpc", method=method, result=asdict(response.result))
        return response.result

    def pump(seconds, predicate=None):
        nonlocal heartbeat_at
        deadline = time.monotonic()+seconds
        last = None
        while time.monotonic() < deadline:
            for name, process in processes:
                if process.poll() is not None:
                    raise RuntimeError(f"{name} exited {process.returncode}")
            observe_wire()
            now = time.monotonic()
            if lease is not None and keep_heartbeats and now >= heartbeat_at:
                rpc("heartbeat", {})
                heartbeat_at = now+0.25
            last = client.get_status()
            record("status", **asdict(last))
            if last.phase == "FAULT":
                raise AssertionError(last.reason)
            if predicate is not None and predicate(last):
                return last
            time.sleep(0.05)
        if predicate is not None:
            raise TimeoutError(f"Condition not met: {last}")
        return last

    def trace():
        path = args.output / "native-trace.jsonl"
        return [] if not path.exists() else [json.loads(s) for s in path.read_text().splitlines()]

    try:
        launch([str(origin/".venv_sim/bin/python"), str(previous/"sonic_sim_worker.py"),
                "--output", str(args.output), "--initial-yaw-deg", str(args.initial_yaw_deg)], "mujoco", str(args.native))
        binary = args.native / "gear_sonic_deploy/target/release/g1_deploy_onnx_ref"
        policy = origin / "gear_sonic_deploy/policy/sonic_v1_1"
        result["controller_binary_sha256"] = hashlib.sha256(binary.read_bytes()).hexdigest()
        controller_argv = [str(binary), "lo", str(policy/"model_decoder.onnx"),
                str(origin/"gear_sonic_deploy/reference/example"),
                "--encoder-file", str(policy/"model_encoder.onnx"),
                "--planner-file", str(origin/"gear_sonic_deploy/planner/target_vel/V2/planner_sonic.onnx"),
                "--obs-config", str(policy/"observation_config.yaml"),
                "--input-type", "zmq_manager", "--harness-planner-hold", "--output-type", "zmq",
                "--disable-crc-check", "--zmq-host", "127.0.0.1", "--zmq-port", "11556",
                "--zmq-out-port", "11557", "--logs-dir", str(args.output/"controller")]
        if args.controller_trace:
            controller_argv.extend(['--planner-trace-file', str(args.output/'planner-trace.jsonl')])
        launch(controller_argv, "sonic", str(args.native), slave)
        launch([str(args.native/".venv/bin/python"), str(Path(__file__).with_name("native_locomotion_worker.py")),
                "--output", str(args.output), "--profile", str(args.profile), "--endpoint", endpoint],
               "native", str(args.native))
        deadline = time.monotonic()+30
        while not Path(endpoint[6:]).exists():
            for name, process in processes:
                if process.poll() is not None:
                    raise RuntimeError(f"{name} exited {process.returncode}")
            if time.monotonic() > deadline:
                raise TimeoutError("Native RPC did not start")
            time.sleep(0.1)
        client = G1Client(endpoint)
        client.get_status()
        # Native keyboard path performs operator startup; no second actuator PUB.
        # Controller initialization can outlast the first native keyboard command.
        # Wait for its actual input/output setup before sending startup once.
        deadline = time.monotonic()+60
        while "Init Done" not in (args.output/"sonic.log").read_text():
            if any(p.poll() is not None for _,p in processes):
                raise RuntimeError("Component exited during controller initialization")
            if time.monotonic() > deadline:
                raise TimeoutError("Controller did not finish initialization")
            time.sleep(0.1)
        time.sleep(0.3)
        keyboard.send_string("k")
        pump(30, lambda s: s.controller_running and s.policy_ready and s.telemetry_age_s is not None and s.telemetry_age_s < 0.5)
        (args.output/"release-band").touch()
        pump(2)
        lease = rpc("claim_control", {"registry_sha256": profile.registry_sha256})
        rpc("reset_standing", {"execution_id": "", "open_hands": False})
        pump(16, lambda s: s.phase in {"COMPLETED", "INTERRUPTED"})
        assert client.get_status().phase == "COMPLETED", "Initial measured reset failed"
        result["initial_reset"] = "passed_measured_dwell"

        result['gate'] = 'walking'
        rpc('walk_for', dict(direction='backward', duration_s=1.0, speed_mps=0.2))
        walked = pump(3, lambda s: s.phase in {'COMPLETED', 'INTERRUPTED'})
        assert walked.phase == 'COMPLETED' and walked.hold_confirmed, walked.reason
        result['walking'] = 'passed_duration_and_stop_ack'
        if args.post_walk_hold_s:
            cutoff = time.monotonic()
            waited = pump(args.post_walk_hold_s)
            assert waited.phase == 'COMPLETED' and waited.hold_confirmed
            packets = [row for row in wire_events if row['at'] >= cutoff and row['topic'] == 'planner']
            assert len(packets) >= 20 and all(stationary(row['fields']) for row in packets)
            result['post_walk_hold_elapsed_s'] = time.monotonic() - cutoff
            result['post_walk_hold_stationary_packets'] = len(packets)
        result['gate'] = 'turning'
        execution = rpc('turn_by', dict(angle_rad=math.radians(args.turn_angle_deg), rate_rps=math.radians(10)))
        started = time.monotonic()
        terminal = pump(11, lambda s: s.phase in {'COMPLETED', 'INTERRUPTED'})
        result['turn_elapsed_s'] = time.monotonic()-started
        result['turn_status'] = asdict(terminal)
        result['turning'] = 'passed_measured_dwell_and_stop_ack' if terminal.phase == 'COMPLETED' else 'interrupted'
        if terminal.phase == 'INTERRUPTED':
            result['gate'] = 'turn_interruption_hold'
            acknowledged = pump(3, lambda s: s.phase == 'INTERRUPTED' and s.hold_confirmed
                                and s.telemetry_index > terminal.telemetry_index)
            assert acknowledged.inference_epoch != execution.inference_epoch
            interruption_samples = trace()
            interrupted = next(x for x in interruption_samples if x.get('event') == 'turn_sample'
                               and x['phase'] == 'INTERRUPTED')
            result['hold_ack_elapsed_s'] = time.monotonic()-interrupted['at']
            result['hold_status'] = asdict(acknowledged)
            cutoff = time.monotonic()
            pump(1.0)
            held = client.get_status()
            assert held.phase == 'INTERRUPTED' and held.hold_confirmed
            assert held.inference_epoch == acknowledged.inference_epoch
            packets = [row for row in wire_events if row['at'] >= cutoff and row['topic'] == 'planner']
            assert len(packets) >= 20 and all(stationary(row['fields']) for row in packets)
            commands = [x for x in trace() if x['event'] == 'planner_conversion' and x['at'] >= cutoff]
            assert len(commands) >= 5
            facing = commands[0]['world']['facing']
            assert all(x['world']['facing'] == facing for x in commands)
            result['post_interrupt_stationary_packets'] = len(packets)
            result['turn_interruption_hold'] = 'passed_fresh_ack_and_no_translation_or_resume'
        else:
            result['gate'] = 'coordinator_loss'
            rpc('walk_for', dict(direction='forward', duration_s=5.0, speed_mps=0.2))
            keep_heartbeats = False
            lost = pump(3, lambda s: s.phase == 'INTERRUPTED' and s.hold_confirmed and s.owner_session_id is None)
            assert lost.reason == 'lease_expired', lost.reason
            cutoff = time.monotonic()
            pump(1.0)
            packets = [row for row in wire_events if row['at'] >= cutoff and row['topic'] == 'planner']
            assert len(packets) >= 20 and all(stationary(row['fields']) for row in packets)
            result['coordinator_loss'] = dict(outcome='passed', status=asdict(lost), stationary_packets=len(packets))
        assert not any(x['event'] == 'unexpected_latent_publication_attempt' for x in trace())
        assert wire_counts.get('pose:v4', 0) == 0
        result.update(outcome='passed_control_checks', gate='actual_native_loop_locomotion',
                      turn_success=terminal.phase == 'COMPLETED', latent_wire_count=0)
    except Exception as exc:
        result.update(reason=str(exc))
    finally:
        if client is not None:
            client.close()
        for name, process in reversed(processes):
            if name == 'sonic' and args.controller_trace and process.poll() is None:
                # Existing controller keyboard emergency stop, after trial checks.
                # Main freezes/flushes the memory trace before thread cleanup.
                os.write(master, b'o')
                try:
                    process.wait(3)
                except subprocess.TimeoutExpired:
                    record('controller_cleanup_fallback', trace_exists=(args.output/'planner-trace.jsonl').exists())
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGINT if name == "native" else signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    process.wait(3)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    process.wait()
        for log in logs:
            log.close()
        keyboard.close()
        wire.close()
        context.term()
        os.close(master)
        os.close(slave)
        result['wire_counts'] = wire_counts
        result['owned_process_exit_codes'] = {name:p.returncode for name,p in processes}
        if args.controller_trace:
            try:
                result['controller_trace_metadata'] = validate_controller_trace(args.output/'planner-trace.jsonl')
                result['controller_trace_written'] = True
            except (OSError, ValueError) as error:
                result['controller_trace_written'] = False
                result.update(outcome='failed', reason=f'Controller trace incomplete: {error}')
        (args.output/"result.json").write_text(json.dumps(result,indent=2)+'\n')
        events.close()
    print(json.dumps(result))
    return 0 if result['outcome']=='passed_control_checks' else 1


if __name__ == "__main__":
    raise SystemExit(main())
