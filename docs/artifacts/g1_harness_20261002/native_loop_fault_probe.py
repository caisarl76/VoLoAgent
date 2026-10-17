"""Bounded actual-main-loop fault probe: simulated DDS lo, scripted inputs.

The native VLA main loop is the only action publisher. This driver publishes
keyboard input, calls local RPC and observes wire traffic; no latent is sent.
"""

import argparse
from dataclasses import asdict
import hashlib
import json
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--native", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scenario", required=True,
                        choices=("policy-stall", "late-result-reset"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    origin = Path("/home/jihun/work/GR00T-WholeBodyControl")
    previous = Path(__file__).parents[1] / "g1_harness_20261001"
    endpoint = f"ipc:///tmp/g1-native-fault-{uuid.uuid4().hex}.sock"
    profile = load_profile(args.profile)
    result = {"outcome": "failed", "gate": "initialization", "robot_connection": False,
              "dds_interface": "lo", "gpu": 0, "action_port": 11556,
              "state_port": 11557, "keyboard_port": 11580,
              "native_main_loop": True, "scripted_camera": True, "scripted_policy": True,
              "scenario": args.scenario,
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
            record("action_wire", topic=topic.decode(), version=header["v"],
                   size=len(payload), sha256=hashlib.sha256(payload).hexdigest())
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
            if lease is not None and now >= heartbeat_at:
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
                "--output", str(args.output)], "mujoco", str(args.native))
        binary = args.native / "gear_sonic_deploy/target/release/g1_deploy_onnx_ref"
        policy = origin / "gear_sonic_deploy/policy/sonic_v1_1"
        result["controller_binary_sha256"] = hashlib.sha256(binary.read_bytes()).hexdigest()
        launch([str(binary), "lo", str(policy/"model_decoder.onnx"),
                str(origin/"gear_sonic_deploy/reference/example"),
                "--encoder-file", str(policy/"model_encoder.onnx"),
                "--planner-file", str(origin/"gear_sonic_deploy/planner/target_vel/V2/planner_sonic.onnx"),
                "--obs-config", str(policy/"observation_config.yaml"),
                "--input-type", "zmq_manager", "--harness-planner-hold", "--output-type", "zmq",
                "--disable-crc-check", "--zmq-host", "127.0.0.1", "--zmq-port", "11556",
                "--zmq-out-port", "11557", "--logs-dir", str(args.output/"controller")],
               "sonic", str(args.native), slave)
        launch([str(args.native/".venv/bin/python"), str(Path(__file__).with_name("scripted_native_worker.py")),
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
        reset = rpc("reset_standing", {"execution_id": "", "open_hands": False})
        pump(16, lambda s: s.phase in {"COMPLETED", "INTERRUPTED"})
        assert client.get_status().phase == "COMPLETED", "Initial measured reset failed"
        result["initial_reset"] = "passed_measured_dwell"

        execution = rpc("start_manipulation", {"skill_id": "bottle_to_right_table"})
        pump(2, lambda s: (args.output/"policy-started-2").exists())
        if args.scenario == "policy-stall":
            result["gate"] = "policy_stall"
            interrupted = pump(3, lambda s: s.phase == "INTERRUPTED")
            assert interrupted.reason == "policy_or_task_timeout", interrupted.reason
            assert interrupted.inference_epoch != execution.inference_epoch
            (args.output/"release-policy-2").touch()
            pump(3, lambda s: any(e['event']=='result_consumed' and e.get('epoch')==execution.inference_epoch for e in trace()))
            stopped = pump(3, lambda s: s.phase == "INTERRUPTED" and s.hold_confirmed)
            result["policy_stall"] = {"outcome": "passed", "reason": stopped.reason,
                                      "hold_confirmed": stopped.hold_confirmed, "old_epoch": execution.inference_epoch}
        else:
            result["gate"] = "late_result_during_reset"
            rpc("pause_manipulation", {"execution_id": execution.execution_id})
            reset = rpc("reset_standing", {"execution_id": execution.execution_id, "open_hands": False})
            (args.output/"release-policy-2").touch()
            pump(3, lambda s: any(e['event']=='result_consumed' and e.get('epoch')==execution.inference_epoch for e in trace()))
            settled = pump(16, lambda s: s.phase in {"COMPLETED", "INTERRUPTED"})
            assert settled.phase == "COMPLETED", settled.reason
            assert settled.execution_id == reset.execution_id
            assert settled.inference_epoch != execution.inference_epoch
            old_result = next(e for e in trace() if e['event']=='result_consumed'
                              and e.get('epoch')==execution.inference_epoch)
            assert old_result['valid'], "Late result must be a valid chunk"
            assert old_result['captured_age_s'] < 0.8, "Late chunk must still be fresh during reset"
            result["late_result_during_reset"] = {"outcome": "passed", "old_epoch": execution.inference_epoch,
                                                   "reset_epoch": settled.inference_epoch,
                                                   "late_result_age_s": old_result['captured_age_s'],
                                                   "reset_completed": True, "hold_confirmed": settled.hold_confirmed}
        pump(0.5)
        assert not any(e['event']=='unexpected_latent_publication_attempt' for e in trace())
        assert not any(e['event']=='policy_acceptance' and e.get('accepted') for e in trace())
        assert wire_counts.get('pose:v4', 0) == 0
        rpc("release_control", {})
        lease = None
        result.update(outcome="passed", gate="actual_native_loop_faults", latent_wire_count=0)
    except Exception as exc:
        result.update(reason=str(exc))
    finally:
        # Unblock scripted calls before stopping the actual worker and its loop.
        for number in range(2, 9):
            (args.output/f"release-policy-{number}").touch()
        if client is not None:
            client.close()
        for name, process in reversed(processes):
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
        (args.output/"result.json").write_text(json.dumps(result,indent=2)+'\n')
        events.close()
    print(json.dumps(result))
    return 0 if result['outcome']=='passed' else 1


if __name__ == "__main__":
    raise SystemExit(main())
