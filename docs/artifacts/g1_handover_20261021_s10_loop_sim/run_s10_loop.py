"""Run the production VoLoAgent handover autopilot with the served s10 policy against the loopback simulator.

Usage: run_s10_loop.py --policy-endpoint 127.0.0.1:5561 [--box-front-m 0.28] [--cycles 1] --gpu 1 --output DIR

SIMULATION FIXTURES ONLY:
- Policy: none. sonic_policy_server.py serves the s10 checkpoint; the native loop drives it through its LeRobot client
  (profile lerobot_policy: the s10 robot sessions' 0.3 s chunk blend and 0.05 token step).
- Camera: the simulator renders the fitted D435 head camera (d435i_head.json, as lerobot's Isaac evaluation).
- Box, bottle and person: loop_sim_worker.py, plus the person thread here. The case (black, 0.475 x 0.35 m, top
  0.845 m) and the bottle (clear PET 15.5 cm body, yellow cap) follow the real setup's measurements. The standing
  bottle is a contact-free prop until the hand closes on it, so this tests the loop, not grasp physics.
- Vision: an oracle that answers each handover phase from the simulator's bottle state.

Production code under test:
- VoLoAgent HandoverRunner.run_autopilot (claim, phases, leases, pause, ready returns);
- the native executor (harness control, the LeRobot policy client and 50 Hz token stream, StandingReset, ready
  clearance);
- the SONIC controller.

Start: as in g1_handover_20261019_loop_sim, the robot is pre-posed to the recorded standing start pose with the box
away; the box arrives after the autopilot's startup ready return.
"""
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
import threading
import time
import uuid

import yaml
import zmq

from vlm_orchestrator.harness.g1.autopilot import HandoverRunner
from vlm_orchestrator.harness.g1.client import G1Client, RPCError
from vlm_orchestrator.harness.g1.handover_monitor import CRITERIA, HandoverMonitor
from vlm_orchestrator.harness.g1.registry import load_profile

from scene_s10 import BOTTLE_HALF_HEIGHT_M, BOTTLE_PELVIS_M, BOTTLE_RADIUS_M, restyle_scene

ROOT = Path(__file__).parent
ART = ROOT.parent
NATIVE = Path("/home/jihun/work/VoLoAgent/.worktrees/g1-harness-native")
ORIGIN = Path("/home/jihun/work/GR00T-WholeBodyControl")
PROFILE_TEMPLATE = ART.parent.parent / "configs/g1/handover.example.yaml"
S10_MODEL = "/mnt/data/jihun/g1_models/wbt_groot_handover261020_he_full/pretrained_model"
# The s10 robot sessions' streamer settings (--chunk-blend-s 0.3 --max-token-step 0.05); the real robot and this
# simulation (scene_fixture.configure_hand_order) both use the dataset's right-hand order.
LEROBOT_POLICY = dict(chunk_blend_s=0.3, max_token_step=0.05, dex3_right_order="dataset")
MIN_FREE_GPU_MIB = 2048


def oracle(output):
    """Answer the handover phase criterion from the simulator's bottle state."""
    def call(system, content):
        text = " ".join(item.get("text", "") for item in content if isinstance(item, dict))
        phase = next(name for name, criterion in CRITERIA.items() if criterion in text)
        truth = json.loads((output / "bottle-state.json").read_text())
        bottle = truth["bottle"]
        if bottle in ("fallen", "dropped"):
            outcome = "failure"
        elif phase == "startup_empty":
            outcome = "yes" if bottle in ("on_box", "taken") else "no"
        elif phase == "ready":
            outcome = "yes" if bottle == "on_box" and not truth["box_away"] else "no"
        elif phase == "pick_offer":
            outcome = "yes" if bottle == "on_palm" else "no"
        else:  # wait_empty
            outcome = "yes" if bottle == "taken" else "no"
        with (output / "oracle.jsonl").open("a") as stream:
            stream.write(json.dumps(dict(at=time.monotonic(), phase=phase, bottle=bottle, outcome=outcome)) + "\n")
        return json.dumps(dict(outcome=outcome, reason=f"sim oracle: bottle {bottle}"))
    return call


def person(endpoint, output, stop, record, pre_pose_execution):
    """Restore the box after the startup return; take offered bottles; place new ones."""
    client = G1Client(endpoint)
    retried, taken, placed, offered_at, returned_at = False, 0, 0, None, None
    try:
        while not stop.is_set():
            try:
                status = client.get_status()
            except RPCError as error:  # A busy action loop must not end the person for the rest of the run.
                record("person_status_error", error=str(error))
                time.sleep(0.05)
                continue
            try:
                truth = json.loads((output / "bottle-state.json").read_text())
            except (OSError, ValueError):
                time.sleep(0.05)
                continue
            now = time.monotonic()
            # The startup return is a standalone reset_ready; later returns complete the bottle_handover execution.
            ready_done = status.phase == "COMPLETED" and status.execution_id != pre_pose_execution
            if not retried and ready_done and truth["box_away"]:
                # The box is not at the robot yet: --box-after-startup, or the start pose overlapped it.
                (output / f"box-restore-{1 if not (output / 'box-restore-1').exists() else 2}").touch()
                retried = True
                record("person_box_restore_after_startup_return")
            if truth["bottle"] == "on_palm" and status.phase == "PAUSED":
                offered_at = offered_at or now
                if now - offered_at >= 1.0:
                    taken += 1
                    (output / f"take-bottle-{taken}").touch()
                    record("person_take", cycle=taken)
                    offered_at = None
            if truth["bottle"] == "taken" and ready_done and placed < taken:
                returned_at = returned_at or now
                if now - returned_at >= 0.5:
                    placed += 1
                    (output / f"place-bottle-{placed}").touch()
                    record("person_place", cycle=placed)
                    returned_at = None
            time.sleep(0.05)
    finally:
        client.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--policy-endpoint", required=True, help="host:port of sonic_policy_server.py serving s10")
    parser.add_argument("--box-front-m", type=float, default=0.28,
                        help="Case near edge from the pelvis: the real 20 cm gap from the torso front (0.08 m).")
    parser.add_argument("--box-height-m", type=float, default=0.845)
    parser.add_argument("--cycles", type=int, default=1)
    parser.add_argument("--manipulation-deadline-s", type=float, default=60.0)
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--dds-domain", type=int, default=0,
                        help="DDS domain for the sim and deploy; non-zero keeps other G1 sims on lo from cross-feeding lowcmd.")
    parser.add_argument("--box-after-startup", action="store_true",
                        help="Keep the box away until the autopilot's startup ready return completes.")
    parser.add_argument("--gravity-feedforward", action="store_true",
                        help="Set the opt-in ready_gravity_feedforward in the simulation profile copy.")
    args = parser.parse_args()
    gpu = subprocess.run(["nvidia-smi", "--id", str(args.gpu), "--query-gpu=index,name,memory.free",
                          "--format=csv,noheader,nounits"], check=True, capture_output=True, text=True).stdout.strip()
    if int(gpu.rsplit(",", 1)[1]) < MIN_FREE_GPU_MIB:
        raise RuntimeError(f"SONIC needs {MIN_FREE_GPU_MIB} MiB free: {gpu}")
    for port in (11555, 11556, 11557, 11580):
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", port))
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    bottle_min_x = args.box_front_m + BOTTLE_RADIUS_M + 0.01  # Fully on the box.
    policy_host, policy_port = args.policy_endpoint.rsplit(":", 1)
    subprocess.run([str(ORIGIN / ".venv_sim/bin/python"), "-c",
                    "from scene import build_scene; import sys; build_scene(sys.argv[1], desk_front_m=float(sys.argv[2]), "
                    "desk_height_m=float(sys.argv[3]))", str(output / "scene.xml"), str(args.box_front_m),
                    str(args.box_height_m)], cwd=ART / "g1_handover_20261012_acceptance", check=True)
    scene = restyle_scene(output / "scene.xml", args.box_front_m, args.box_height_m,
                          json.loads((ROOT / "d435i_head.json").read_text()))
    data = yaml.safe_load(PROFILE_TEMPLATE.read_text())
    data.update(policy_host=policy_host, policy_port=int(policy_port), checkpoint=S10_MODEL,
                lerobot_policy=LEROBOT_POLICY)
    data["limits"]["manipulation_deadline_s"] = args.manipulation_deadline_s
    handover = data["handover"]
    handover.update(checkpoint_verified=True, ready_pose_reviewed=True,
                    desk_top_above_floor_m=args.box_height_m, desk_front_from_pelvis_m=args.box_front_m,
                    ready_gravity_feedforward=args.gravity_feedforward)
    profile_path = output / "simulation-profile.yaml"
    profile_path.write_text("# SIMULATION TEST FIXTURE ONLY: gates do not qualify a checkpoint or physical return.\n"
                            + yaml.safe_dump(data, sort_keys=False))
    profile = load_profile(profile_path)
    endpoint = "ipc:///tmp/g1-handover-loop-" + uuid.uuid4().hex + ".sock"
    result = dict(completed=False, box_front_m=args.box_front_m, box_height_m=args.box_height_m,
                  gravity_feedforward=args.gravity_feedforward, box_after_startup=args.box_after_startup,
                  dds_domain=args.dds_domain,
                  bottle_pelvis_frame_m=list(BOTTLE_PELVIS_M), bottle_min_x_m=bottle_min_x, cycles=args.cycles,
                  policy_endpoint=args.policy_endpoint, checkpoint=S10_MODEL, lerobot_policy=LEROBOT_POLICY,
                  scene=scene, served_policy=True, rendered_head_camera=True, oracle_vision=True,
                  scripted_person=True, gpu_preflight=gpu,
                  hardware_acceptance=False, robot_connection=False,
                  native_base_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=NATIVE, text=True).strip(),
                  volo_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ART, text=True).strip())
    context = zmq.Context()
    keyboard = context.socket(zmq.PUB)
    keyboard.setsockopt(zmq.LINGER, 0)
    keyboard.bind("tcp://127.0.0.1:11580")
    master, slave = pty.openpty()
    processes, logs, client = [], [], None
    events = (output / "events.jsonl").open("x")
    stop = threading.Event()

    def record(event, **fields):
        events.write(json.dumps(dict(at=time.monotonic(), event=event, **fields), default=str) + "\n")
        events.flush()

    def launch(command, name, stdin=None):
        log = (output / (name + ".log")).open("x")
        logs.append(log)
        process = subprocess.Popen(command, cwd=NATIVE, stdin=stdin, stdout=log, stderr=log, start_new_session=True,
                                   env={**os.environ, "PYTHONPATH": str(NATIVE), "CUDA_VISIBLE_DEVICES": str(args.gpu),
                                        "MUJOCO_GL": "egl"})
        processes.append((name, process))
        record("launch", name=name, command=command, pid=process.pid)

    def wait_for(predicate, seconds, what):
        deadline = time.monotonic() + seconds
        while not predicate():
            if any(p.poll() is not None for _, p in processes):
                raise RuntimeError(f"Child exited while waiting for {what}; inspect logs")
            if time.monotonic() > deadline:
                raise TimeoutError(what)
            time.sleep(0.1)

    try:
        policy = ORIGIN / "gear_sonic_deploy/policy/sonic_v1_1"  # SONIC v1.1 decoder/encoder, as s10's tokens.
        binary = NATIVE / "gear_sonic_deploy/target/release/g1_deploy_onnx_ref"
        result["binary_sha256"] = hashlib.sha256(binary.read_bytes()).hexdigest()
        launch([str(ORIGIN / ".venv_sim/bin/python"), str(ROOT / "loop_sim_worker.py"), "--output", str(output),
                "--scene", str(output / "scene.xml"), "--box-height-m", str(args.box_height_m),
                "--bottle-forward-m", str(BOTTLE_PELVIS_M[0]), "--bottle-left-m", str(BOTTLE_PELVIS_M[1]),
                "--bottle-min-x-m", str(bottle_min_x), "--bottle-half-height-m", str(BOTTLE_HALF_HEIGHT_M),
                "--camera-port", "11555", "--dds-domain", str(args.dds_domain)], "mujoco")
        launch([str(binary), "lo", str(policy / "model_decoder.onnx"), str(ORIGIN / "gear_sonic_deploy/reference/example"),
                "--encoder-file", str(policy / "model_encoder.onnx"), "--planner-file",
                str(ORIGIN / "gear_sonic_deploy/planner/target_vel/V2/planner_sonic.onnx"),
                "--obs-config", str(policy / "observation_config.yaml"), "--input-type", "zmq_manager",
                "--harness-planner-hold", "--output-type", "zmq", "--disable-crc-check",
                "--dds-domain", str(args.dds_domain), "--zmq-host", "127.0.0.1", "--zmq-port", "11556", "--zmq-out-port", "11557",
                "--logs-dir", str(output / "controller")], "sonic", slave)
        launch([str(NATIVE / ".venv/bin/python"), str(ROOT / "loop_native_worker.py"), "--output", str(output),
                "--profile", str(profile_path), "--endpoint", endpoint], "native")
        wait_for(lambda: Path(endpoint[6:]).exists() and "Init Done" in (output / "sonic.log").read_text(),
                 60, "controller initialization")
        client = G1Client(endpoint)
        time.sleep(0.3)
        keyboard.send_string("k")
        wait_for(lambda: (lambda s: s.controller_running and s.policy_ready and s.telemetry_age_s is not None
                          and s.telemetry_age_s < 0.5)(client.get_status()), 60, "controller running and policy ready")
        (output / "startup-request").touch()
        wait_for(lambda: (output / "startup-receipt.json").exists(), 10, "grounded startup")
        time.sleep(2.0)
        # Fixture pre-pose to the recorded standing start pose (box away), then hand over.
        lease = client.request("claim_control", {"registry_sha256": profile.registry_sha256},
                               session_id="loop-pre-pose").result
        pre_pose = client.request("reset_ready", {"execution_id": ""}, session_id="loop-pre-pose",
                                  lease_id=lease.lease_id).result
        deadline, sequence = time.monotonic() + 25, 0
        while (status := client.get_status()).phase not in ("COMPLETED", "INTERRUPTED", "FAULT"):
            if time.monotonic() > deadline:
                raise TimeoutError(f"Start pre-pose did not settle: {status.phase}")
            sequence += 1
            client.request("heartbeat", {"sequence": sequence}, session_id="loop-pre-pose", lease_id=lease.lease_id)
            time.sleep(0.2)
        record("pre_pose_terminal", phase=status.phase, reason=status.reason)
        if status.phase != "COMPLETED":
            raise RuntimeError(f"Start pre-pose failed: {status.phase} {status.reason}")
        client.request("release_control", {}, session_id="loop-pre-pose", lease_id=lease.lease_id)
        if not args.box_after_startup:
            (output / "box-restore-1").touch()
            wait_for(lambda: (output / "box-restore-1.json").exists(), 5, "box restore decision")
            result["box_restored_at_start"] = json.loads((output / "box-restore-1.json").read_text())
        helper = threading.Thread(target=person, args=(endpoint, output, stop, record, pre_pose.execution_id),
                                  daemon=True)
        helper.start()
        monitor = HandoverMonitor(oracle(output), time.monotonic, profile.limits)
        runner = HandoverRunner(profile, lambda: G1Client(endpoint), monitor, output / "missions")
        record("autopilot_start", cycles=args.cycles)
        mission = runner.run_autopilot(args.cycles)
        record("autopilot_end", outcome=mission.outcome, reason=mission.reason)
        result.update(mission=asdict(mission), completed=True, autopilot_outcome=mission.outcome,
                      autopilot_reason=mission.reason, final_status=asdict(client.get_status()))
    except Exception as error:
        result["error"] = f"{type(error).__name__}: {error}"
    finally:
        stop.set()
        if client is not None:
            client.close()
        for name, process in reversed(processes):
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGINT if name == "native" else signal.SIGTERM)
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
        for log in logs:
            log.close()
        keyboard.close()
        context.term()
        os.close(master)
        os.close(slave)
        events.close()
        result["owned_process_exit_codes"] = {name: p.returncode for name, p in processes}
        result["owned_processes_stopped"] = all(p.poll() is not None for _, p in processes)
        (output / "result.json").write_text(json.dumps(result, indent=2, default=str) + "\n")
    print(json.dumps({k: result.get(k) for k in ("completed", "autopilot_outcome", "autopilot_reason", "error")}, indent=2))
    return 0 if result.get("autopilot_outcome") == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
