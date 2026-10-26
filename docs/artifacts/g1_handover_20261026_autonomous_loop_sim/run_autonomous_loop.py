"""Run the autonomous G1 handover loop end to end against the loopback MuJoCo simulator (Task 13).

Usage (VoLoAgent worktree root):
  PYTHONPATH=. .venv/bin/python docs/artifacts/g1_handover_20261026_autonomous_loop_sim/run_autonomous_loop.py \
      --policy-endpoint 127.0.0.1:5571 --trials 3 --gpu 0 --dds-domain 37 --output OUT

SIMULATION FIXTURE ONLY: it checks the loop's mechanics; it does not qualify the policy.

What runs (production code under test):
- VoLoAgent `cli loop` (LoopRunner, RearmMonitor, OperatorConsole) as a separate process; its stdin is the
  scripted operator console below;
- VoLoAgent `cli verdict` as a separate process, with scripted verdicts;
- the native harness (run_vla_inference + HarnessControl + LoopController) through loop_native_worker.py, which
  adds a native-side message log only;
- the deploy binary built from the native head, on DDS domain --dds-domain;
- the s08 policy server (already running, not started here) and the local Qwen3-VL server (started here, used
  only for the side check).

Fixtures: the scene, D435 head camera and contact-free grasp from g1_handover_20261021_s10_loop_sim
(loop_sim_worker.py, scene_s10.py), the scene-truth re-arm oracle (oracle_vlm_server.py) and the scripted person
and console here. After the session, a separate step starts a fresh take-over and kills the native harness in
POSE to record the deploy's reaction to a dead pose stream (R19).
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

from vlm_orchestrator.harness.g1.client import G1Client, RPCError
from vlm_orchestrator.harness.g1.registry import load_profile

from scene_s10 import BOTTLE_HALF_HEIGHT_M, BOTTLE_PELVIS_M, BOTTLE_RADIUS_M, restyle_scene

ROOT = Path(__file__).resolve().parent
ART = ROOT.parent
VOLO = ART.parent.parent
NATIVE = Path("/home/jihun/work/VoLoAgent/.worktrees/g1-loop-native")
ORIGIN = Path("/home/jihun/work/GR00T-WholeBodyControl")
LEROBOT_PY = Path.home() / "work/lerobot/.venv/bin/python"
VLM_SNAPSHOT = (Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen3-VL-2B-Instruct/snapshots/"
                "89644892e4d85e24eaac8bacfd4f463576704203")
PROFILE_TEMPLATE = VOLO / "configs/g1/handover-loop.example.yaml"
PORTS = (11555, 11556, 11557, 11580, 8301, 8302)
MIN_FREE_GPU_MIB = 2048 + 4400 + 600  # SONIC, the local VLM, the renderer
STOP_AFTER_RAISE_PROMPT_S = 1.8  # 0.5 s answer + ~0.05 s RPC + 1.0 s blend to the raise start -> ~0.25 s into the raise
CLEAR_WAIT_S = 1.5  # R21: presses in the first 1.0 s at "clear?" are ignored
LEFTOVER_WAIT_S = 10.0
SESSION_TIMEOUT_S = 1200.0


def clean_env(gpu):
    env = {k: v for k, v in os.environ.items() if k != "__EGL_VENDOR_LIBRARY_DIRS"}
    env["LD_LIBRARY_PATH"] = ":".join(p for p in env.get("LD_LIBRARY_PATH", "").split(":") if p and "/snap/" not in p)
    env.update(CUDA_VISIBLE_DEVICES=str(gpu), MUJOCO_GL="egl")
    return env


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--policy-endpoint", required=True, help="host:port of the running s08 sonic_policy_server.py")
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--box-front-m", type=float, default=0.28)
    parser.add_argument("--box-height-m", type=float, default=0.845)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--dds-domain", type=int, default=37)
    parser.add_argument("--no-vlm-server", action="store_true",
                        help="Do not start the local VLM server (another one already serves :8301; the primary run "
                             "uses the oracle, and vlm_side_check.py runs against :8301 afterwards).")
    args = parser.parse_args()
    gpu = subprocess.run(["nvidia-smi", "--id", str(args.gpu), "--query-gpu=index,name,memory.free",
                          "--format=csv,noheader,nounits"], check=True, capture_output=True, text=True).stdout.strip()
    need = MIN_FREE_GPU_MIB - (4400 if args.no_vlm_server else 0)
    if int(gpu.rsplit(",", 1)[1]) < need:
        raise RuntimeError(f"Need {need} MiB free: {gpu}")
    for port in PORTS if not args.no_vlm_server else [p for p in PORTS if p != 8301]:
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", port))
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    session_dir = output / "session"
    env = clean_env(args.gpu)
    bottle_min_x = args.box_front_m + BOTTLE_RADIUS_M + 0.01
    policy_host, policy_port = args.policy_endpoint.rsplit(":", 1)
    subprocess.run([str(ORIGIN / ".venv_sim/bin/python"), "-c",
                    "from scene import build_scene; import sys; build_scene(sys.argv[1], desk_front_m=float(sys.argv[2]), "
                    "desk_height_m=float(sys.argv[3]))", str(output / "scene.xml"), str(args.box_front_m),
                    str(args.box_height_m)], cwd=ART / "g1_handover_20261012_acceptance", check=True, env=env)
    scene = restyle_scene(output / "scene.xml", args.box_front_m, args.box_height_m,
                          json.loads((ROOT / "d435i_head.json").read_text()))
    data = yaml.safe_load(PROFILE_TEMPLATE.read_text())
    data.update(policy_host=policy_host, policy_port=int(policy_port))
    data["handover"].update(checkpoint_verified=True,  # R5: the simulation copy only; the repo profile keeps false
                            desk_top_above_floor_m=args.box_height_m, desk_front_from_pelvis_m=args.box_front_m)
    profile_path = output / "simulation-profile.yaml"
    profile_path.write_text("# SIMULATION TEST FIXTURE ONLY: gates do not qualify a checkpoint or physical return.\n"
                            + yaml.safe_dump(data, sort_keys=False))
    profile = load_profile(profile_path)
    endpoint = "ipc:///tmp/g1-autonomous-loop-" + uuid.uuid4().hex + ".sock"
    binary = NATIVE / "gear_sonic_deploy/target/release/g1_deploy_onnx_ref"
    git = lambda repo: subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()  # noqa: E731
    result = dict(completed=False, trials=args.trials, policy_endpoint=args.policy_endpoint,
                  checkpoint=profile.checkpoint, dds_domain=args.dds_domain, gpu_preflight=gpu, scene=scene,
                  box_front_m=args.box_front_m, box_height_m=args.box_height_m,
                  bottle_pelvis_frame_m=list(BOTTLE_PELVIS_M), served_policy=True, rendered_head_camera=True,
                  rearm_vision="scene-truth oracle (oracle_vlm_server.py) behind RearmMonitor",
                  scripted_person=True, scripted_console=True, scripted_verdicts=True,
                  hardware_acceptance=False, robot_connection=False,
                  native_commit=git(NATIVE), volo_commit=git(VOLO),
                  binary=str(binary), binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest())
    context = zmq.Context()
    keyboard = context.socket(zmq.PUB)
    keyboard.setsockopt(zmq.LINGER, 0)
    keyboard.bind("tcp://127.0.0.1:11580")
    master, slave = pty.openpty()
    processes, logs, threads = {}, [], []
    events = (output / "events.jsonl").open("x")
    lock = threading.Lock()
    stop = threading.Event()

    def record(event, **fields):
        with lock:
            events.write(json.dumps(dict(at=time.monotonic(), wall=time.time(), event=event, **fields), default=str) + "\n")
            events.flush()

    def launch(command, name, stdin=None, stdout=None, cwd=NATIVE, extra_env=None):
        log = (output / (name + ".log")).open("x")
        logs.append(log)
        process = subprocess.Popen(command, cwd=cwd, stdin=stdin, stdout=stdout or log, stderr=log,
                                   start_new_session=True, env={**env, **(extra_env or {})},
                                   text=stdout is not None or stdin == subprocess.PIPE, bufsize=1 if stdout else -1)
        processes[name] = process
        record("launch", name=name, command=command, pid=process.pid)
        return process

    def wait_for(predicate, seconds, what, watch=None):
        deadline = time.monotonic() + seconds
        while not predicate():
            dead = [n for n in (watch or processes) if processes[n].poll() is not None]
            if dead:
                raise RuntimeError(f"{dead} exited while waiting for {what}; inspect logs")
            if time.monotonic() > deadline:
                raise TimeoutError(what)
            time.sleep(0.1)

    def spawn(target, *a):
        def guarded():
            try:
                target(*a)
            except Exception as exc:  # a dead fixture thread must be visible in the evidence
                record("thread_error", thread=target.__name__, error=f"{type(exc).__name__}: {exc}")
                raise

        thread = threading.Thread(target=guarded, daemon=True)
        thread.start()
        threads.append(thread)

    def deploy_tail():
        """Timestamp the deploy's stdout lines (its log has no clock) for the take-over and dead-stream checks."""
        path, offset, partial = output / "sonic.log", 0, ""
        with (output / "sonic-lines.jsonl").open("x") as out:
            while not stop.is_set() or offset < path.stat().st_size:
                with path.open("rb") as f:
                    f.seek(offset)
                    chunk = f.read()
                offset += len(chunk)
                text = partial + chunk.decode(errors="replace")
                *lines, partial = text.split("\n")
                now = time.monotonic()
                for line in lines:
                    out.write(json.dumps(dict(at=now, line=line)) + "\n")
                out.flush()
                if stop.is_set() and not chunk:
                    break
                time.sleep(0.05)

    py_volo = str(VOLO / ".venv/bin/python")
    cli = [py_volo, "-u", "-m", "vlm_orchestrator.harness.g1.cli", "--profile", str(profile_path), "--endpoint", endpoint]
    volo_env = {"PYTHONPATH": str(VOLO)}
    outcome = {}  # trial -> (verdict line, outcome)
    person_state = dict(leftover=None, takes=[], removals=[], placements=[])
    try:
        policy = ORIGIN / "gear_sonic_deploy/policy/sonic_v1_1"
        if not args.no_vlm_server:
            launch([str(LEROBOT_PY), str(VOLO / "scripts/g1_local_vlm_server.py"), "--model", str(VLM_SNAPSHOT),
                    "--port", "8301"], "vlm-server", cwd=VOLO, extra_env={"HF_HUB_OFFLINE": "1"})
        launch([py_volo, str(ROOT / "oracle_vlm_server.py"), "--output", str(output), "--port", "8302",
                "--box-front-m", str(args.box_front_m), "--box-height-m", str(args.box_height_m)], "oracle", cwd=ROOT)
        launch([str(ORIGIN / ".venv_sim/bin/python"), str(ROOT / "loop_sim_worker.py"), "--output", str(output),
                "--scene", str(output / "scene.xml"), "--box-height-m", str(args.box_height_m),
                "--bottle-forward-m", str(BOTTLE_PELVIS_M[0]), "--bottle-left-m", str(BOTTLE_PELVIS_M[1]),
                "--bottle-min-x-m", str(bottle_min_x), "--bottle-half-height-m", str(BOTTLE_HALF_HEIGHT_M),
                "--camera-port", "11555", "--dds-domain", str(args.dds_domain)], "mujoco",
                extra_env={"PYTHONPATH": str(NATIVE)})
        launch([str(binary), "lo", str(policy / "model_decoder.onnx"), str(ORIGIN / "gear_sonic_deploy/reference/example"),
                "--encoder-file", str(policy / "model_encoder.onnx"), "--planner-file",
                str(ORIGIN / "gear_sonic_deploy/planner/target_vel/V2/planner_sonic.onnx"),
                "--obs-config", str(policy / "observation_config.yaml"), "--input-type", "zmq_manager",
                "--harness-planner-hold", "--output-type", "zmq", "--disable-crc-check",
                "--dds-domain", str(args.dds_domain), "--zmq-host", "127.0.0.1", "--zmq-port", "11556",
                "--zmq-out-port", "11557", "--logs-dir", str(output / "controller")], "sonic", slave)
        spawn(deploy_tail)
        launch([str(NATIVE / ".venv/bin/python"), str(ROOT / "loop_native_worker.py"), "--output", str(output),
                "--profile", str(profile_path), "--endpoint", endpoint], "native",
               extra_env={"PYTHONPATH": str(NATIVE)})
        wait_for(lambda: Path(endpoint[6:]).exists() and "Init Done" in (output / "sonic.log").read_text(),
                 90, "controller initialization")
        client = G1Client(endpoint)
        time.sleep(0.3)
        keyboard.send_string("k")
        wait_for(lambda: (lambda s: s.controller_running and s.policy_ready and s.telemetry_age_s is not None
                          and s.telemetry_age_s < 0.5)(client.get_status()), 90, "controller running and policy ready")
        (output / "startup-request").touch()
        wait_for(lambda: (output / "startup-receipt.json").exists(), 10, "grounded startup")
        time.sleep(2.0)
        (output / "box-restore-1").touch()
        wait_for(lambda: (output / "box-restore-1.json").exists(), 5, "box restore decision")
        result["box_restore"] = json.loads((output / "box-restore-1.json").read_text())
        if not result["box_restore"]["restored"]:
            raise RuntimeError("The case could not be restored next to the standing robot")
        if not args.no_vlm_server:
            wait_for(lambda: "serving" in (output / "vlm-server.log").read_text(), 180, "local VLM server",
                     watch=["vlm-server"])
        result["vlm_server_started_here"] = not args.no_vlm_server
        record("startup_done", status=asdict(client.get_status()))

        # --- the session --------------------------------------------------------------------------------------
        verdict = launch([*cli, "verdict", "--session-dir", str(session_dir)], "verdict",
                         stdin=subprocess.PIPE, stdout=subprocess.PIPE, cwd=VOLO, extra_env=volo_env)
        runner = launch([*cli, "loop", "--session-dir", str(session_dir), "--trials", str(args.trials),
                         "--vlm-base-url", "http://127.0.0.1:8302/v1", "--vlm-model", "sim-oracle"], "loop",
                        stdin=subprocess.PIPE, stdout=subprocess.PIPE, cwd=VOLO, extra_env=volo_env)

        def press(why, prompt=None):
            runner.stdin.write("\n")
            runner.stdin.flush()
            record("console_press", why=why, prompt=prompt)

        def console():
            """Scripted operator: answers every gate; one stop mid-raise; waits > 1 s at "clear?" (R21)."""
            stopped = False
            with (output / "loop-stdout.log").open("x") as copy:
                for line in runner.stdout:
                    copy.write(line)
                    copy.flush()
                    text = line.strip()
                    record("console_line", line=text)
                    if text.startswith("clear?"):
                        time.sleep(CLEAR_WAIT_S)
                        press("clear", text)
                    elif text.startswith(("deploy ready?", "robot in position?", "start loop?", "restart loop?",
                                          "lower?")):
                        time.sleep(0.5)
                        press("gate", text)
                    elif text.startswith("raise?"):
                        time.sleep(0.5)
                        press("gate", text)
                        if not stopped:
                            stopped = True
                            time.sleep(STOP_AFTER_RAISE_PROMPT_S - 0.5)
                            press("stop mid-raise")

        def evidence_step():
            """(trial, last re-arm/trial event) of the runner's newest evidence file."""
            files = sorted(session_dir.glob("trial_*_evidence.jsonl"))
            if not files:
                return None, None
            last = None
            for line in files[-1].read_text().splitlines():
                event = json.loads(line)
                if event["kind"] in ("rearm_begin", "rearm_end", "trial_start"):
                    last = event
            return int(files[-1].name[6:9]), last

        def person():
            """Places a bottle when the runner waits for one, takes it from the hand, clears the case after a trial.
            Once (the first trial end with a bottle still standing on the case) it waits LEFTOVER_WAIT_S first."""
            watcher = G1Client(endpoint)
            placed = taken = 0
            hand_since = hand_z = palm_since = remove_at = place_at = None
            placed_for, judged = None, set()
            try:
                while not stop.is_set():
                    time.sleep(0.1)
                    try:
                        s = watcher.get_status()
                        truth = json.loads((output / "bottle-state.json").read_text())
                    except (RPCError, OSError, ValueError):
                        continue
                    now, bottle = time.monotonic(), truth["bottle"]
                    at_ready = (s.phase, s.loop_stage) == ("LOOP_HOLD", "ready")
                    trial = s.loop_trial

                    def take(why):
                        nonlocal taken
                        taken += 1
                        (output / f"take-bottle-{taken}").touch()
                        entry = dict(at=now, n=taken, why=why, bottle=bottle, trial=trial, phase=s.phase,
                                     stage=s.loop_stage, palm=truth["palm_position"])
                        (person_state["removals"] if at_ready or s.phase != "LOOP_TRIAL" else person_state["takes"]).append(entry)
                        record("person_take", **{k: v for k, v in entry.items() if k != "at"})

                    # during a trial: take the bottle from the hand (lifted 5 cm, held 2 s) or from the open palm
                    if s.phase == "LOOP_TRIAL" and bottle == "in_hand":
                        if hand_since is None:
                            hand_since, hand_z = now, truth["palm_position"][2]
                        if now - hand_since >= 2.0 and truth["palm_position"][2] - hand_z >= 0.05:
                            take("from the hand")
                    else:
                        hand_since = None
                    if s.phase == "LOOP_TRIAL" and bottle == "on_palm":
                        palm_since = palm_since or now
                        if now - palm_since >= 1.0:
                            take("from the open palm")
                    else:
                        palm_since = None
                    # after a trial (its files written, robot back at ready): judge it, then clear the case
                    if at_ready and trial is not None and trial not in judged and \
                            (session_dir / f"trial_{trial:03d}_frames.npz").exists():
                        judged.add(trial)
                        took = [t for t in person_state["takes"] if t["trial"] == trial]
                        if took:
                            outcome[trial] = ("s", f"person took the bottle {took[0]['why']}")
                        elif bottle == "on_box":
                            forced = (output / f"no-grasp-{placed}").exists()
                            outcome[trial] = ("f 8 fixture: grasp disabled (leftover-bottle trial)" if forced else
                                              "f 1 sim: bottle not grasped", "bottle left standing")
                        elif bottle in ("dropped", "fallen"):
                            outcome[trial] = ("f 3 sim: bottle released palm-down", f"bottle {bottle}")
                        elif bottle in ("in_hand", "on_palm"):
                            outcome[trial] = ("f 6 sim: still held at the trial end", f"bottle {bottle}")
                        else:
                            outcome[trial] = ("f 8 sim: no bottle", f"bottle {bottle}")
                        record("person_judged", trial=trial, verdict=outcome[trial][0], why=outcome[trial][1])
                        if bottle != "taken":
                            leftover = bottle == "on_box" and person_state["leftover"] is None
                            remove_at = now + (LEFTOVER_WAIT_S if leftover else 1.0)
                            if leftover:
                                person_state["leftover"] = dict(trial=trial, seen_at=now, remove_at=remove_at)
                                record("person_leftover_begin", trial=trial, remove_at=remove_at)
                    if remove_at is not None and now >= remove_at and bottle != "taken":
                        take("clear the case")
                        remove_at = None
                        if person_state["leftover"] and "removed_at" not in person_state["leftover"]:
                            person_state["leftover"]["removed_at"] = now
                    # place: the runner waits for a bottle (re-arm step "bottle") and the case is clear
                    n, last = evidence_step()
                    waiting = at_ready and last is not None and last["kind"] == "rearm_begin" and last["step"] == "bottle"
                    if waiting and bottle == "taken" and placed_for != n and remove_at is None:
                        place_at = place_at or now + 1.0
                        if now >= place_at:
                            placed += 1
                            if n == 1 and person_state["leftover"] is None:  # force the leftover scenario once
                                (output / f"no-grasp-{placed}").touch()
                                record("person_no_grasp", placement=placed, trial=n)
                            (output / f"place-bottle-{placed}").touch()
                            placed_for, place_at = n, None
                            person_state["placements"].append(dict(at=now, n=placed, trial=n))
                            record("person_place", n=placed, trial=n)
                    else:
                        place_at = None
            finally:
                watcher.close()

        def verdicts():
            """Scripted verdict console: scores each trial from the person's judgement; EOF after the last one."""
            scored = 0
            with (output / "verdict-stdout.log").open("x") as copy:
                for line in verdict.stdout:
                    copy.write(line)
                    copy.flush()
                    record("verdict_line", line=line.strip())
                    if line.startswith("trial "):
                        trial = int(line.split()[1].rstrip(":"))
                        while trial not in outcome and not stop.is_set():
                            time.sleep(0.1)
                        if trial in outcome:
                            verdict.stdin.write(outcome[trial][0] + "\n")
                            verdict.stdin.flush()
                            record("verdict_sent", trial=trial, line=outcome[trial][0])
                            scored += 1
                            if scored == args.trials:
                                time.sleep(1.0)
                                verdict.stdin.close()
                                return

        spawn(console)
        spawn(person)
        spawn(verdicts)
        record("session_start")
        runner.wait(timeout=SESSION_TIMEOUT_S)
        record("session_end", returncode=runner.returncode)
        result["loop_returncode"] = runner.returncode
        try:
            verdict.wait(timeout=60)
        except subprocess.TimeoutExpired:
            record("verdict_console_timeout")
        result["verdict_returncode"] = verdict.poll()
        result["person"] = person_state
        result["outcomes"] = {str(k): v for k, v in outcome.items()}
        result["status_after_session"] = asdict(client.get_status())
        time.sleep(2.0)

        # --- R19: the deploy's reaction to a dead pose stream ------------------------------------------------
        sid = "dead-stream-" + uuid.uuid4().hex[:8]
        lease = client.request("claim_control", {"registry_sha256": profile.registry_sha256}, session_id=sid).result
        beat = threading.Event()

        def heartbeat():
            sequence = 0
            hb = G1Client(endpoint)
            try:
                while not beat.wait(0.2):
                    sequence += 1
                    try:
                        hb.request("heartbeat", {"sequence": sequence}, session_id=sid, lease_id=lease.lease_id)
                    except RPCError:
                        pass
            finally:
                hb.close()

        spawn(heartbeat)
        response = client.request("loop_take_over", {}, session_id=sid, lease_id=lease.lease_id)
        if response.error:
            raise RuntimeError(f"dead-stream take-over refused: {response.error}")
        record("dead_stream_take_over")
        wait_for(lambda: (lambda s: (s.phase, s.loop_stage) == ("LOOP_HOLD", "stance"))(client.get_status()), 10,
                 "dead-stream stance", watch=["mujoco", "sonic", "native"])
        time.sleep(1.0)
        before = asdict(client.get_status())
        killed_at = time.monotonic()
        os.killpg(processes["native"].pid, signal.SIGKILL)
        beat.set()
        processes["native"].wait()
        record("dead_stream_native_killed", status_before=before)
        time.sleep(3.5)
        record("dead_stream_observed", seconds=3.5)
        result["dead_stream"] = dict(killed_at=killed_at, status_before=before,
                                     deploy_alive=processes["sonic"].poll() is None,
                                     sim_alive=processes["mujoco"].poll() is None)
        result["completed"] = True
    except Exception as error:
        result["error"] = f"{type(error).__name__}: {error}"
        record("error", error=result["error"])
    finally:
        stop.set()
        (output / "stop-simulator").touch()
        for name in reversed(list(processes)):
            process = processes[name]
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGINT if name in ("loop", "verdict") else signal.SIGTERM)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
        for thread in threads:
            thread.join(timeout=5)
        for log in logs:
            log.close()
        keyboard.close()
        context.term()
        os.close(master)
        os.close(slave)
        events.close()
        result["owned_process_pids"] = {name: p.pid for name, p in processes.items()}
        result["owned_process_exit_codes"] = {name: p.returncode for name, p in processes.items()}
        result["owned_processes_stopped"] = all(p.poll() is not None for p in processes.values())
        (output / "result.json").write_text(json.dumps(result, indent=2, default=str) + "\n")
    print(json.dumps({k: result.get(k) for k in ("completed", "loop_returncode", "error")}, indent=2))
    return 0 if result.get("completed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
