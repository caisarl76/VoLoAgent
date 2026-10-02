"""Bounded SONIC/MuJoCo probe; loopback DDS only, isolated test ZMQ ports.

Run with the native worktree Python and explicit PYTHONPATH. This intentionally
starts simulated control and requires the deployment/sim workflow's approval.
"""

import argparse
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import pty
import signal
import subprocess
import time
import uuid

import numpy as np
import zmq

from gear_sonic.utils.data_collection.zmq_state_subscriber import ZMQStateSubscriber
from gear_sonic.utils.inference.harness_control import HarnessControl, RuntimeFacts
from gear_sonic.utils.inference.bounded_planner import wrap
from gear_sonic.utils.inference.planner_heading_frame import (
    planner_command_in_reference_frame,
)
from gear_sonic.utils.inference.standing_reset import StandingReset, _measured_joints
from gear_sonic.utils.teleop.xr_upperbody_bridge import feedback_payload_heading_yaw
from gear_sonic.utils.teleop.zmq.zmq_planner_sender import build_command_message


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--native", required=True, type=Path)
    parser.add_argument("--profile", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--release-band", action="store_true")
    parser.add_argument("--sonic-policy", choices=("release", "sonic_v1_1"), default="release")
    parser.add_argument("--initial-yaw-deg", type=float, default=0.0)
    parser.add_argument("--turn-angle-deg", type=float, default=15.0)
    parser.add_argument("--turn-rate-deg-s", type=float, default=10.0)
    parser.add_argument("--prepare-standing", action="store_true")
    parser.add_argument("--diagnostic-turn-mode", type=int, choices=(0, 1))
    parser.add_argument(
        "--scenario", choices=("transitions", "faults", "baseline"), default="transitions"
    )
    args = parser.parse_args()
    if not np.isfinite(args.initial_yaw_deg) or not -180 <= args.initial_yaw_deg <= 180:
        parser.error("Initial simulator heading must be finite and within +/-180 degrees")
    if not np.isfinite(args.turn_angle_deg) or not 0 < abs(args.turn_angle_deg) <= 45:
        parser.error("Turn angle must be finite and within the nonzero +/-45 degree bound")
    if not np.isfinite(args.turn_rate_deg_s) or not 0 < args.turn_rate_deg_s <= 10:
        parser.error("Turn rate must be finite and within the positive 10 degree/s bound")
    origin = Path("/home/jihun/work/GR00T-WholeBodyControl")
    args.output.mkdir(parents=True, exist_ok=False)
    events = (args.output / "events.jsonl").open("w")
    logs, processes = [], []
    context = zmq.Context()
    pub = context.socket(zmq.PUB)
    pub.setsockopt(zmq.LINGER, 0)
    pub.bind("tcp://127.0.0.1:11556")
    subscriber = ZMQStateSubscriber("127.0.0.1", 11557)
    master, slave = pty.openpty()
    result = {
        "outcome": "failed",
        "gate": "initialization",
        "robot_connection": False,
        "dds_interface": "lo",
        "action_port": 11556,
        "state_port": 11557,
        "scenario": args.scenario,
        "gpu": args.gpu,
        "release_band": args.release_band,
        "initial_yaw_deg": args.initial_yaw_deg,
        "turn_angle_deg": args.turn_angle_deg,
        "turn_rate_deg_s": args.turn_rate_deg_s,
        "prepare_standing": args.prepare_standing,
        "diagnostic_turn_mode": args.diagnostic_turn_mode,
    }
    binary = args.native / "gear_sonic_deploy/target/release/g1_deploy_onnx_ref"
    sonic_policy = origin / "gear_sonic_deploy/policy" / args.sonic_policy
    result["sonic_assets"] = {
        name: {"path": str(sonic_policy / name), "sha256": hashlib.sha256((sonic_policy / name).read_bytes()).hexdigest()}
        for name in ("model_decoder.onnx", "model_encoder.onnx", "observation_config.yaml")
    }
    result["controller_binary"] = str(binary)
    result["controller_binary_sha256"] = hashlib.sha256(binary.read_bytes()).hexdigest()

    def record(event, **data):
        events.write(
            json.dumps(
                dict(at=time.monotonic(), event=event, **data),
                default=lambda value: value.tolist() if isinstance(value, np.ndarray) else str(value),
            )
            + "\n"
        )
        events.flush()

    def launch(command, name, cwd, env, stdin=None):
        log = (args.output / (name + ".log")).open("w")
        logs.append(log)
        process = subprocess.Popen(
            command,
            cwd=cwd,
            env=env,
            stdin=stdin or subprocess.DEVNULL,
            stdout=log,
            stderr=log,
            start_new_session=True,
        )
        processes.append(process)
        record("launch", name=name, pid=process.pid, command=command)

    class Hooks:
        def __init__(self):
            self.epoch = 0
            self.command = self.reset = None
            self.last_reset = None
            self.feedback = None
            self.running = False

        def runtime_facts(self):
            return RuntimeFacts(self.running, False, False, False, "PLANNER")

        def invalidate_policy_actions(self):
            self.epoch += 1
            return self.epoch

        def set_policy_enabled(self, enabled):
            assert not enabled, "This probe has no VLA publisher"

        def set_policy_prompt(self, prompt):
            raise ValueError("Manipulation is outside this planner probe")

        def request_planner_hold(self, feedback, open_hands):
            self.command = StandingReset(
                feedback,
                feedback.get("left_hand_q_measured", feedback.get("left_hand_q")),
                feedback.get("right_hand_q_measured", feedback.get("right_hand_q")),
            ).command
            self.reset = None
            pub.send(
                planner_command_in_reference_frame(self.command, feedback).encode()
            )
            pub.send(build_command_message(start=True, stop=False, planner=True))

        def begin_standing_reset(self, feedback, open_hands):
            self.reset = StandingReset(
                feedback,
                np.zeros(7)
                if open_hands
                else feedback.get("left_hand_q_measured", feedback.get("left_hand_q")),
                np.zeros(7)
                if open_hands
                else feedback.get(
                    "right_hand_q_measured", feedback.get("right_hand_q")
                ),
                compensate_tracking_bias=True,
                joint_tolerance_rad=control.profile.limits.reset_joint_tolerance_rad,
            )
            self.command = self.reset.command
            self.last_reset = self.reset
            pub.send(
                planner_command_in_reference_frame(self.command, feedback).encode()
            )
            pub.send(build_command_message(start=True, stop=False, planner=True))
            return self.reset

        def set_planner_command(self, command):
            if control.phase == "TURNING" and args.diagnostic_turn_mode is not None:
                command = replace(command, mode=args.diagnostic_turn_mode)
            self.reset, self.command = None, command

        def stop_planner_motion(self):
            if self.command is not None:
                self.command = replace(
                    self.command, mode=0, movement=(0.0, 0.0, 0.0), speed=0.0
                )
            self.reset = None

    hooks = Hooks()
    control = HarnessControl(args.profile, hooks, uuid.uuid4().hex)
    control.locomotion_enabled = True
    session, lease = "simulation-probe", None

    def request(method, params):
        nonlocal lease
        response = control.dispatch(
            dict(
                version=1,
                request_id=uuid.uuid4().hex,
                runtime_id=control.runtime_id,
                session_id=session,
                lease_id=lease,
                method=method,
                params=params,
            ),
            time.monotonic(),
        )
        record("request", method=method, response=response)
        if response["error"]:
            raise RuntimeError(response["error"])
        if method == "claim_control":
            lease = response["result"]["lease_id"]
        return response["result"]

    def pump(
        duration,
        stop_when=None,
        heartbeat=True,
        publish=True,
        prepare=False,
        drop_feedback=False,
        allow_fault=False,
        operator_reset=False,
    ):
        deadline, last_at, heartbeat_at = (
            time.monotonic() + duration,
            time.monotonic(),
            0.0,
        )
        sample_at = 0.0
        while time.monotonic() < deadline:
            now = time.monotonic()
            for process in processes:
                if process.poll() is not None:
                    raise RuntimeError(
                        f"Simulation component exited {process.returncode}"
                    )
            feedback = subscriber.get_msg()
            if drop_feedback:
                feedback = None
            if feedback is not None:
                hooks.feedback = feedback
            control.tick(now, feedback, now if feedback is not None else None)
            if prepare and now >= heartbeat_at:
                # C++ publishes measured telemetry only after operator startup.
                # PLANNER starts with its stationary IDLE movement defaults.
                pub.send(build_command_message(start=True, stop=False, planner=True))
                heartbeat_at = now + 0.25
            if control.owner is not None and heartbeat and now >= heartbeat_at:
                request("heartbeat", {})
                heartbeat_at = now + 0.25
            if (
                hooks.reset is not None
                and (
                    (control.hold_confirmed and control.phase == "RESETTING")
                    or (
                        operator_reset and feedback is not None
                        and feedback["planner_reference_active"][0] == 1
                    )
                )
            ):
                hooks.command = hooks.reset.advance(hooks.feedback, now - last_at)
            if hooks.command is not None and hooks.feedback is not None and publish:
                pub.send(
                    planner_command_in_reference_frame(
                        hooks.command, hooks.feedback
                    ).encode()
                )
            if feedback is not None:
                record("status", **control.status(now))
                if now >= sample_at:
                    record(
                        "measurement",
                        feedback=feedback,
                        command=None if hooks.command is None else hooks.command.__dict__,
                    )
                    sample_at = now + 0.1
            if stop_when is not None and stop_when():
                return
            if control.phase == "FAULT" and not allow_fault:
                raise RuntimeError(control.reason)
            last_at = now
            time.sleep(0.02)
        if stop_when is not None:
            raise TimeoutError("Measured/acknowledged condition did not complete")

    try:
        env = {
            **os.environ,
            "PYTHONPATH": str(args.native),
            "CUDA_VISIBLE_DEVICES": str(args.gpu),
        }
        launch(
            [
                str(origin / ".venv_sim/bin/python"),
                str(Path(__file__).with_name("sonic_sim_worker.py")),
                "--output",
                str(args.output),
                "--initial-yaw-deg",
                str(args.initial_yaw_deg),
            ],
            "mujoco",
            args.native,
            env,
        )
        deploy = origin / "gear_sonic_deploy"
        launch(
            [
                str(binary),
                "lo",
                str(sonic_policy / "model_decoder.onnx"),
                str(deploy / "reference/example"),
                "--encoder-file",
                str(sonic_policy / "model_encoder.onnx"),
                "--planner-file",
                str(deploy / "planner/target_vel/V2/planner_sonic.onnx"),
                "--obs-config",
                str(sonic_policy / "observation_config.yaml"),
                "--input-type",
                "zmq_manager",
                "--harness-planner-hold",
                "--output-type",
                "zmq",
                "--disable-crc-check",
                "--zmq-host",
                "127.0.0.1",
                "--zmq-port",
                "11556",
                "--zmq-out-port",
                "11557",
                "--logs-dir",
                str(args.output / "controller"),
            ],
            "sonic",
            args.native,
            env,
            slave,
        )
        hooks.running = True
        pump(
            30,
            lambda: control.feedback is not None and control._fresh(time.monotonic()),
            prepare=True,
        )
        if args.scenario == "baseline":
            # Stationary target without any harness lease/reset transition.
            baseline = StandingReset(control.feedback, np.zeros(7), np.zeros(7))
            hooks.command = replace(
                baseline.command,
                upper_body_position=baseline.target[:17].tolist(),
                left_hand_position=[0.0] * 7,
                right_hand_position=[0.0] * 7,
            )
            record("baseline_start", target=baseline.target.tolist())
            if args.release_band:
                # Match the normal viewer's operator key 9 after startup.
                (args.output / "release-band").touch()
            pump(12)
            errors = _measured_joints(hooks.feedback) - baseline.target
            result.update(
                outcome="observed",
                gate="stationary_baseline",
                baseline_joint_error_rad=errors.tolist(),
                baseline_upper_max_error_rad=float(np.max(np.abs(errors[:17]))),
                baseline_hand_max_error_rad=float(np.max(np.abs(errors[17:]))),
                baseline_last_feedback=hooks.feedback,
            )
            return 0
        # Native k-style preparation is confined to the loopback simulator.
        hooks.request_planner_hold(control.feedback, False)
        if args.prepare_standing:
            # Simulated operator preparation, before granting an agent lease.
            # Match a legacy k-style standing ramp before lowering the support.
            hooks.reset = StandingReset(
                control.feedback,
                control.feedback["left_hand_q_measured"],
                control.feedback["right_hand_q_measured"],
            )
            record("operator_standing_preparation", target=hooks.reset.target)
            pump(4, operator_reset=True)
        else:
            pump(2)
        if args.release_band:
            (args.output / "release-band").touch()
            pump(2, operator_reset=args.prepare_standing)
        if args.prepare_standing:
            hooks.request_planner_hold(control.feedback, False)
        request("claim_control", {"registry_sha256": control.profile.registry_sha256})
        if args.scenario == "faults":
            result["gate"] = "cancellation"
            execution = request(
                "reset_standing", {"execution_id": "", "open_hands": False}
            )
            pump(0.5)
            request("cancel", {"execution_id": execution["execution_id"]})
            pump(2, lambda: control.phase == "INTERRUPTED" and control.hold_confirmed)
            assert hooks.command.speed == 0.0
            result["cancellation"] = "passed"

            request("release_control", {})
            control.operator_override("i")  # New simulated operator preparation.
            request(
                "claim_control", {"registry_sha256": control.profile.registry_sha256}
            )
            request("reset_standing", {"execution_id": "", "open_hands": False})
            result["gate"] = "coordinator_loss"
            pump(
                3,
                lambda: control.owner is None and control.hold_confirmed,
                heartbeat=False,
            )
            assert control.phase == "INTERRUPTED" and control.reason == "lease_expired"
            assert hooks.command.speed == 0.0
            result["coordinator_loss"] = "passed"

            control.operator_override("i")
            request(
                "claim_control", {"registry_sha256": control.profile.registry_sha256}
            )
            request("reset_standing", {"execution_id": "", "open_hands": False})
            pump(0.5)
            result["gate"] = "stale_feedback"
            pump(1.5, lambda: control.phase == "FAULT", drop_feedback=True)
            assert control.reason == "stale_feedback" and not control.hold_confirmed
            assert hooks.command.speed == 0.0
            result["stale_feedback"] = "passed"
            pump(
                3,
                lambda: control.owner is None and control.hold_confirmed,
                heartbeat=False,
                allow_fault=True,
            )

            result["gate"] = "lost_planner_input"
            # An open-hand baseline equals C++'s fallback and cannot detect
            # override loss. Establish a distinct, measured hand pose first.
            hand_target = _measured_joints(hooks.feedback)[17:] * 0.5
            hooks.command = replace(
                hooks.command,
                left_hand_position=hand_target[:7].tolist(),
                right_hand_position=hand_target[7:].tolist(),
            )
            # Let the physical hand finish approaching the target before
            # measuring drift; stopping early confuses convergence with loss.
            pump(3)
            before = _measured_joints(hooks.feedback)[17:]
            pump(2.5, heartbeat=False, publish=False)
            after = _measured_joints(hooks.feedback)[17:]
            hand_error = float(np.max(np.abs(after - before)))
            result["lost_input_hand_change_rad"] = hand_error
            result["lost_input_hands_before_rad"] = before.tolist()
            result["lost_input_hands_after_rad"] = after.tolist()
            record("lost_input_measurement", **result)
            assert hand_error <= control.profile.limits.reset_joint_tolerance_rad, (
                "C++ lost-input fallback did not preserve measured hands"
            )
            result["lost_planner_input"] = "passed_measured_hand_tolerance_only"
            result["outcome"], result["gate"] = "passed", "faults"
            return 0
        result["gate"] = "standing_reset"
        reset_started_at = time.monotonic()
        request("reset_standing", {"execution_id": "", "open_hands": False})
        pump(16, lambda: control.phase in {"COMPLETED", "INTERRUPTED"})
        assert control.phase == "COMPLETED", control.reason
        result["standing_reset"] = "passed"
        errors = _measured_joints(hooks.feedback) - hooks.last_reset.target
        result["reset_elapsed_s"] = time.monotonic() - reset_started_at
        result["reset_joint_error_rad"] = errors.tolist()
        result["reset_max_joint_error_rad"] = float(np.max(np.abs(errors)))
        result["reset_target_rad"] = hooks.last_reset.target.tolist()
        result["reset_command_rad"] = hooks.last_reset._position.tolist()
        result["gate"] = "walking"
        request(
            "walk_for", {"direction": "backward", "duration_s": 1.0, "speed_mps": 0.2}
        )
        pump(3, lambda: control.phase == "COMPLETED")
        result["walking"] = "passed_duration_and_stop_ack"
        result["gate"] = "turning"
        initial_yaw = feedback_payload_heading_yaw(hooks.feedback)
        angle = float(np.deg2rad(args.turn_angle_deg))
        turn_goal = wrap(initial_yaw + angle)
        turn_started_at = time.monotonic()
        request("turn_by", {"angle_rad": angle, "rate_rps": float(np.deg2rad(args.turn_rate_deg_s))})
        pump(11, lambda: control.phase in {"COMPLETED", "INTERRUPTED"})
        result["turn_elapsed_s"] = time.monotonic() - turn_started_at
        result["turn_goal_yaw_rad"] = turn_goal
        result["turn_initial_yaw_rad"] = initial_yaw
        result["turn_crosses_wrap_boundary"] = abs(initial_yaw + angle) > np.pi
        result["turn_measured_yaw_rad"] = feedback_payload_heading_yaw(hooks.feedback)
        result["turn_error_rad"] = abs(wrap(result["turn_measured_yaw_rad"] - turn_goal))
        assert control.phase == "COMPLETED", control.reason
        result["turning"] = "passed_measured_dwell_and_stop_ack"
        result["gate"] = "coordinator_loss"
        request(
            "walk_for", {"direction": "forward", "duration_s": 5.0, "speed_mps": 0.2}
        )
        pump(
            3, lambda: control.owner is None and control.hold_confirmed, heartbeat=False
        )
        assert control.phase == "INTERRUPTED" and hooks.command.speed == 0.0
        result["coordinator_loss"] = "passed"
        result["outcome"], result["gate"] = "passed", "planner_transitions"
    except Exception as exc:
        result["reason"] = str(exc)
        result["status"] = control.status(time.monotonic())
        if hooks.feedback is not None:
            (args.output / "last-feedback.json").write_text(
                json.dumps(
                    hooks.feedback,
                    default=lambda value: value.tolist(),
                    indent=2,
                )
                + "\n"
            )
        reset = hooks.reset or hooks.last_reset
        if result["gate"] == "standing_reset" and reset is not None and hooks.feedback is not None:
            errors = _measured_joints(hooks.feedback) - reset.target
            result["reset_joint_error_rad"] = errors.tolist()
            result["reset_max_joint_error_rad"] = float(np.max(np.abs(errors)))
            result["reset_target_rad"] = reset.target.tolist()
            result["reset_command_rad"] = reset._position.tolist()
        record("failure", **result)
    finally:
        if hooks.running:
            pub.send(build_command_message(start=False, stop=True, planner=True))
        for process in reversed(processes):
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGINT)
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=5)
        subscriber.close()
        pub.close()
        context.term()
        os.close(master)
        os.close(slave)
        for log in logs:
            log.close()
        events.close()
        (args.output / "result.json").write_text(
            json.dumps(result, default=lambda value: value.tolist(), indent=2) + "\n"
        )
    print(json.dumps(result))
    return 0 if result["outcome"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
