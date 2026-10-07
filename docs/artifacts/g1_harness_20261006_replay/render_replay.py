"""Render recorded poses only: no simulator loop, policy, DDS, or mj_step."""

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess

os.environ.setdefault("MUJOCO_GL", "egl")
import cv2
import mujoco
import numpy as np


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def yaw(q):
    w, x, y, z = q
    return math.degrees(math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


def latest(rows, times, at):
    index = int(np.searchsorted(times, at, side="right")) - 1
    return rows[index] if index >= 0 else None


def text(frame, value, x, y, color=(235, 239, 244), scale=0.52):
    cv2.putText(frame, value, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trial", type=Path, required=True)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    states = read_rows(args.trial / "ground-truth.jsonl")
    events = read_rows(args.trial / "events.jsonl")
    native = read_rows(args.trial / "native-trace.jsonl")
    controls = [r for r in read_rows(args.trial / "planner-trace.jsonl") if r["event"] == "control"]
    conversions = [r for r in native if r["event"] == "planner_conversion"]
    turns = [r for r in native if r["event"] == "turn_sample"]
    times = np.array([r["at"] for r in states])
    assert len(states) and np.all(np.diff(times) > 0)
    control_times = np.array([r["telemetry_end"] for r in controls])
    conversion_times = np.array([r["at"] for r in conversions])
    assert np.all(np.diff(control_times) > 0) and np.all(np.diff(conversion_times) > 0)
    milestones = {r["method"]: r["result"]["started_at"] for r in events
                  if r["event"] == "rpc" and r.get("method") in ("reset_standing", "walk_for", "turn_by")}
    milestones["support_released"] = next(r["at"] for r in states if not r["band_enabled"])
    milestones["interrupted"] = next(r["at"] for r in turns if r["phase"] == "INTERRUPTED")
    milestones["last_control_sample"] = float(control_times[-1])
    goal = math.degrees(turns[0]["goal_yaw_rad"])
    model = mujoco.MjModel.from_xml_path(str(args.scene.resolve()))
    data = mujoco.MjData(model)
    # Invert DefaultEnv.prepare_obs's joint-class ordering, not actuator order.
    mappings = {}
    for key, parts in (("body_q", ("hip", "knee", "ankle", "waist", "shoulder", "elbow", "wrist")),
                       ("left_hand_q", ("left_hand",)), ("right_hand_q", ("right_hand",))):
        ids = [i for i in range(model.njnt) if any(part in model.joint(i).name for part in parts)]
        adr = model.jnt_qposadr[ids]
        assert np.array_equal(adr, np.array(ids) + 6), "Recorded observation indexing differs from this model"
        mappings[key] = adr
    assert [len(mappings[k]) for k in mappings] == [29, 7, 7]
    assert sorted([*range(7), *np.concatenate(list(mappings.values()))]) == list(range(model.nq))
    for state in states:
        for key, size in (("floating_base_pose", 7), ("body_q", 29), ("left_hand_q", 7), ("right_hand_q", 7)):
            assert np.asarray(state[key]).shape == (size,) and np.isfinite(state[key]).all()
        assert abs(np.linalg.norm(state["floating_base_pose"][3:]) - 1) < 1e-5
    cameras = []
    for azimuth, elevation, distance, lookat in (
            (20, -16, 2.8, (0.1, 0, 0.65)),
            (90, -14, 2.8, (0.1, 0, 0.65)),
            (0, -80, 2.3, (0.1, 0, 0.15))):
        camera = mujoco.MjvCamera()
        camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        camera.lookat[:] = lookat
        camera.azimuth, camera.elevation, camera.distance = azimuth, elevation, distance
        cameras.append(camera)
    colors = [(80, 220, 255), (255, 175, 75), (235, 105, 220), (235, 239, 244)]
    labels = ["Body: saved torso pose", "Bounded facing: sent to planner", "Target: active controller reference", "Goal: absolute world heading"]
    fps = 10
    first, last = float(times[0]), float(times[-1])
    # Preserve wall-clock gaps with zero-order hold; never interpolate poses.
    frame_times = first + np.arange(math.ceil((last - first) * fps) + 1) / fps
    focus_frame = max(0, math.floor((milestones["support_released"] - first - 1) * fps))
    video = args.output / "full-trial.mp4"
    encoder_log = (args.output / "ffmpeg-render.log").open("w")
    encoder = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo",
        "-pixel_format", "rgb24", "-video_size", "1280x800", "-framerate", str(fps),
        "-i", "pipe:0", "-an", "-c:v", "libx264", "-preset", "fast", "-crf", "19",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(video)], stdin=subprocess.PIPE,
        stderr=encoder_log)
    frame_map = []
    previews = [0, focus_frame, int((milestones["turn_by"] - first + 2) * fps),
                int((milestones["interrupted"] - first) * fps) + 1, len(frame_times) - 1]
    try:
        with mujoco.Renderer(model, height=480, width=420) as renderer:
            for i, at in enumerate(frame_times):
                index = min(len(states) - 1, int(np.searchsorted(times, at, side="right")) - 1)
                state = states[index]
                data.qpos[:7] = state["floating_base_pose"]
                for key, adr in mappings.items():
                    data.qpos[adr] = state[key]
                data.time = state["sim_time"]
                mujoco.mj_forward(model, data)
                control = latest(controls, control_times, state["at"])
                conversion = latest(conversions, conversion_times, state["at"])
                target = yaw(control["target_quat"]) if control and control["planner_motion"] else None
                facing = math.degrees(math.atan2(conversion["world"]["facing"][1], conversion["world"]["facing"][0])) if conversion else None
                body = yaw(data.body("torso_link").xquat)
                current_goal = goal if at >= milestones["turn_by"] else None
                angles = [body, facing, target, current_goal]
                if at > milestones["last_control_sample"]:
                    phase = "SHUTDOWN TAIL / after last recorded control tick"
                    angles[1:3] = [None, None]
                elif at >= milestones["interrupted"]:
                    phase = "TURN INTERRUPTED / stationary command hold"
                elif at >= milestones["turn_by"]:
                    phase = "TURN -15 deg / requested rate <= 10 deg/s"
                elif at >= milestones["walk_for"]:
                    phase = "WALK BACKWARD / 0.2 m/s requested for 1 s"
                elif at >= milestones["reset_standing"]:
                    phase = "RESET STANDING / measured dwell check"
                elif state["band_enabled"]:
                    phase = "INITIALIZATION / virtual support band ON"
                else:
                    phase = "SUPPORT RELEASED / settling"
                frame = np.full((800, 1280, 3), (17, 23, 33), dtype=np.uint8)
                text(frame, "G1 SONIC trial | replay of saved poses (~10 Hz) | no new simulation", 16, 26, scale=0.62)
                text(frame, f"{phase}   |   full-run time {i/fps:05.1f} s", 16, 56, scale=0.62)
                for j, camera in enumerate(cameras):
                    renderer.update_scene(data, camera=camera)
                    frame[80:560, j * 430:j * 430 + 420] = renderer.render()
                    text(frame, ["FRONT OBLIQUE", "SIDE", "OVERHEAD"][j], j * 430 + 12, 104)
                origin = (100, 678)
                cv2.circle(frame, origin, 67, (70, 80, 95), 1, cv2.LINE_AA)
                cv2.line(frame, (33, 678), (167, 678), (50, 60, 75), 1)
                text(frame, "world +X = 0 deg", 20, 776, scale=0.45)
                text(frame, "-yaw = clockwise", 20, 794, scale=0.4)
                for j, (angle, color, label) in enumerate(zip(angles, colors, labels)):
                    if angle is not None:
                        length = 67 - j * 8
                        end = (round(origin[0] + length * math.cos(math.radians(angle))),
                               round(origin[1] - length * math.sin(math.radians(angle))))
                        cv2.arrowedLine(frame, origin, end, color, 2, cv2.LINE_AA, tipLength=0.2)
                    value = f"{angle:+.2f} deg" if angle is not None else "unavailable"
                    text(frame, f"{label}: {value}", 190, 619 + j * 30, color, scale=0.5)
                text(frame, f"Pelvis height: {data.qpos[2]:.3f} m | support band: {'ON' if state['band_enabled'] else 'OFF'}", 685, 619)
                text(frame, f"Source pose row: {index} | held pose age: {at - state['at']:.3f} s", 685, 649)
                text(frame, "Fixed cameras | no pose interpolation | 1x wall time", 685, 679)
                text(frame, "Red floor sphere: fixed model marker (not a bottle)", 685, 709, scale=0.48)
                text(frame, "Turn test only; this clip does not evaluate bottle manipulation.", 190, 755, scale=0.53)
                text(frame, "Shutdown tail starts after the last recorded control sample; exact stop time was not logged.", 190, 783, scale=0.48)
                encoder.stdin.write(frame.tobytes())
                if i in previews:
                    cv2.imwrite(str(args.output / f"frame-{i:04d}.jpg"), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                frame_map.append({"video_frame": i, "full_video_time_s": i / fps, "pose_row": index,
                    "host_at": float(at), "pose_host_at": state["at"], "sim_time": state["sim_time"],
                    "phase": phase, "torso_yaw_deg": body, "facing_deg": angles[1], "target_deg": angles[2],
                    "goal_deg": current_goal, "pelvis_height_m": float(data.qpos[2])})
                if i % 50 == 0:
                    print(f"Rendered {i}/{len(frame_times)} frames", flush=True)
    finally:
        encoder.stdin.close()
        returncode = encoder.wait(timeout=30)
        encoder_log.close()
    if returncode:
        raise RuntimeError(f"ffmpeg failed: {returncode}")
    with (args.output / "frame-map.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(frame_map[0]))
        writer.writeheader()
        writer.writerows(frame_map)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-ss", str(focus_frame / fps),
        "-i", str(video), "-an", "-c:v", "libx264", "-crf", "19", "-pix_fmt", "yuv420p",
        "-movflags", "+faststart", str(args.output / "turn-focus.mp4")], check=True)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(args.output / "turn-focus.mp4"),
        "-vf", "setpts=2*PTS,drawbox=x=680:y=658:w=595:h=33:color=0x111721:t=fill,"
        "drawtext=fontfile=/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf:"
        "text='HALF SPEED - labels follow original time':x=685:y=669:fontsize=16:fontcolor=white",
        "-r", str(fps), "-an", "-c:v", "libx264", "-crf", "19",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(args.output / "turn-focus-half-speed.mp4")], check=True)
    source_hashes = {name: hashlib.sha256((args.trial / name).read_bytes()).hexdigest() for name in
                     ("ground-truth.jsonl", "events.jsonl", "native-trace.jsonl", "planner-trace.jsonl")}
    metadata = {"kind": "offline_saved_pose_replay", "robot_connection": False, "mj_step_calls": 0,
        "mujoco_version": mujoco.__version__, "backend": os.environ["MUJOCO_GL"], "scene": str(args.scene.resolve()),
        "scene_sha256": hashlib.sha256(args.scene.read_bytes()).hexdigest(), "trial": str(args.trial.resolve()),
        "source_sha256": source_hashes, "pose_rows": len(states), "rendered_frames": len(frame_times),
        "fps": fps, "sample_selection": "latest recorded state at or before video timestamp; zero-order hold",
        "focus_start_full_time_s": focus_frame / fps, "milestones_full_time_s": {k: v - first for k, v in milestones.items()},
        "milestones_focus_time_s": {k: v - first - focus_frame / fps for k, v in milestones.items()},
        "absolute_goal_yaw_deg": goal, "joint_mapping": {k: v.tolist() for k, v in mappings.items()},
        "shutdown_label": "after last recorded control tick; exact shutdown timestamp unavailable",
        "limitations": ["10 Hz pose sampling cannot resolve every 50 Hz foot/control event",
                        "rendered replay, not an original screen recording", "no recorded contact-force visualization",
                        "current scene assets; no as-run scene hash was saved", "single trial; no causal attribution from video alone"]}
    (args.output / "replay-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "frames": len(frame_times), "focus_start": focus_frame / fps}), flush=True)


if __name__ == "__main__":
    main()
