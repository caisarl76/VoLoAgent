"""Read D435i RGB profile calibration without starting a camera pipeline.

Run on the camera host with pyrealsense2 installed and the device attached.
This exports factory intrinsics, not a measured camera-to-robot transform.
"""

import argparse
import json
from pathlib import Path


def inspect(args):
    base = {"pipeline_started": False, "robot_commands_sent": False,
            "camera_to_robot_calibrated": False,
            "requested_profile": {"width": args.width, "height": args.height,
                                  "fps": args.fps, "format": "rgb8"}}
    try:
        import pyrealsense2 as rs
    except ImportError:
        return {**base, "status": "blocked", "reason": "pyrealsense2 is not installed"}
    try:
        context = rs.context()
        devices = list(context.query_devices())
        devices = [d for d in devices if "D435I" in d.get_info(rs.camera_info.name).upper()
                   and (args.serial is None or d.get_info(rs.camera_info.serial_number) == args.serial)]
        if len(devices) != 1:
            return {**base, "status": "blocked",
                    "reason": f"Found {len(devices)} matching D435i devices; select one with --serial if needed."}
        device = devices[0]
        profiles = []
        for sensor in device.query_sensors():
            for profile in sensor.get_stream_profiles():
                if profile.stream_type() != rs.stream.color or profile.format() != rs.format.rgb8:
                    continue
                video = profile.as_video_stream_profile()
                if (video.width(), video.height(), profile.fps()) != (args.width, args.height, args.fps):
                    continue
                intrinsic = video.get_intrinsics()
                profiles.append({"width": intrinsic.width, "height": intrinsic.height,
                                 "fps": profile.fps(), "format": "rgb8",
                                 "fx": intrinsic.fx, "fy": intrinsic.fy,
                                 "cx": intrinsic.ppx, "cy": intrinsic.ppy,
                                 "distortion_model": str(intrinsic.model),
                                 "distortion_coefficients": list(intrinsic.coeffs)})
        return {**base, "status": "available" if profiles else "blocked",
                "reason": None if profiles else "Requested color profile is not advertised.",
                "device_name": device.get_info(rs.camera_info.name),
                "firmware_version": device.get_info(rs.camera_info.firmware_version),
                "profile_intrinsics": profiles,
                "limitations": ["No RGB optical-to-robot transform is measured.",
                                "Select the actual recording/runtime resolution and frame rate.",
                                "No pipeline was opened and no image was captured."]}
    except RuntimeError as error:
        return {**base, "status": "blocked", "reason": str(error)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--serial")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=15)
    args = parser.parse_args()
    if min(args.width, args.height, args.fps) <= 0:
        parser.error("width, height and fps must be positive")
    if args.output.exists():
        parser.error("output already exists; choose a new path")
    result = inspect(args)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0 if result["status"] == "available" else 2



from types import SimpleNamespace
print(json.dumps(inspect(SimpleNamespace(width=640,height=480,fps=15,serial=None)),indent=2))
