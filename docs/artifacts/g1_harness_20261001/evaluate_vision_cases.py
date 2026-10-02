"""Inspect recorded frames with the real G1 monitor, without any robot client.

Cases are independent unless they share an explicit sequence_id. Replay
sequences check two-frame confirmation on recorded images, without robot control.
"""

import argparse
import base64
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import time

from openai import OpenAI

from vlm_orchestrator.harness.g1.contract import Execution, ObservationSnapshot
from vlm_orchestrator.harness.g1.monitor import G1CompletionMonitor
from vlm_orchestrator.harness.g1.registry import SkillDefinition
from vlm_orchestrator.vlm.api import chat_create, parse_json


def frame(video, at_s, output):
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-ss",
            str(at_s),
            "-i",
            str(video),
            "-frames:v",
            "1",
            "-y",
            str(output),
        ],
        check=True,
    )
    if not output.is_file():
        raise ValueError(f"No frame at {at_s}s in {video}")
    return output.read_bytes()


def snapshot(jpeg, execution, at_s):
    return ObservationSnapshot(
        execution.runtime_id,
        execution.execution_id,
        execution.inference_epoch,
        "ego_view",
        at_s,
        hashlib.sha256(jpeg).hexdigest(),
        time.monotonic(),
        0.0,
        base64.b64encode(jpeg).decode(),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", type=Path, default=Path(__file__).with_name("vision_cases.json")
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--vlm-model")
    parser.add_argument("--vlm-base-url")
    parser.add_argument("--env-file", type=Path)
    parser.add_argument("--case", action="append", help="Evaluate only these case IDs")
    args = parser.parse_args()
    if not args.prepare_only and not (args.vlm_model and args.vlm_base_url):
        parser.error("Evaluation needs --vlm-model and --vlm-base-url")
    manifest = json.loads(args.manifest.read_text())
    if args.case:
        known = {c["id"] for c in manifest["cases"]}
        if set(args.case) - known:
            parser.error("Unknown case ID")
        manifest["cases"] = [c for c in manifest["cases"] if c["id"] in args.case]
    api_key = (
        os.environ.get("GENON_API_KEY")
        or os.environ.get("VLM_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
    )
    if args.env_file:
        for line in args.env_file.read_text().splitlines():
            name, separator, value = line.strip().removeprefix("export ").partition("=")
            if separator and name.strip() == "GENON_API_KEY":
                tokens = shlex.split(value, comments=True)
                if len(tokens) == 1:
                    api_key = tokens[0]
    if not args.prepare_only and not api_key:
        parser.error("Set GENON_API_KEY or provide --env-file containing it")
    args.output.mkdir(parents=True, exist_ok=True)
    frames = args.output / "frames"
    frames.mkdir(exist_ok=True)
    report = {
        "robot_connection": False,
        "scope": "Recorded frame judgments and explicit replay sequences; no live robot mission",
        "label_provenance": manifest["label_provenance"],
        "model": args.vlm_model,
        "base_url": args.vlm_base_url,
        "episodes": {},
        "cases": [],
    }
    for name, episode in manifest["episodes"].items():
        video = Path(episode["video"])
        report["episodes"][name] = {
            "video": str(video),
            "sha256": hashlib.sha256(video.read_bytes()).hexdigest(),
            "user_outcome": episode["outcome"],
        }
    api = (
        None
        if args.prepare_only
        else OpenAI(
            api_key=api_key,
            base_url=args.vlm_base_url,
            timeout=10.0,
            max_retries=0,
        )
    )

    def vlm_call(system, content):
        return (
            chat_create(
                api,
                model=args.vlm_model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": content},
                ],
                temperature=0.0,
            )
            .choices[0]
            .message.content
        )

    monitors = {}
    for case in manifest["cases"]:
        episode = manifest["episodes"][case["episode"]]
        jpeg = frame(episode["video"], case["at_s"], frames / f"{case['id']}.jpg")
        initial_s = case.get("initial_s", episode["initial_s"])
        initial_path = frames / f"{case['episode']}_initial_{initial_s:g}.jpg"
        initial = (
            initial_path.read_bytes()
            if initial_path.exists()
            else frame(episode["video"], initial_s, initial_path)
        )
        record = {**case, "image": f"frames/{case['id']}.jpg"}
        if api is not None:
            skill = SkillDefinition(
                case["episode"], episode["prompt"], episode["completion_criteria"]
            )
            key = (case["episode"], case.get("sequence_id", case["id"]))
            if key not in monitors:
                execution = Execution(
                    "offline-replay", key[1], skill.skill_id,
                    "MANIPULATING", 1, time.monotonic(), None,
                )
                monitor = G1CompletionMonitor(skill, vlm_call, time.monotonic)
                monitor.begin(execution, snapshot(initial, execution, initial_s))
                monitors[key] = (execution, monitor, initial_s)
            execution, monitor, last_s = monitors[key]
            if case.get("sequence_id") and case["at_s"] <= last_s:
                raise ValueError("Replay sequence frames must follow the initial image and increase in time")
            monitors[key] = (execution, monitor, case["at_s"])
            decision = monitor.check(snapshot(jpeg, execution, case["at_s"]))
            raw = monitor.raw_response
            parsed = parse_json(raw) if raw is not None else None
            claim = (
                None
                if decision.outcome == "unavailable" or not isinstance(parsed, dict)
                else (
                    parsed.get("status") == "complete"
                    and parsed.get("action") == "next"
                )
            )
            record.update(
                decision=asdict(decision),
                raw_response=raw,
                frame_complete=claim,
                matches_label=claim == case["complete"] if claim is not None else None,
                decision_latency_s=decision.decided_at - decision.captured_at,
                confirmation_streak=monitor.streak,
                matches_monitor_outcome=(
                    decision.outcome == case["expected_monitor_outcome"]
                    if "expected_monitor_outcome" in case else None
                ),
            )
        report["cases"].append(record)
        (args.output / "result.json").write_text(json.dumps(report, indent=2) + "\n")
        print(
            json.dumps(
                {
                    "case": case["id"],
                    "prepared": True,
                    "matches_label": record.get("matches_label"),
                }
            ),
            flush=True,
        )
    if not args.prepare_only:
        cases = report["cases"]
        report["summary"] = {
            "total": len(cases),
            "unavailable": sum(c["frame_complete"] is None for c in cases),
            "valid_negative_cases": sum(
                not c["complete"] and c["frame_complete"] is not None for c in cases
            ),
            "false_complete": sum(
                not c["complete"] and c["frame_complete"] is True for c in cases
            ),
            "positive_label_disagreements": sum(
                c["complete"] and c["frame_complete"] is False for c in cases
            ),
            "sequence_confirmations": sum(
                bool(c.get("sequence_id")) and c["decision"]["outcome"] == "complete"
                for c in cases
            ),
            "monitor_outcome_disagreements": sum(c["matches_monitor_outcome"] is False for c in cases),
        }
        (args.output / "result.json").write_text(json.dumps(report, indent=2) + "\n")
        return (
            1
            if report["summary"]["unavailable"]
            or any(c["matches_label"] is False for c in cases)
            or report["summary"]["monitor_outcome_disagreements"]
            else 0
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
