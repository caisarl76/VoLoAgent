"""Agent-facing CLI; hardware startup remains in the native deployment workflow."""

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import time

from openai import OpenAI

from vlm_orchestrator.vlm.api import chat_create

from .client import G1Client
from .contract import SkillCall
from .monitor import G1CompletionMonitor, decode_image
from .registry import load_profile, validate_skill_call
from .runner import HarnessRunner


def main(argv=None):
    parser = argparse.ArgumentParser(description="Unitree G1 skill harness")
    parser.add_argument(
        "--profile", type=Path, default=Path("configs/g1/workstation.yaml")
    )
    parser.add_argument("--endpoint", default="ipc:///tmp/volo-g1-harness.sock")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("skills")
    sub.add_parser("status")
    observe = sub.add_parser("observe")
    observe.add_argument("--output", required=True, type=Path)
    run = sub.add_parser("run")
    run.add_argument("--skill", required=True)
    execute = sub.add_parser("execute")
    execute.add_argument("--plan-file", required=True, type=Path)
    for p in (run, execute):
        p.add_argument("--vlm-model")
        p.add_argument("--vlm-base-url")
        p.add_argument("--vlm-api-key")
        p.add_argument("--evidence-dir", type=Path, default=Path("results/g1-harness"))
        p.add_argument("--locomotion", action="store_true")
    args = parser.parse_args(argv)
    try:
        profile = load_profile(args.profile)
        if args.command == "skills":
            print(
                json.dumps(
                    dict(
                        manipulation=[asdict(s) for s in profile.skills.values()],
                        procedural=[
                            "observe",
                            "pause_manipulation",
                            "cancel",
                            "reset_standing",
                        ],
                        locomotion_requires_opt_in=["walk_for", "turn_by"],
                        checkpoint_expected=profile.checkpoint,
                    )
                )
            )
            return 0

        def factory():
            return G1Client(args.endpoint, profile.limits.rpc_timeout_s)

        if args.command in {"status", "observe"}:
            client = factory()
            try:
                result = (
                    client.get_status()
                    if args.command == "status"
                    else client.observe()
                )
                if args.command == "observe":
                    from PIL import Image

                    Image.fromarray(decode_image(result)).save(args.output)
                    print(
                        json.dumps(
                            {
                                "frame_id": result.frame_id,
                                "output": str(args.output),
                                "age_s": result.age_s,
                            }
                        )
                    )
                else:
                    print(json.dumps(asdict(result)))
            finally:
                client.close()
            return 0
        if args.command == "run":
            calls = [SkillCall(args.skill, {})]
        else:
            data = json.loads(args.plan_file.read_text())
            if (
                not isinstance(data, dict)
                or data.keys() != {"version", "skills"}
                or type(data["version"]) is not int
                or data["version"] != 1
                or not isinstance(data["skills"], list)
            ):
                raise ValueError("Invalid sequence schema")
            calls = []
            for entry in data["skills"]:
                if not isinstance(entry, dict) or entry.keys() != {
                    "skill_id",
                    "params",
                }:
                    raise ValueError("Invalid sequence skill")
                calls.append(SkillCall(**entry))
        for call in calls:
            validate_skill_call(profile, call, locomotion_enabled=args.locomotion)
        manipulation = [c for c in calls if c.skill_id in profile.skills]
        monitor = None
        if manipulation:
            if (
                not args.vlm_model
                or not args.vlm_base_url
                or "YOUR_" in args.vlm_model
                or "YOUR_" in args.vlm_base_url
            ):
                raise ValueError("Manipulation requires --vlm-model and --vlm-base-url")
            api = OpenAI(
                api_key=args.vlm_api_key
                or os.environ.get("VLM_API_KEY")
                or os.environ.get("OPENAI_API_KEY")
                or "local",
                base_url=args.vlm_base_url,
                timeout=profile.limits.monitor_deadline_s,
                max_retries=0,
            )

            def vlm_call(system, content):
                response = chat_create(
                    api,
                    model=args.vlm_model,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": content},
                    ],
                    temperature=0.0,
                )
                return response.choices[0].message.content

            monitor = G1CompletionMonitor(
                profile.require_skill(manipulation[0].skill_id),
                vlm_call,
                time.monotonic,
                profile.limits,
            )
        runner = HarnessRunner(profile, factory, monitor, args.evidence_dir)
        runner.locomotion_enabled = args.locomotion
        result = runner.run_sequence(calls)
        print(json.dumps(asdict(result), default=str))
        return 0 if result.outcome == "completed" else 1
    except (ValueError, OSError, RuntimeError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())
