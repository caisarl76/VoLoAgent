"""Validate reviewed subskill metadata and prepare contained action windows.

This creates a window manifest, not a GR00T dataset. Review assertions and
source digests need independent verification before dataset export or training.
"""

import argparse
import json
from pathlib import Path


SCOPE = "Reviewed metadata only; no label-truth validation, dataset export, training or policy activation."


def rejection(errors):
    return {
        "ready": False,
        "errors": errors,
        "training_windows": [],
        "held_out_windows": [],
        "scope": SCOPE,
    }


def text(value):
    return isinstance(value, str) and bool(value.strip())


def evidence(value):
    return isinstance(value, list) and bool(value) and all(text(item) for item in value)


def assess(plan, catalogue):
    errors = []
    if not isinstance(plan, dict) or not isinstance(catalogue, dict):
        return rejection(["plan and catalogue must be objects"])
    for name, data in [("plan", plan), ("catalogue", catalogue)]:
        for key, expected in [
            ("schema_version", 1),
            ("fps", 50),
            ("action_horizon", 40),
        ]:
            if type(data.get(key)) is not int or data[key] != expected:
                errors.append(f"{name}: {key} must be {expected}")
    skills = catalogue.get("skills")
    if not isinstance(skills, dict) or not skills:
        return rejection(errors + ["catalogue skills must be a nonempty object"])
    for skill_id, skill in skills.items():
        if (
            not text(skill_id)
            or not isinstance(skill, dict)
            or not text(skill.get("prompt"))
        ):
            errors.append("invalid catalogue skill or prompt")
            continue
        minimum = skill.get("minimum_terminal_frames")
        if type(minimum) is not int or minimum < 100:
            errors.append(f"{skill_id}: at least 100 stable terminal frames required")
    if errors:
        return rejection(errors)
    episodes = plan.get("episodes")
    if not isinstance(episodes, list):
        return rejection(["episodes must be a list"])
    seen_sources, seen_prepared, seen_digests = set(), set(), set()
    windows = {"train": [], "held_out": []}
    covered = {"train": set(), "held_out": set()}
    for episode in episodes:
        if not isinstance(episode, dict):
            errors.append("episode must be an object")
            continue
        source, prepared = (
            episode.get("source_episode"),
            episode.get("prepared_episode"),
        )
        if (
            type(source) is not int
            or source < 0
            or type(prepared) is not int
            or prepared < 0
        ):
            errors.append("invalid source/prepared episode identity")
            continue
        if source in seen_sources or prepared in seen_prepared:
            errors.append(
                f"source {source}: duplicate identity; split by whole source episode"
            )
        seen_sources.add(source)
        seen_prepared.add(prepared)
        digest = episode.get("source_sha256")
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(c not in "0123456789abcdef" for c in digest)
        ):
            errors.append(f"source {source}: missing source SHA256")
        else:
            if digest in seen_digests:
                errors.append(f"source {source}: duplicate source content")
            seen_digests.add(digest)
        split, frames = episode.get("split"), episode.get("frames")
        if not isinstance(split, str) or split not in {
            "train",
            "held_out",
            "quarantine",
            "review_needed",
        }:
            errors.append(f"source {source}: invalid split")
            continue
        if type(frames) is not int or frames <= 0:
            errors.append(f"source {source}: invalid frame count")
            continue
        segments = episode.get("segments", [])
        if not isinstance(segments, list):
            errors.append(f"source {source}: segments must be a list")
            continue
        if split not in windows:
            if segments:
                errors.append(
                    f"source {source}: pending/quarantined source has accepted segments"
                )
            continue
        if split == "train" and not segments:
            errors.append(f"source {source}: training source lacks reviewed segments")
        ranges = []
        for segment in segments:
            if not isinstance(segment, dict):
                errors.append(f"source {source}: segment must be an object")
                continue
            skill_id = segment.get("skill_id")
            if not isinstance(skill_id, str) or skill_id not in skills:
                errors.append(f"source {source}: unknown manipulation skill")
                continue
            skill = skills[skill_id]
            a, b = segment.get("start_frame"), segment.get("end_frame_exclusive")
            if (
                type(a) is not int
                or type(b) is not int
                or not 0 <= a < b <= frames
                or b - a < 40
            ):
                errors.append(f"source {source}: invalid 40-frame segment bounds")
                continue
            terminal = segment.get("terminal_stable_frames")
            valid = (
                segment.get("review_state") == "accepted"
                and segment.get("outcome") == "success"
                and segment.get("task_prompt") == skill["prompt"]
                and text(segment.get("reviewer"))
                and evidence(segment.get("entry_evidence"))
                and evidence(segment.get("terminal_evidence"))
                and type(terminal) is int
                and skill["minimum_terminal_frames"] <= terminal <= b - a
            )
            if not valid:
                errors.append(
                    f"source {source}/{skill_id}: accepted review, prompt, success and stable entry/terminal evidence required"
                )
                continue
            if any(a < old_b and old_a < b for old_a, old_b in ranges):
                errors.append(f"source {source}: overlapping skill segments")
            ranges.append((a, b))
            covered[split].add(skill_id)
            windows[split].append(
                {
                    "source_episode": source,
                    "prepared_episode": prepared,
                    "source_sha256": digest,
                    "skill_id": skill_id,
                    "task_prompt": skill["prompt"],
                    "start_frame": a,
                    "end_frame_exclusive": b,
                    "first_start_frame": a,
                    "last_start_frame": b - 40,
                    "count": b - a - 39,
                }
            )
    for split in covered:
        for skill_id in sorted(set(skills) - covered[split]):
            errors.append(f"{split}: no accepted windows for {skill_id}")
    if errors:
        return rejection(errors)
    return {
        "ready": True,
        "errors": [],
        "training_windows": windows["train"],
        "held_out_windows": windows["held_out"],
        "scope": SCOPE,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path)
    parser.add_argument("--catalogue", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        help="Write a window manifest only after all metadata checks pass",
    )
    args = parser.parse_args()
    try:
        result = assess(
            json.loads(args.plan.read_text()), json.loads(args.catalogue.read_text())
        )
    except (OSError, ValueError) as exc:
        result = rejection([f"Cannot read review metadata: {exc}"])
    payload = json.dumps(result, indent=2) + "\n"
    print(payload, end="")
    if result["ready"] and args.output:
        args.output.write_text(payload)
    return 0 if result["ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
