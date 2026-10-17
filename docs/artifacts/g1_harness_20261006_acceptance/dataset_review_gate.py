"""Assess task-reviewed future dataset eligibility without changing data."""

import argparse
import json
from pathlib import Path


PROMPT = "pick drink bottle and place it on the right table"


def assess(plan):
    errors, windows, seen = [], [], set()
    if not isinstance(plan, dict):
        return {"ready": False, "errors": ["plan must be an object"], "training_windows": []}
    if plan.get("prompt") != PROMPT or plan.get("action_horizon") != 40:
        errors.append("plan must use the registered bottle prompt and 40-frame horizon")
    episodes = plan.get("episodes", [])
    if not isinstance(episodes, list):
        return {"ready": False, "errors": ["episodes must be a list"], "training_windows": []}
    for episode in episodes:
        if not isinstance(episode, dict):
            errors.append("episode must be an object")
            continue
        source = episode.get("source_episode")
        if type(source) is not int or source < 0:
            errors.append("invalid source episode")
            continue
        if source in seen:
            errors.append(f"duplicate source episode {source}; splits must be by whole source episode")
        seen.add(source)
        split = episode.get("split")
        if not isinstance(split, str) or split not in {"train", "held_out", "quarantine", "review_needed"}:
            errors.append(f"episode {source}: invalid split")
        digest = episode.get("source_sha256", "")
        if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            errors.append(f"episode {source}: missing source checksum")
        segments = episode.get("segments", [])
        if not isinstance(segments, list):
            errors.append(f"episode {source}: segments must be a list")
            continue
        if split != "train":
            if segments:
                errors.append(f"episode {source}: non-training episode has training segments")
            continue
        frames = episode.get("frames")
        if type(frames) is not int or frames < 40:
            errors.append(f"episode {source}: invalid frame count")
            continue
        if not segments:
            errors.append(f"episode {source}: training episode has no reviewed segments")
        ranges = []
        for segment in segments:
            if not isinstance(segment, dict):
                errors.append(f"episode {source}: segment must be an object")
                continue
            valid_review = (segment.get("review_state") == "accepted"
                            and segment.get("task_prompt") == PROMPT
                            and segment.get("outcome") == "success"
                            and isinstance(segment.get("evidence"), list)
                            and bool(segment["evidence"])
                            and all(isinstance(p, str) and p for p in segment["evidence"]))
            a, b = segment.get("start_frame"), segment.get("end_frame_exclusive")
            valid_bounds = (type(a) is int and type(b) is int and 0 <= a < b <= frames and b - a >= 40)
            if not valid_review or not valid_bounds:
                errors.append(f"episode {source}: segment lacks accepted task/success evidence or valid bounds")
                continue
            if any(a < old_b and old_a < b for old_a, old_b in ranges):
                errors.append(f"episode {source}: overlapping reviewed segments")
            ranges.append((a, b))
            windows.append({"source_episode": source, "first_start_frame": a,
                            "last_start_frame": b - 40, "count": b - a - 39})
    if not windows:
        errors.append("no accepted training windows; sampled screening cannot enable training")
    if not any(isinstance(e, dict) and e.get("split") == "held_out" for e in episodes):
        errors.append("no whole-episode held-out split")
    return {"ready": not errors, "errors": errors,
            "training_windows": windows if not errors else [],
            "scope": "Review-metadata eligibility only; does not export data, validate label truth or authorize training."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path)
    args = parser.parse_args()
    result = assess(json.loads(args.plan.read_text()))
    print(json.dumps(result, indent=2))
    return 0 if result["ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
