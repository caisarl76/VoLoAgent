"""Training preparation rejects pending labels and source/split contamination."""

import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest


ARTIFACT = Path(__file__).resolve().parents[1] / "docs/artifacts/g1_subskills_20261007"


def gate():
    path = ARTIFACT / "subskill_dataset.py"
    assert path.is_file(), "Subskill review gate is missing"
    spec = importlib.util.spec_from_file_location("subskill_review_gate", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fixture():
    skills = {
        "pick_bottle_and_hold": {
            "prompt": "pick the bottle and hold it",
            "minimum_terminal_frames": 100,
        },
        "place_held_bottle_on_stool": {
            "prompt": "place the held bottle on the green stool",
            "minimum_terminal_frames": 100,
        },
    }
    catalogue = {"schema_version": 1, "fps": 50, "action_horizon": 40, "skills": skills}
    episodes = []
    for source, split in [(52, "train"), (18, "held_out")]:
        segments = []
        for index, (skill, spec) in enumerate(skills.items()):
            segments.append(
                {
                    "skill_id": skill,
                    "task_prompt": spec["prompt"],
                    "start_frame": 10 + index * 150,
                    "end_frame_exclusive": 160 + index * 150,
                    "review_state": "accepted",
                    "outcome": "success",
                    "reviewer": "fixture-reviewer",
                    "entry_evidence": ["fixture entry frame"],
                    "terminal_evidence": ["fixture terminal interval"],
                    "terminal_stable_frames": 100,
                }
            )
        episodes.append(
            {
                "source_episode": source,
                "prepared_episode": source - 1,
                "split": split,
                "frames": 500,
                "source_sha256": f"{source:064x}",
                "segments": segments,
            }
        )
    return {
        "schema_version": 1,
        "fps": 50,
        "action_horizon": 40,
        "episodes": episodes,
    }, catalogue


def test_valid_review_enumerates_only_contained_windows():
    plan, catalogue = fixture()
    result = gate().assess(plan, catalogue)
    assert result["ready"] and not result["errors"]
    assert len(result["training_windows"]) == len(result["held_out_windows"]) == 2
    first, second = result["training_windows"]
    assert (first["first_start_frame"], first["last_start_frame"], first["count"]) == (
        10,
        120,
        111,
    )
    assert second["first_start_frame"] == 160
    assert first["last_start_frame"] + 40 <= second["first_start_frame"]


def test_known_failed_skill_cannot_be_relabeled_as_success():
    plan, catalogue = fixture()
    plan["episodes"][0]["failed_skills"] = ["place_held_bottle_on_stool"]
    result = gate().assess(plan, catalogue)
    assert not result["ready"]
    assert any("failed skill" in error for error in result["errors"])
    assert result["training_windows"] == result["held_out_windows"] == []


def test_failure_on_placement_does_not_discard_an_independent_pick():
    plan, catalogue = fixture()
    episode = plan["episodes"][0]
    other = copy.deepcopy(episode)
    other.update(source_episode=90, prepared_episode=89, source_sha256=f"{90:064x}")
    other["segments"] = other["segments"][1:]
    episode["segments"] = episode["segments"][:1]
    episode["failed_skills"] = ["place_held_bottle_on_stool"]
    plan["episodes"].append(other)
    result = gate().assess(plan, catalogue)
    assert result["ready"], result["errors"]
    assert [
        (w["source_episode"], w["skill_id"]) for w in result["training_windows"]
    ] == [
        (52, "pick_bottle_and_hold"),
        (90, "place_held_bottle_on_stool"),
    ]


@pytest.mark.parametrize(
    "value", [{}, "place_held_bottle_on_stool", [None], [[]], ["invented_skill"]]
)
def test_malformed_failed_skill_metadata_is_rejected(value):
    plan, catalogue = fixture()
    plan["episodes"][0]["failed_skills"] = value
    result = gate().assess(plan, catalogue)
    assert not result["ready"] and result["errors"]


def excluded(start, end):
    return {
        "start_frame": start,
        "end_frame_exclusive": end,
        "reason": "Return to start after prior episode",
        "reviewer": "user",
    }


def test_positive_segment_cannot_touch_excluded_previous_episode_frames():
    plan, catalogue = fixture()
    plan["episodes"][0]["excluded_intervals"] = [excluded(0, 11)]
    result = gate().assess(plan, catalogue)
    assert not result["ready"]
    assert any("excluded interval" in error for error in result["errors"])


def test_positive_segment_can_start_exactly_after_excluded_interval():
    plan, catalogue = fixture()
    plan["episodes"][0]["excluded_intervals"] = [excluded(0, 10)]
    result = gate().assess(plan, catalogue)
    assert result["ready"], result["errors"]
    assert result["training_windows"][0]["first_start_frame"] == 10


@pytest.mark.parametrize(
    "value",
    [
        {},
        [None],
        [excluded(True, 10)],
        [excluded(-1, 10)],
        [excluded(10, 10)],
        [excluded(0, 501)],
        [{"start_frame": 0, "end_frame_exclusive": 10}],
    ],
)
def test_malformed_exclusion_metadata_is_rejected(value):
    plan, catalogue = fixture()
    plan["episodes"][0]["excluded_intervals"] = value
    result = gate().assess(plan, catalogue)
    assert not result["ready"] and result["errors"]


@pytest.mark.parametrize(
    "path,value",
    [
        (("split",), []),
        (("split",), {}),
        (("split",), "validation"),
        (("source_episode",), []),
        (("source_episode",), True),
        (("prepared_episode",), -1),
        (("source_sha256",), "missing"),
        (("frames",), True),
        (("segments",), {}),
        (("segments", 0, "skill_id"), []),
        (("segments", 0, "skill_id"), "approach_table"),
        (("segments", 0, "task_prompt"), "pick the red block"),
        (("segments", 0, "review_state"), "pending"),
        (("segments", 0, "outcome"), "failure"),
        (("segments", 0, "reviewer"), ""),
        (("segments", 0, "entry_evidence"), []),
        (("segments", 0, "terminal_evidence"), [None]),
        (("segments", 0, "terminal_stable_frames"), 99),
        (("segments", 0, "terminal_stable_frames"), 151),
        (("segments", 0, "start_frame"), True),
        (("segments", 0, "end_frame_exclusive"), 999),
        (("segments", 0, "end_frame_exclusive"), 49),
    ],
)
def test_bad_segment_metadata_returns_structured_rejection(path, value):
    plan, catalogue = fixture()
    node = plan["episodes"][0]
    for key in path[:-1]:
        node = node[key]
    node[path[-1]] = value
    result = gate().assess(plan, catalogue)
    assert not result["ready"] and result["errors"]
    assert result["training_windows"] == result["held_out_windows"] == []


def test_whole_source_cannot_leak_into_both_splits():
    plan, catalogue = fixture()
    other = copy.deepcopy(plan["episodes"][0])
    other["split"] = "held_out"
    plan["episodes"].append(other)
    assert not gate().assess(plan, catalogue)["ready"]


def test_prepared_episode_alias_cannot_be_assigned_twice():
    plan, catalogue = fixture()
    plan["episodes"][1]["prepared_episode"] = plan["episodes"][0]["prepared_episode"]
    assert not gate().assess(plan, catalogue)["ready"]


def test_overlapping_subskills_rejected():
    plan, catalogue = fixture()
    plan["episodes"][0]["segments"][1]["start_frame"] = 159
    assert not gate().assess(plan, catalogue)["ready"]


@pytest.mark.parametrize("split", ["train", "held_out"])
def test_each_skill_needs_reviewed_coverage_in_each_split(split):
    plan, catalogue = fixture()
    next(e for e in plan["episodes"] if e["split"] == split)["segments"].pop()
    assert not gate().assess(plan, catalogue)["ready"]


@pytest.mark.parametrize("plan", [None, [], {"episodes": {}}, {"episodes": [None]}])
def test_bad_plan_rejected(plan):
    _, catalogue = fixture()
    assert not gate().assess(plan, catalogue)["ready"]


def test_pending_artifact_cannot_generate_windows(tmp_path):
    path = ARTIFACT / "candidate-manifest.json"
    catalogue = json.loads((ARTIFACT / "candidate-skills.json").read_text())
    result = gate().assess(json.loads(path.read_text()), catalogue)
    assert not result["ready"] and result["training_windows"] == []
    output = tmp_path / "windows.json"
    run = subprocess.run(
        [
            sys.executable,
            str(ARTIFACT / "subskill_dataset.py"),
            str(path),
            "--catalogue",
            str(ARTIFACT / "candidate-skills.json"),
            "--output",
            str(output),
        ],
        capture_output=True,
        text=True,
    )
    assert run.returncode == 1 and not output.exists()
    assert not json.loads(run.stdout)["ready"]


def test_cli_malformed_json_reports_rejection(tmp_path):
    path = tmp_path / "malformed.json"
    path.write_text("{broken")
    run = subprocess.run(
        [
            sys.executable,
            str(ARTIFACT / "subskill_dataset.py"),
            str(path),
            "--catalogue",
            str(ARTIFACT / "candidate-skills.json"),
        ],
        capture_output=True,
        text=True,
    )
    assert run.returncode == 1 and not json.loads(run.stdout)["ready"]
