import os
import subprocess
from pathlib import Path

import pytest
import yaml

from vlm_orchestrator.harness.g1.contract import validate_params
from vlm_orchestrator.harness.g1.registry import load_profile

ROOT = Path(__file__).resolve().parents[1]
NAMES = [
    f"right_{name}_joint"
    for name in (
        "shoulder_pitch",
        "shoulder_roll",
        "shoulder_yaw",
        "elbow",
        "wrist_roll",
        "wrist_pitch",
        "wrist_yaw",
    )
]
TARGET = [
    -0.1041309312,
    -0.0850760117,
    0.3027574718,
    -0.2182087749,
    1.7013045549,
    -0.0961015001,
    -0.0728760734,
]
RAISED = [.0111932661, -.6817826033, -.5984442830, .1641239673,
          1.0858786106, -.1572330296, -.4897353649]


def handover_profile(tmp_path, *, verified=True, reviewed=True):
    data = yaml.safe_load((ROOT / "configs/g1/workstation.yaml").read_text())
    data["checkpoint"] = "/test-only/handover-checkpoint"
    data["limits"]["monitor_interval_s"] = 0.01
    data["skills"] = {
        "bottle_handover": {
            "prompt": "pick up the drink bottle and offer it on your open palm to the person",
            "completion_criteria": "Bottle stably supported on level open palm.",
            "completion_action": "hold",
        }
    }
    data["handover"] = {
        "skill_id": "bottle_handover",
        "checkpoint_verified": verified,
        "ready_pose_reviewed": reviewed,
        "ready_right_arm_joints": dict(zip(reversed(NAMES), reversed(TARGET))),
        "raised_right_arm_joints": dict(zip(reversed(NAMES), reversed(RAISED))),
    }
    path = tmp_path / "handover.yaml"
    path.write_text(yaml.safe_dump(data))
    return path


def test_handover_target_uses_named_motor_order(tmp_path):
    profile = load_profile(handover_profile(tmp_path))
    assert profile.handover.right_arm_target == tuple(TARGET)
    assert profile.handover.right_arm_waypoint == tuple(RAISED)
    validate_params("reset_ready", {"execution_id": ""})
    with pytest.raises(ValueError):
        validate_params("reset_ready", {"execution_id": "x", "joints": TARGET})


@pytest.mark.parametrize("field", ["ready_right_arm_joints", "raised_right_arm_joints"])
@pytest.mark.parametrize(
    "value",
    [
        [],
        {},
        [1] * 7,
        {NAMES[0]: float("nan")},
        {**dict(zip(NAMES, TARGET)), NAMES[4]: 2.1},
    ],
)
def test_handover_rejects_malformed_or_out_of_bounds_targets(tmp_path, field, value):
    path = handover_profile(tmp_path)
    data = yaml.safe_load(path.read_text())
    data["handover"][field] = value
    path.write_text(yaml.safe_dump(data))
    with pytest.raises(ValueError):
        load_profile(path)


def test_handover_rejects_missing_raise_target(tmp_path):
    path = handover_profile(tmp_path)
    data = yaml.safe_load(path.read_text())
    del data["handover"]["raised_right_arm_joints"]
    path.write_text(yaml.safe_dump(data))
    with pytest.raises(ValueError):
        load_profile(path)


def test_example_is_disabled_and_separate_from_placement():
    example = load_profile(ROOT / "configs/g1/handover.example.yaml")
    assert not example.handover.checkpoint_verified
    assert not example.handover.ready_pose_reviewed
    assert "pnp_bottle" not in example.checkpoint


def test_handover_desk_pose_is_optional_and_paired(tmp_path):
    assert load_profile(handover_profile(tmp_path)).handover.desk_top_above_floor_m is None
    example = load_profile(ROOT / "configs/g1/handover.example.yaml").handover
    assert (example.desk_top_above_floor_m, example.desk_front_from_pelvis_m) == (0.845, 0.15)


@pytest.mark.parametrize(
    "desk",
    [
        {"desk_top_above_floor_m": 0.85},
        {"desk_front_from_pelvis_m": 0.40},
        {"desk_top_above_floor_m": float("nan"), "desk_front_from_pelvis_m": 0.40},
        {"desk_top_above_floor_m": True, "desk_front_from_pelvis_m": 0.40},
        {"desk_top_above_floor_m": 0.85, "desk_front_from_pelvis_m": -0.1},
        {"desk_top_above_floor_m": 0.063, "desk_front_from_pelvis_m": 0.40},
    ],
)
def test_handover_rejects_partial_or_invalid_desk_pose(tmp_path, desk):
    path = handover_profile(tmp_path)
    data = yaml.safe_load(path.read_text())
    data["handover"].update(desk)
    path.write_text(yaml.safe_dump(data))
    with pytest.raises(ValueError):
        load_profile(path)


@pytest.mark.parametrize("value", [True, False, 1, "true", None])
def test_handover_gravity_feedforward_is_optional_bool(tmp_path, value):
    assert not load_profile(handover_profile(tmp_path)).handover.ready_gravity_feedforward
    path = handover_profile(tmp_path)
    data = yaml.safe_load(path.read_text())
    data["handover"]["ready_gravity_feedforward"] = value
    path.write_text(yaml.safe_dump(data))
    if type(value) is bool:
        assert load_profile(path).handover.ready_gravity_feedforward is value
    else:
        with pytest.raises(ValueError):
            load_profile(path)


def test_example_ready_pose_is_the_recorded_palm_in_pose_for_both_arms():
    example = load_profile(ROOT / "configs/g1/handover.example.yaml").handover
    assert example.right_arm_target == (-0.2287, -0.3292, 0.0495, -0.3238, 0.4396, 0.1176, -0.0919)
    assert example.left_arm_target == (0.1367, 0.309, -0.4384, 0.7003)


@pytest.mark.parametrize("bad", [
    {name: 0.0 for name in NAMES[:4]},  # Right-arm names.
    {name.replace("right_", "left_"): 9.0 for name in NAMES[:4]},  # Outside bounds.
    {name.replace("right_", "left_"): 0.0 for name in NAMES[:3]},
    {name.replace("right_", "left_"): 0.0 for name in NAMES},  # Wrists cannot be set.
])
def test_handover_left_arm_ready_pose_is_optional_and_validated(tmp_path, bad):
    assert load_profile(handover_profile(tmp_path)).handover.left_arm_target is None
    path = handover_profile(tmp_path)
    data = yaml.safe_load(path.read_text())
    data["handover"]["ready_left_arm_joints"] = bad
    path.write_text(yaml.safe_dump(data))
    with pytest.raises(ValueError):
        load_profile(path)


LOOP_PROFILE = ROOT / "configs/g1/handover-loop.example.yaml"


def test_loop_example_profile_enables_anchoring():
    profile = load_profile(LOOP_PROFILE)
    assert profile.lerobot_policy.chunk_anchor_s == 1.0
    assert profile.handover.checkpoint_verified is False


def test_native_parser_accepts_the_loop_example_profile(native_runtime):
    checkout, python = native_runtime
    code = "import sys; from gear_sonic.utils.inference.harness_profile import load_profile as l; print(l(sys.argv[1]).lerobot_policy.chunk_anchor_s)"
    out = subprocess.run(
        [str(python), "-c", code, str(LOOP_PROFILE)],
        cwd=checkout, env={**os.environ, "PYTHONPATH": str(checkout)}, capture_output=True, text=True, check=True,
    )
    assert out.stdout.strip() == "1.0"
