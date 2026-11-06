from pathlib import Path

from vlm_orchestrator.harness.g1.registry import load_profile

PROFILE = Path(__file__).resolve().parents[1] / "configs/g1/trash-subtasks.example.yaml"
# pnp_trash_subtasks meta/tasks.jsonl; the policy only knows these exact strings.
TRAINED_PROMPTS = {
    "approach brown table",
    "pick the apple",
    "pick the cup",
    "pick the pill bottle",
    "pick the pill box",
    "pick the red bottle",
    "turn left and approach the trash bin",
    "turn right and approach the trash bin",
    "put it in to the trash bin",
}


def test_trash_profile_registers_exactly_the_trained_prompts():
    profile = load_profile(PROFILE)
    assert {s.prompt for s in profile.skills.values()} == TRAINED_PROMPTS
    # Every step before the last keeps its hand targets for the next skill.
    actions = {s.skill_id: s.completion_action for s in profile.skills.values()}
    assert actions.pop("put_in_trash_bin") == "reset_standing"
    assert set(actions.values()) == {"hold"}
