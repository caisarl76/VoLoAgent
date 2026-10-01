import pytest

from test_g1_runner import Executor, LocalClient, runner
from vlm_orchestrator.harness.g1.contract import SkillCall


def test_internal_reset_execution_id_rejected_before_claim(tmp_path):
    e = Executor()
    with pytest.raises(ValueError):
        runner(e, tmp_path).run_sequence(
            [
                SkillCall("bottle_to_right_table", {}),
                SkillCall(
                    "reset_standing", {"open_hands": False, "execution_id": "bogus"}
                ),
            ]
        )
    assert not e.calls


def test_lost_start_reply_cancels_owned_observed_execution(tmp_path):
    e = Executor()

    class LostReply(LocalClient):
        def request(self, method, params, **kwargs):
            response = super().request(method, params, **kwargs)
            if method == "start_manipulation":
                raise TimeoutError("Both acceptance replies lost")
            return response

    e.client = lambda: LostReply(e)
    result = runner(e, tmp_path).run(SkillCall("bottle_to_right_table", {}))
    assert result.outcome != "completed"
    assert e.calls.count("cancel") == 1 and e.calls.count("release_control") == 1
