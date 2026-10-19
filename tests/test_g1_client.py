import importlib.util
import importlib

import pytest


def client_module():
    assert importlib.util.find_spec("vlm_orchestrator.harness.g1.client"), (
        "RPC client missing"
    )
    return importlib.import_module("vlm_orchestrator.harness.g1.client")


def test_req_socket_recreated_after_timeout(tmp_path):
    c = client_module().G1Client(
        "ipc://" + str(tmp_path / "unavailable.sock"), timeout_s=0.02
    )
    try:
        with pytest.raises(TimeoutError):
            c.get_status()
        with pytest.raises(TimeoutError):
            c.get_status()  # A timed-out REQ must not retain EFSM state.
    finally:
        c.close()


def test_client_rejects_remote_control_endpoint():
    with pytest.raises(ValueError):
        client_module().G1Client("tcp://192.168.0.2:9000")
