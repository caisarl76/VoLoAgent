import os
from pathlib import Path

import pytest


def pytest_addoption(parser):
    parser.addoption("--g1-native-checkout", default=os.getenv("G1_NATIVE_CHECKOUT"))
    parser.addoption("--g1-native-python", default=os.getenv("G1_NATIVE_PYTHON"))
    parser.addoption("--require-g1-e2e", action="store_true")


@pytest.fixture
def native_runtime(request):
    checkout = request.config.getoption("--g1-native-checkout")
    python = request.config.getoption("--g1-native-python")
    if not checkout or not python:
        message = (
            "G1 cross-process checks require an explicit native checkout and Python"
        )
        if request.config.getoption("--require-g1-e2e"):
            pytest.fail(message)
        pytest.skip(message)
    checkout, python = Path(checkout), Path(python)
    assert python.is_file(), python
    assert (checkout / "gear_sonic/tests/harness_fake_runtime.py").is_file()
    return checkout, python
