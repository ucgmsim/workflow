from pathlib import Path

import pytest
from pytest import Parser


def pytest_addoption(parser: Parser):
    parser.addoption(
        "--genslip-path",
        action="store",
        type=Path,
        help="Path to genslip binary",
    )
    parser.addoption(
        "--velocity-model-path",
        action="store",
        type=Path,
        help="Path to velocity model",
    )
    parser.addoption(
        "--nshmdb-path",
        action="store",
        type=Path,
        help="Path to nshmdb",
    )


@pytest.fixture
def genslip_path(request):
    return request.config.getoption("--genslip-path")


@pytest.fixture
def velocity_model_path(request):
    return request.config.getoption("--velocity-model-path")


@pytest.fixture
def nshmdb_path(request):
    return request.config.getoption("--nshmdb-path")
