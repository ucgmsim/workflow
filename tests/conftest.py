"""Fixtures shared across the test suite."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from workflow import defaults
from workflow.realisations import NZCVMSettings

NZCVM_DATA_PREFIX = "/nzcvm/"
"""Where the scientific defaults expect NZCVM's data root."""


def _nzcvm(*arguments: str | Path) -> None:
    """Run the installed `nzcvm` command line."""
    subprocess.run(
        [str(Path(sys.executable).parent / "nzcvm"), *map(str, arguments)],
        check=True,
        capture_output=True,
    )


@pytest.fixture(scope="session")
def nzcvm_data_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """NZCVM's synthetic data root, laid out like the real one.

    Built with the installed nzcvm, the way nzcvm's own `just synthetic`
    recipe builds it, minus the basins (which need nzcvm's meshing extra).
    """
    root = tmp_path_factory.mktemp("nzcvm")
    scratch = tmp_path_factory.mktemp("nzcvm-inputs")
    (root / "resources").mkdir()
    (root / "models").mkdir()
    _nzcvm("synthetic", "dem", scratch / "dem.h5")
    _nzcvm("surface", "convert", scratch / "dem.h5", root / "resources/dem.zarr")
    _nzcvm("synthetic", "vs30", scratch / "vs30.h5")
    _nzcvm(
        "surface", "convert", scratch / "vs30.h5",
        root / "resources/vs30.zarr", "--scalar-key", "vs30", "--no-flip",
    )  # fmt: skip
    _nzcvm("synthetic", "coastline", root / "resources/coastline.wkb.gz")
    _nzcvm("synthetic", "tomography", scratch / "tomography.csv")
    _nzcvm(
        "tomography", "convert", scratch / "tomography.csv",
        root / "models/tomography.zarr",
    )  # fmt: skip
    return root


@pytest.fixture(scope="session")
def nzcvm_settings(nzcvm_data_root: Path) -> NZCVMSettings:
    """The default NZCVM settings, reading the synthetic data root."""
    settings = json.dumps(
        defaults.load_defaults(defaults.DefaultsVersion.v26_7_1Hz)["nzcvm"]
    ).replace(f'"{NZCVM_DATA_PREFIX}', f'"{nzcvm_data_root}/')
    return NZCVMSettings.from_dict(json.loads(settings))
