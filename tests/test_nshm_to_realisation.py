import json
import subprocess
from pathlib import Path

import pytest

from workflow.defaults import DefaultsVersion
from workflow.scripts import nshm2022_to_realisation

REALISATION_FILES = Path("tests/realisations")


@pytest.mark.parametrize("rupture_id", [0, 1, 599, 1190])
def test_nshm_to_realisation(nshmdb_path: Path, tmp_path: Path, rupture_id: int):
    realisation_path = tmp_path / "realisation.json"
    nshm2022_to_realisation.generate_realisation(
        nshmdb_path, rupture_id, realisation_path, DefaultsVersion.v24_2_2_1
    )
    with open(realisation_path, "r") as generated_realisation, open(
        REALISATION_FILES / f"rupture_{rupture_id}.json", "r"
    ) as expected_realisation:
        assert json.load(generated_realisation) == json.load(expected_realisation)
