import filecmp
import shutil
from pathlib import Path

import pytest

from workflow.scripts import generate_stoch

REALISATION_FILES = Path("tests/realisations")
SRF_FILES = Path("tests/srfs")
STOCH_FILES = Path("tests/stoch")


@pytest.mark.parametrize(
    "rupture_id",
    [
        (0),
        (1),
        (599),
    ],
)
def test_generate_stoch(tmp_path: Path, srf2stoch_path: Path, rupture_id: int):
    realisation_path = tmp_path / "realisation.json"
    stoch_path = tmp_path / "realisation.stoch"
    shutil.copy(REALISATION_FILES / f"rupture_{rupture_id}.json", realisation_path)
    generate_stoch.generate_stoch(
        realisation_path,
        SRF_FILES / f"rupture_{rupture_id}.srf",
        stoch_path,
        srf2stoch_path=srf2stoch_path,
    )
    assert filecmp.cmp(
        stoch_path, STOCH_FILES / f"rupture_{rupture_id}.stoch", shallow=False
    )
