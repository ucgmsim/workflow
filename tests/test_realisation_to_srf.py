import hashlib
import shutil
from pathlib import Path

import pytest

from workflow.scripts import realisation_to_srf

REALISATION_FILES = Path("tests/realisations")


def md5file(filepath: Path) -> str:
    with open(filepath, "rb") as handle:
        return hashlib.file_digest(handle, "md5").hexdigest()


@pytest.mark.parametrize(
    "rupture_id,valid_srf_hash",
    [
        (0, "08bcaaa78e8a38cf3d0afb32df2d626b"),
        (1, "02d50bdea955b0d044b1402bac62207c"),
        (599, "fcb66cce0e5576014a600b7fbcf8bb13"),
        (1190, "478e506fc86638f31be8289124a53ce2"),
    ],
)
def test_realisation_to_srf(
    tmp_path: Path,
    genslip_path: Path,
    velocity_model_path: Path,
    rupture_id: int,
    valid_srf_hash: str,
):
    realisation_path = tmp_path / "realisation.json"
    work_directory = tmp_path / "work"
    srf_path = tmp_path / "realisation.srf"
    work_directory.mkdir()
    shutil.copy(REALISATION_FILES / f"rupture_{rupture_id}.json", realisation_path)
    realisation_to_srf.generate_srf(
        realisation_path,
        srf_path,
        work_directory=work_directory,
        velocity_model=velocity_model_path,
        genslip_path=genslip_path,
    )
    assert md5file(srf_path) == valid_srf_hash
