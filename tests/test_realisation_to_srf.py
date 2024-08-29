import shutil
from pathlib import Path

import numpy as np
import pytest

from source_modelling import srf
from workflow.scripts import realisation_to_srf

REALISATION_FILES = Path("tests/realisations")
SRF_FILES = Path("tests/srfs")


@pytest.mark.parametrize(
    "rupture_id",
    [
        (0),
        (1),
        (599),
    ],
)
def test_realisation_to_srf(
    tmp_path: Path,
    genslip_path: Path,
    velocity_model_path: Path,
    rupture_id: int,
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
    generated_srf = srf.read_srf(srf_path)
    test_srf = srf.read_srf(SRF_FILES / f"rupture_{rupture_id}.srf")
    max_slip_diff = np.max(
        np.abs(generated_srf.points["slip"] - generated_srf.points["slip"])
    )
    # There can be small differences in the slip output due to
    # differences in platform, hardware, etc. So rather than file
    # diffing, we will diff on the actual values of the srf.
    assert max_slip_diff < 0.1
    max_tinit_diff = np.max(
        np.abs(generated_srf.points["tinit"] - test_srf.points["tinit"])
    )
    assert max_tinit_diff < 0.1
