import pytest
import filecmp
import shutil

from pathlib import Path

from qcore.data import download_data

from workflow.scripts import generate_velocity_model_parameters

REALISATION_FILES = Path("tests/realisations")
CORRECT_REALISATION_FILES = Path("tests/realisations/with_velocity_model_parameters")

@pytest.mark.parametrize('rupture_id', [0, 1, 599, 1190])
def test_velocity_model_parameter_generation(tmp_path: Path, rupture_id: int):
    download_data.download_data()
    realisation_data = REALISATION_FILES / f'rupture_{rupture_id}.json'
    test_realisation_path = tmp_path / 'realisation.json'
    shutil.copy(realisation_data, test_realisation_path)

    generate_velocity_model_parameters.generate_velocity_model_parameters(
        test_realisation_path
    )
    filecmp.cmp(
        test_realisation_path,
        CORRECT_REALISATION_FILES / f'rupture_{rupture_id}.json',
        shallow=False

    )
