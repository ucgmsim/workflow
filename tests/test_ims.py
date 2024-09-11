import shutil
import difflib
import pytest
from pathlib import Path

from workflow.scripts import im_calc

IMS_PATH = Path('tests') / 'ims'

@pytest.mark.parametrize('id', [1, 2, 3, 4, 5])
def test_ims(tmp_path: Path, id: int):
    bb_input = IMS_PATH / str(id) / 'BB.bin'
    realisation_ffp = IMS_PATH / 'realisation.json'
    shutil.copyfile(realisation_ffp, tmp_path / 'realisation.json')
    im_calc.calculate_instensity_measures(
        tmp_path / 'realisation.json',
        bb_input,
        tmp_path
    )
    output_ims_path = tmp_path / 'realisation.csv'
    output_ims = output_ims_path.read_text().split('\n')
    expected_ims_path = IMS_PATH / str(id) / 'realisation.csv'
    expected_ims = expected_ims_path.read_text().split('\n')
    diff = list(difflib.unified_diff(expected_ims, output_ims, fromfile='expected', tofile='actual'))
    assert diff == [], "Unexpected file contents:\n" + "\n".join(diff)
