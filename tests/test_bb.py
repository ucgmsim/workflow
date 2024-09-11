import filecmp
import subprocess
from pathlib import Path

import pytest

from workflow.scripts import bb_sim


@pytest.mark.parametrize('realisation,vs3dfile_url', [('2013p552776','https://www.dropbox.com/scl/fi/r41dwth8wo2r5mgxdw05r/vs3dfile.s?rlkey=s36skaejia56ug119pppyc2rb&st=fdm9v6ns&dl=0')])
def test_bb_sim_2013p552776(tmp_path: Path, realisation: str, vs3dfile_url: str):
    bb_test_data_path = Path('tests') / 'bb' / realisation
    expected_bb_path = bb_test_data_path / 'BB_expected.bin'
    output_bb_path = tmp_path / 'BB.bin'
    work_directory = tmp_path / 'work'
    work_directory.mkdir()
    vm_path = tmp_path / 'VM'
    vm_path.mkdir(exist_ok=True)
    subprocess.check_call(['wget', vs3dfile_url, '-O', str(vm_path / 'vs3dfile.s')])
    bb_sim.combine_hf_and_lf(
        bb_test_data_path / 'realisation.json',
        bb_test_data_path / 'stations.ll',
        bb_test_data_path / 'stations.vs30',
        # NOTE: Even though it feels like you can remove the OutBin
        # directory, this *will* cause the tests to fail...
        bb_test_data_path / 'LF' / 'OutBin',
        bb_test_data_path / 'HF.bin',
        vm_path,
        output_bb_path,
        work_directory=work_directory
    )
    assert filecmp.cmp(tmp_path / 'BB.bin', expected_bb_path, shallow=False)
