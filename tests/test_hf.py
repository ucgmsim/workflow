import os
import shutil
from pathlib import Path

import numpy as np
import pytest

from workflow.realisations import DomainParameters
from workflow.scripts import hf_sim

HF_INPUT_PATH = Path("tests/hf")


@pytest.mark.parametrize("rupture_id,expected_station_count", [(0, 3), (599, 30)])
def test_hf_output(
    tmp_path: Path,
    rupture_id: int,
    expected_station_count: int,
    velocity_model_path: Path,
    hf_sim_path: Path,
):
    stoch_file = HF_INPUT_PATH / f"rupture_{rupture_id}" / "input.stoch"
    realisation = HF_INPUT_PATH / f"rupture_{rupture_id}" / "realisation.json"
    station_list = HF_INPUT_PATH / f"rupture_{rupture_id}" / "station_list.ll"
    expected_output_hf = HF_INPUT_PATH / f"rupture_{rupture_id}" / "output.hf"

    shutil.copyfile(stoch_file, tmp_path / "input.stoch")
    shutil.copyfile(realisation, tmp_path / "realisation.json")
    shutil.copyfile(station_list, tmp_path / "station_list.ll")

    work_directory = tmp_path / "work"
    work_directory.mkdir()

    hf_sim.run_hf(
        tmp_path / "realisation.json",
        tmp_path / "input.stoch",
        tmp_path / "station_list.ll",
        tmp_path / "output.hf",
        velocity_model=velocity_model_path,
        hf_sim_path=hf_sim_path,
        work_directory=work_directory,
    )

    assert (
        os.stat(tmp_path / "output.hf").st_size == os.stat(expected_output_hf).st_size
    )
    with open(tmp_path / "output.hf", "rb") as test_output, open(
        expected_output_hf, "rb"
    ) as expected_output:
        header_size = 288
        stations_output_size = 24 * expected_station_count
        offset = 512
        domain_parameters = DomainParameters.read_from_realisation(realisation)
        nt = int(np.round(domain_parameters.duration / domain_parameters.dt))
        test_output.seek(header_size + offset + stations_output_size)
        expected_output.seek(header_size + offset + stations_output_size)
        test_waveforms = np.fromfile(test_output, dtype=np.float32).reshape(
            (expected_station_count, nt, 3)
        )
        expected_waveforms = np.fromfile(expected_output, dtype=np.float32).reshape(
            (expected_station_count, nt, 3)
        )
        assert (np.abs(test_waveforms - expected_waveforms) < 5e-2).all()
