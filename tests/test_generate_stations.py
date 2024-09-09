import difflib
from pathlib import Path

import pytest

from workflow.scripts import generate_station_coordinates

STATIONS_INPUT = Path("tests") / "stations"


@pytest.mark.parametrize("rupture_id", [0, 1, 599, 1190])
def test_station_coordinates(
    tmp_path: Path,
    rupture_id: int,
):
    rupture_directory = STATIONS_INPUT / f"rupture_{rupture_id}"
    station_path = STATIONS_INPUT / "stations.ll"
    realisation_path = rupture_directory / "realisation.json"
    expected_statcords = rupture_directory / "stations.statcords"
    output_statcords = tmp_path / "stations.statcords"
    generate_station_coordinates.generate_fd_files(
        realisation_path, tmp_path, stat_file=station_path
    )
    output_statlines = output_statcords.read_text().split("\n")
    expected_statlines = expected_statcords.read_text().split("\n")
    diff = list(difflib.unified_diff(expected_statlines, output_statlines, fromfile='expected', tofile='actual'))
    assert diff == [], "Unexpected file contents:\n" + "\n".join(diff)


def test_dummy_scenario(tmp_path: Path):
    dummy_directory = STATIONS_INPUT / 'dummy'
    station_path = dummy_directory / "stations.ll"
    realisation_path = dummy_directory / "realisation.json"
    expected_statcords = dummy_directory / "stations.statcords"
    output_statcords = tmp_path / "stations.statcords"
    generate_station_coordinates.generate_fd_files(
        realisation_path, tmp_path, stat_file=station_path
    )
    output_statlines = output_statcords.read_text().split("\n")
    expected_statlines = expected_statcords.read_text().split("\n")
    diff = list(difflib.unified_diff(expected_statlines, output_statlines, fromfile='expected', tofile='actual'))
    assert diff == [], "Unexpected file contents:\n" + "\n".join(diff)
