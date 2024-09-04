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
    diff = list(difflib.unified_diff(output_statlines, expected_statlines))
    assert diff == [], "Unexpected file contents:\n" + "\n".join(diff)
