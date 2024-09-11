import difflib
from pathlib import Path

import pytest

from workflow.scripts import generate_model_coordinates

MODEL_COORDINATES_OUTPUT = Path("tests") / "model_coordinates"
REALSATIONS_PATH = Path("tests") / "realisations" / "with_velocity_model_parameters"


@pytest.mark.parametrize("rupture_id", [0, 1, 599, 1190])
def test_generate_model_coordinates(tmp_path: Path, rupture_id: int):
    realisation = REALSATIONS_PATH / f"rupture_{rupture_id}.json"
    expected_model_coordinates = (
        MODEL_COORDINATES_OUTPUT / f"rupture_{rupture_id}" / "model_params"
    )
    expected_grid_file = (
        MODEL_COORDINATES_OUTPUT / f"rupture_{rupture_id}" / "grid_file"
    )
    generate_model_coordinates.generate_model_coordinates(realisation, tmp_path)
    model_coordinates_diff = list(
        difflib.unified_diff(
            expected_model_coordinates.read_text().split("\n"),
            (tmp_path / "model_params").read_text().split("\n"),
            fromfile="expected",
            tofile="actual",
        )
    )
    assert model_coordinates_diff == [], "Files do not match:\n" + "\n".join(
        model_coordinates_diff
    )
    grid_file_diff = list(
        difflib.unified_diff(
            expected_grid_file.read_text().split("\n"),
            (tmp_path / "grid_file").read_text().split("\n"),
            fromfile="expected",
            tofile="actual",
        )
    )
    assert grid_file_diff == [], "Files do not match:\n" + "\n".join(grid_file_diff)
