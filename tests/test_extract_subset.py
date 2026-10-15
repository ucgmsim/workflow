"""Tests for extracting station subsets from waveform and IM files."""

from pathlib import Path

import numpy as np
import pytest
import xarray as xr
from typer.testing import CliRunner

from workflow.scripts import extract_subset
from workflow.waveforms import Component

STATIONS = ["CCCC", "AKL", "GRID01", "0a1b2c3", "XY", "WEL1"]
OBSERVED = ["CCCC", "AKL", "WEL1"]


def write_waveform_file(path: Path, stations: list[str] = STATIONS) -> Path:
    """Write a small waveform file in the LF/HF/BB layout."""
    n_stations = len(stations)
    waveform = np.arange(len(Component) * n_stations * 4, dtype=np.float32).reshape(
        len(Component), n_stations, 4
    )
    dataset = xr.Dataset(
        {
            "waveform": (("component", "station", "time"), waveform),
            "vs30": ("station", np.linspace(200.0, 700.0, n_stations)),
        },
        coords={
            "component": list(Component),
            "station": stations,
            "time": np.arange(4) * 0.01,
            "latitude": ("station", np.linspace(-44.0, -43.0, n_stations)),
            "longitude": ("station", np.linspace(172.0, 173.0, n_stations)),
        },
        attrs={"dt": 0.01, "units": "g"},
    )
    dataset.to_netcdf(
        path, engine="h5netcdf", encoding={"waveform": {"fletcher32": True}}
    )
    return path


def write_im_file(path: Path) -> Path:
    """Write a small IM file in the `im-calc` layout: a nested tree."""
    n_stations = len(STATIONS)
    station_coords = {
        "station": STATIONS,
        "rrup": ("station", np.arange(n_stations, dtype=float)),
    }
    tree = xr.DataTree.from_dict(
        {
            "PGA": xr.Dataset(
                {
                    "PGA": (
                        ("station", "component"),
                        np.ones((n_stations, len(Component))),
                    )
                },
                coords=station_coords | {"component": list(Component)},
            ),
            "empirical/PGA": xr.Dataset(
                {"mean": ("station", np.zeros(n_stations))}, coords=station_coords
            ),
        },
        nested=True,
    )
    tree.attrs = {"event": "test", "magnitude": 5.0}
    tree.to_netcdf(path, engine="h5netcdf")
    return path


@pytest.mark.parametrize(
    "name, matched",
    [
        ("AKL", True),
        ("CCCC", True),
        ("WEL1", True),
        ("XY", False),
        ("GRID01", False),
        ("0a1b2c3", False),
        ("AK-L", False),
    ],
)
def test_default_pattern_matches_observation_stations(name: str, matched: bool) -> None:
    assert extract_subset.matches_any(
        [name], [extract_subset.OBSERVATION_STATION_PATTERN]
    ) == [matched]


def test_matches_any_patterns() -> None:
    assert extract_subset.matches_any(STATIONS, [r"^XY$", r"^GRID"]) == [
        False,
        False,
        True,
        False,
        True,
        False,
    ]


@pytest.mark.parametrize(
    "name, expected",
    [
        ("realisation.bb", "realisation_subset.bb"),
        ("intensity_measures.h5", "intensity_measures_subset.h5"),
        ("realisation", "realisation_subset"),
    ],
)
def test_subset_path(tmp_path: Path, name: str, expected: str) -> None:
    assert extract_subset.subset_path(tmp_path / name) == tmp_path / expected


@pytest.mark.parametrize("suffix", [".lf", ".hf", ".bb"])
def test_waveform_subset(tmp_path: Path, suffix: str) -> None:
    input_ffp = write_waveform_file(tmp_path / f"realisation{suffix}")

    (output_ffp,) = extract_subset.extract_subset([input_ffp])

    assert output_ffp == tmp_path / f"realisation_subset{suffix}"
    with (
        xr.open_dataset(input_ffp) as original,
        xr.open_dataset(output_ffp) as subset,
    ):
        assert subset["station"].values.tolist() == OBSERVED
        xr.testing.assert_identical(subset, original.sel(station=OBSERVED))
        assert subset.attrs == original.attrs
        assert subset["waveform"].encoding["fletcher32"]


def test_im_subset(tmp_path: Path) -> None:
    input_ffp = write_im_file(tmp_path / "intensity_measures.h5")

    (output_ffp,) = extract_subset.extract_subset([input_ffp])

    with (
        xr.open_datatree(input_ffp) as original,
        xr.open_datatree(output_ffp) as subset,
    ):
        assert subset.attrs == original.attrs
        assert [node.path for node in subset.subtree] == [
            node.path for node in original.subtree
        ]
        for node in ("PGA", "empirical/PGA"):
            xr.testing.assert_identical(
                subset[node].to_dataset(),
                original[node].to_dataset().sel(station=OBSERVED),
            )


def test_custom_patterns(tmp_path: Path) -> None:
    input_ffp = write_waveform_file(tmp_path / "realisation.bb")

    (output_ffp,) = extract_subset.extract_subset([input_ffp], [r"^GRID", r"^XY$"])

    with xr.open_dataset(output_ffp) as subset:
        assert subset["station"].values.tolist() == ["GRID01", "XY"]


def test_no_matches_writes_empty_subset(tmp_path: Path) -> None:
    input_ffp = write_waveform_file(tmp_path / "realisation.bb")

    (output_ffp,) = extract_subset.extract_subset([input_ffp], [r"^NOPE$"])

    with xr.open_dataset(output_ffp) as subset:
        assert subset.sizes["station"] == 0
        assert subset["waveform"].shape == (len(Component), 0, 4)


def test_file_without_stations_fails(tmp_path: Path) -> None:
    input_ffp = tmp_path / "not_stations.h5"
    xr.Dataset({"x": ("time", np.zeros(3))}).to_netcdf(input_ffp, engine="h5netcdf")

    with pytest.raises(ValueError, match="no station dimension"):
        extract_subset.extract_subset([input_ffp])


def test_cli_multiple_files(tmp_path: Path) -> None:
    lf = write_waveform_file(tmp_path / "realisation.lf")
    im = write_im_file(tmp_path / "intensity_measures.h5")

    result = CliRunner().invoke(
        extract_subset.app, [str(lf), str(im), "-p", r"^AKL$", "-p", r"^WEL1$"]
    )

    assert result.exit_code == 0, result.output
    with xr.open_dataset(tmp_path / "realisation_subset.lf") as subset:
        assert subset["station"].values.tolist() == ["AKL", "WEL1"]
    with xr.open_datatree(tmp_path / "intensity_measures_subset.h5") as subset:
        pga = subset.children["PGA"].dataset
        assert pga["station"].values.tolist() == ["AKL", "WEL1"]
