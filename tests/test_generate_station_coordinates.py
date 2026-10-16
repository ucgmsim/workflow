from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest

from qcore import coordinates
from workflow import sites
from workflow.bounding_box import BoundingBox
from workflow.defaults import DefaultsVersion
from workflow.realisations import DomainParameters, RealisationMetadata, Resolution
from workflow.scripts import generate_station_coordinates
from workflow.scripts.generate_station_coordinates import Format

RESOLUTION = 0.5


@pytest.fixture
def realisation(tmp_path: Path) -> Path:
    """A 20.5 km by 20 km domain centred on Christchurch, on a 0.5 km grid.

    The grid is 41 points wide and 40 high, to cover both an odd and an even
    number of grid points.
    """
    realisation = tmp_path / "realisation.json"
    RealisationMetadata(
        name="Test_REL01", version="1", defaults_version=DefaultsVersion.v24_2_2_1
    ).write_to_realisation(realisation)
    DomainParameters(
        domain=BoundingBox.from_centroid_bearing_extents(
            centroid=np.array([-43.5, 172.5]),
            bearing=30.0,
            extent_x=20.5,
            extent_y=20.0,
        ),
        depth=10.0,
        duration=35.0,
    ).write_to_realisation(realisation)
    Resolution(resolution=RESOLUTION).write_to_realisation(realisation)
    return realisation


def site_table() -> pd.DataFrame:
    """Two stations in the domain, one far outside it, which is missing a value."""
    return pd.DataFrame(
        {
            "latitude": [-43.5, -43.52, -41.3],
            "longitude": [172.5, 172.53, 174.8],
            "vs30": [400.0, 250.0, 300.0],
            "z1pt0": [0.2, 0.4, np.nan],
            "z2pt5": [1.2, 2.0, 1.5],
            "kappa_s": [0.045, 0.03, 0.04],
            "q_frequency_exponent": [0.6, 0.5, 0.6],
        },
        index=pd.Index(["CHCH", "RICC", "WEL"], name="station"),
    )


@pytest.mark.parametrize("solver", list(Format))
def test_in_domain_sites_are_written(
    realisation: Path, tmp_path: Path, solver: Format
) -> None:
    """Only in-domain stations are kept, so the out-of-domain gap is no error."""
    sites_ffp = tmp_path / "sites.parquet"
    site_table().to_parquet(sites_ffp)
    output = tmp_path / "stations"

    generate_station_coordinates.generate_fd_files(
        realisation, sites_ffp, output, format=solver
    )

    stations = sites.read_sites(output / "stations.parquet")
    assert list(stations.index) == ["CHCH", "RICC"]
    expected = site_table().loc[["CHCH", "RICC"]]
    pd.testing.assert_frame_equal(
        stations.drop(columns=["latitude", "longitude"]),
        expected.drop(columns=["latitude", "longitude"]),
    )
    # Within half a grid cell, diagonally, of where the station really is.
    np.testing.assert_allclose(
        stations[["latitude", "longitude"]],
        expected[["latitude", "longitude"]],
        atol=0.005,
    )
    match solver:
        case Format.EMOD3D:
            lines = (output / "stations.statcords").read_text().splitlines()
            assert lines[0] == "2"
            assert [line.split()[-1] for line in lines[1:]] == ["CHCH", "RICC"]

            # Each written location is its statcords grid point, and that grid
            # point is the one nearest the station.
            domain = DomainParameters.read_from_realisation(realisation)
            mlat, mlon = domain.domain.origin
            proj = coordinates.SphericalProjection(
                mlat=mlat, mlon=mlon, mrot=float(domain.domain.great_circle_bearing)
            )
            centre = np.array([domain.nx(RESOLUTION), domain.ny(RESOLUTION)]) / 2

            def grid_position(table: pd.DataFrame) -> np.ndarray:
                lat = table["latitude"].to_numpy()
                lon = table["longitude"].to_numpy()
                return proj(lat=lat, lon=lon) / RESOLUTION + centre

            statcords = np.loadtxt(
                output / "stations.statcords", skiprows=1, usecols=(0, 1)
            )
            np.testing.assert_allclose(grid_position(stations), statcords, atol=1e-6)
            assert (np.abs(grid_position(expected) - statcords) <= 0.5).all()
        case Format.SW4:
            with h5py.File(output / "stations.h5") as handle:
                assert sorted(handle) == ["CHCH", "RICC"]
            # SW4 records at the station itself, not at a grid point.
            pd.testing.assert_frame_equal(stations, expected)


@pytest.mark.parametrize("solver", list(Format))
def test_in_domain_stations_need_every_value(
    realisation: Path, tmp_path: Path, solver: Format
) -> None:
    table = site_table()
    table.loc["RICC", "kappa_s"] = np.nan
    sites_ffp = tmp_path / "sites.parquet"
    table.to_parquet(sites_ffp)
    output = tmp_path / "stations"

    with pytest.raises(ValueError, match="RICC"):
        generate_station_coordinates.generate_fd_files(
            realisation, sites_ffp, output, format=solver
        )
    # Nothing is written for a table that fails validation.
    assert not any(output.iterdir())


@pytest.mark.parametrize("solver", list(Format))
@pytest.mark.parametrize("column", ["latitude", "longitude"])
def test_every_station_needs_a_location(
    realisation: Path, tmp_path: Path, solver: Format, column: str
) -> None:
    """A station without a location fails, rather than being dropped as out of domain."""
    table = site_table()
    table.loc["WEL", column] = np.nan
    sites_ffp = tmp_path / "sites.parquet"
    table.to_parquet(sites_ffp)
    output = tmp_path / "stations"

    with pytest.raises(
        ValueError, match=f"missing locations: {{'{column}': \\['WEL'\\]}}"
    ):
        generate_station_coordinates.generate_fd_files(
            realisation, sites_ffp, output, format=solver
        )
    assert not any(output.iterdir())


@pytest.mark.parametrize("solver", list(Format))
def test_station_must_be_the_index(
    realisation: Path, tmp_path: Path, solver: Format
) -> None:
    sites_ffp = tmp_path / "sites.parquet"
    site_table().reset_index().to_parquet(sites_ffp)
    output = tmp_path / "stations"

    with pytest.raises(ValueError, match="indexed by 'station'"):
        generate_station_coordinates.generate_fd_files(
            realisation, sites_ffp, output, format=solver
        )
    assert not any(output.iterdir())


@pytest.mark.parametrize("name", ["CH CH", "CHCH ", "\tCHCH", ""])
def test_emod3d_station_names_are_single_words(
    realisation: Path, tmp_path: Path, name: str
) -> None:
    sites_ffp = tmp_path / "sites.parquet"
    site_table().rename(index={"CHCH": name}).to_parquet(sites_ffp)
    output = tmp_path / "stations"

    with pytest.raises(ValueError, match="no whitespace"):
        generate_station_coordinates.generate_fd_files(
            realisation, sites_ffp, output, format=Format.EMOD3D
        )
    assert not any(output.iterdir())


def test_sw4_station_names_may_contain_spaces(
    realisation: Path, tmp_path: Path
) -> None:
    sites_ffp = tmp_path / "sites.parquet"
    site_table().rename(index={"CHCH": "CH CH"}).to_parquet(sites_ffp)
    output = tmp_path / "stations"

    generate_station_coordinates.generate_fd_files(
        realisation, sites_ffp, output, format=Format.SW4
    )
    assert list(sites.read_sites(output / "stations.parquet").index) == [
        "CH CH",
        "RICC",
    ]
