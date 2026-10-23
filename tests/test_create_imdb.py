"""Tests for collecting intensity measure files into an IMDB."""

from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from imdb import IMDB
from workflow.scripts import create_imdb

STATIONS = ["WEL", "AKL1", "0a1b2c3"]
PERIODS = [0.1, 1.0]
SIMULATED_COMPONENTS = ["000", "090", "ver", "geom", "rotd50"]


def write_im_file(
    path: Path,
    name: str,
    frequencies: tuple[float, ...] = (0.5, 1.0),
    supergrid_depth: tuple[float, ...] = (0.0, 0.0, 0.0),
) -> Path:
    """Write a small IM file in the `im-calc` layout."""
    n_stations = len(STATIONS)
    station_coords = {
        "station": STATIONS,
        "latitude": ("station", np.linspace(-41.0, -40.0, n_stations)),
        "longitude": ("station", np.linspace(174.0, 175.0, n_stations)),
        "vs30": ("station", np.linspace(200.0, 600.0, n_stations)),
        "z1pt0": ("station", np.linspace(0.1, 0.3, n_stations)),
        "z2pt5": ("station", np.linspace(1.0, 3.0, n_stations)),
        "rrup": ("station", np.linspace(10.0, 30.0, n_stations)),
        "rjb": ("station", np.linspace(5.0, 25.0, n_stations)),
        "supergrid_depth": ("station", np.array(supergrid_depth)),
    }

    def components(
        shape: tuple[int, ...], dims: tuple[str, ...], value: float
    ) -> dict[str, tuple]:
        return {c: (dims, np.full(shape, value)) for c in SIMULATED_COMPONENTS}

    tree = xr.DataTree.from_dict(
        {
            "PGA": xr.Dataset(
                components((n_stations,), ("station",), 0.2), coords=station_coords
            ),
            "CAV": xr.Dataset(
                {"000": ("station", np.full(n_stations, 3.0))}, coords=station_coords
            ),
            "pSA": xr.Dataset(
                components((n_stations, len(PERIODS)), ("station", "period"), 0.5),
                coords=station_coords | {"period": PERIODS},
            ),
            "FAS": xr.Dataset(
                {
                    "eas": (
                        ("station", "frequency"),
                        np.ones((n_stations, len(frequencies))),
                    )
                },
                coords=station_coords | {"frequency": list(frequencies)},
            ),
            "pSA/empirical/NSHM2022": xr.Dataset(
                {
                    "mean": (
                        ("station", "period"),
                        np.zeros((n_stations, len(PERIODS))),
                    ),
                    "std_Total": (
                        ("station", "period"),
                        np.full((n_stations, len(PERIODS)), 0.6),
                    ),
                },
                coords=station_coords | {"period": PERIODS},
                attrs={
                    "intensity_measure": "pSA",
                    "model": "NSHM2022",
                    "component": "rotd50",
                },
            ),
            "CAV/empirical/BCH_16": xr.Dataset(
                {
                    "mean": ("station", np.full(n_stations, np.log(2.0))),
                    "std_Total": ("station", np.full(n_stations, 0.4)),
                },
                coords=station_coords,
                attrs={
                    "intensity_measure": "CAV",
                    "model": "BCH_16",
                    "component": "geom",
                },
            ),
        }
    )
    tree.attrs = {
        "event": name,
        "magnitude": 6.0,
        "rake": 90.0,
        "dip": 45.0,
        "ztor": 1.0,
        "zbot": 15.0,
        "hypo_lat": -40.5,
        "hypo_lon": 174.5,
        "hypo_depth": 8.0,
        "source": "POLYGON ((174 -41, 175 -41, 175 -40, 174 -41))",
        "trace": "LINESTRING (174 -41, 175 -41)",
        "domain": "POLYGON ((173 -42, 176 -42, 176 -39, 173 -42))",
        "tect_type": "active_shallow",
    }
    tree.to_netcdf(path)
    return path


@pytest.fixture
def imdb_ffp(tmp_path: Path) -> Path:
    """An IMDB of two realisations of one event and one of another."""
    output_ffp = tmp_path / "imdb.duckdb"
    create_imdb.create_imdb(
        output_ffp,
        [
            write_im_file(tmp_path / "b.h5", "Beta"),
            write_im_file(tmp_path / "a2.h5", "Alpha_R2", frequencies=(0.5,)),
            write_im_file(
                tmp_path / "a1.h5",
                "Alpha_R1",
                supergrid_depth=(0.0, 0.0, 400.0),
            ),
        ],
    )
    return output_ffp


def test_groups_realisations_into_events(imdb_ffp: Path) -> None:
    with IMDB(imdb_ffp) as db:
        assert db.validate() == []
        assert db.get_events().index.tolist() == ["Alpha", "Beta"]
        assert db.get_realisations()["event_int_id"].to_dict() == {
            "Alpha_R1": 0,
            "Alpha_R2": 0,
            "Beta": 1,
        }
        assert len(db.get_sites()) == len(STATIONS)
        assert len(db.get_site_event()) == 2 * len(STATIONS)


def test_describes_the_source_and_sites(imdb_ffp: Path) -> None:
    with IMDB(imdb_ffp) as db:
        event = db.get_events().loc["Alpha"]
        sites = db.get_sites()
    assert event["tect_type"] == "ACTIVE_SHALLOW"
    assert event["magnitude"] == pytest.approx(6.0)
    assert sites["is_real"].to_dict() == {"WEL": True, "AKL1": True, "0a1b2c3": False}
    assert sites.loc["AKL1", "z1p0"] == pytest.approx(0.2)


def test_leaves_out_simulations_in_the_absorbing_layer(imdb_ffp: Path) -> None:
    with IMDB(imdb_ffp) as db:
        simulated = db.get_records(rel_ids=["Alpha_R1"], kind="simulated")
        empirical = db.get_records(rel_ids=["Alpha_R1"], kind="gmm")
    assert "0a1b2c3" not in simulated["site_id"].values
    assert "0a1b2c3" in empirical["site_id"].values


def test_stores_empirical_medians_on_their_components(imdb_ffp: Path) -> None:
    with IMDB(imdb_ffp) as db:
        psa = db.get_psa(sigma=True, rel_ids=["Beta"], gmm_key="NSHM2022")
        cav = db.get_scalars(["CAV"], sigma=True, rel_ids=["Beta"], gmm_key="BCH_16")
        components = db.get_records(rel_ids=["Beta"], kind="gmm").groupby("gmm_key")[
            "component"
        ]
    assert np.allclose(psa[["0.1", "1.0"]], 1.0)
    assert np.allclose(psa[["0.1_sigma", "1.0_sigma"]], 0.6)
    assert np.allclose(cav["CAV"], 2.0)
    assert np.allclose(cav["CAV_sigma"], 0.4)
    assert components.unique().to_dict() == {"BCH_16": ["geom"], "NSHM2022": ["rotd50"]}


def test_records_hold_only_the_ims_of_their_component(imdb_ffp: Path) -> None:
    with IMDB(imdb_ffp) as db:
        records = db.get_records(rel_ids=["Beta"], kind="simulated")
        scalars = db.get_scalars(rel_ids=["Beta"], kind="simulated")
        fas = db.get_fas(rel_ids=["Beta"], kind="simulated")
    eas = records.index[records["component"] == "eas"]
    assert set(fas.index) == set(eas)
    assert scalars.index.intersection(eas).empty
    cav = scalars["CAV"].dropna()
    assert set(records.loc[cav.index, "component"]) == {"000"}


def test_places_spectra_on_a_shared_grid(imdb_ffp: Path) -> None:
    with IMDB(imdb_ffp) as db:
        fas = db.get_fas(rel_ids=["Alpha_R2"], kind="simulated")
    assert np.allclose(fas["0.5"], 1.0)
    assert fas["1.0"].isna().all()


def test_rejects_repeated_realisation_names(tmp_path: Path) -> None:
    im_ffps = [
        write_im_file(tmp_path / "1.h5", "Alpha_R1"),
        write_im_file(tmp_path / "2.h5", "Alpha_R1"),
    ]
    with pytest.raises(ValueError, match="realisation name"):
        create_imdb.create_imdb(tmp_path / "imdb.duckdb", im_ffps)
    assert not (tmp_path / "imdb.duckdb").exists()
