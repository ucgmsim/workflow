"""Tests for workflow.scripts.im_calc."""

import functools

import numpy as np
import pytest
import xarray as xr

from IM.ims import IM
from qcore import coordinates
from source_modelling import sources
from workflow import gmm, realisations
from workflow.scripts import im_calc


def test_im_function_map_covers_every_im_except_fas() -> None:
    # FAS is added separately by the caller because it additionally requires
    # a KO matrix directory, so it is expected to be absent here.
    function_map = im_calc._im_function_map(dt=0.01, psa_periods=np.array([1.0]))

    assert set(function_map) == set(IM) - {IM.FAS}


def test_cav5_uses_a_5_cm_per_s_squared_threshold() -> None:
    function_map = im_calc._im_function_map(dt=0.01, psa_periods=np.array([1.0]))

    cav5_function = function_map[IM.CAV5]
    assert isinstance(cav5_function, functools.partial)
    assert cav5_function.keywords["threshold"] == 5.0


def test_every_empirical_im_has_a_component() -> None:
    assert set(im_calc.EMPIRICAL_COMPONENTS) == set(im_calc.EMPIRICAL_IM_NAMES)


def test_calculate_distances_reports_rx_ry_ry0_in_km() -> None:
    # A vertical, north-striking 20 km plane centred on the origin (NZTM),
    # with one station 10 km east of the centroid and one 30 km north.
    centroid_nztm = np.array([5_180_000.0, 1_570_000.0, 5_000.0])
    plane = sources.Plane.from_centroid_strike_dip(
        coordinates.nztm_to_wgs_depth(centroid_nztm),
        dip=90.0,
        length=20.0,
        width=10.0,
        strike_nztm=0.0,
    )
    stations_nztm = centroid_nztm[:2] + np.array([[0.0, 10_000.0], [30_000.0, 0.0]])
    stations = coordinates.nztm_to_wgs_depth(stations_nztm)
    broadband = xr.Dataset(
        coords={
            "station": ["EAST", "NRTH"],
            "latitude": ("station", stations[:, 0]),
            "longitude": ("station", stations[:, 1]),
        }
    )

    distances = im_calc.calculate_distances(
        realisations.SourceConfig(source_geometries={"plane": plane}),
        np.array([*coordinates.nztm_to_wgs_depth(centroid_nztm[:2]), 5.0]),
        broadband,
    )

    assert distances.rx is not None and distances.ry0 is not None
    assert abs(distances.rx.sel(station="EAST").item()) == pytest.approx(10.0, abs=0.1)
    assert distances.ry0.sel(station="EAST").item() == pytest.approx(0.0, abs=0.1)
    assert distances.ry0.sel(station="NRTH").item() == pytest.approx(20.0, abs=0.1)
    assert set(distances.as_dict()) <= set(im_calc.COORDINATE_METADATA)


def _station_values(values: list[float]) -> xr.DataArray:
    return xr.DataArray(
        np.array(values, dtype=np.float64),
        dims="station",
        coords={"station": ["A", "B", "C"]},
    )


@pytest.mark.parametrize("tect_type", list(gmm.TECT_TYPES))
def test_calculate_empirical_names_statistics_for_create_imdb(tect_type: str) -> None:
    """The NSHM 2022 logic tree runs for every tectonic type, with the statistic names create-imdb reads."""
    source_parameters = im_calc.SourceParameters(
        mag=6.5,
        avg_rake=90.0,
        avg_dip=60.0,
        avg_ztor=1.0,
        avg_zbot=15.0,
        hypo_depth=8.0,
    )
    site_parameters = im_calc.SiteParameters(
        vs30=_station_values([300.0, 500.0, 800.0]),
        z1pt0=_station_values([0.3, 0.2, 0.05]),
        z2pt5=_station_values([2.0, 1.5, 0.6]),
    )
    distances = im_calc.Distances(
        rrup=_station_values([5.0, 30.0, 120.0]),
        rjb=_station_values([3.0, 29.0, 119.0]),
        hyp=_station_values([10.0, 32.0, 121.0]),
        epi=_station_values([6.0, 31.0, 120.0]),
        rx=_station_values([2.0, -20.0, 100.0]),
        ry=_station_values([1.0, 5.0, 50.0]),
        ry0=_station_values([0.0, 0.0, 30.0]),
    )
    periods = np.array([0.1, 1.0])

    with pytest.warns(UserWarning, match="Skipping empirical PGA for NSHM2022"):
        results = im_calc.calculate_empirical(
            realisations.EmpiricalParameters(tect_type=tect_type, models=["NSHM2022"]),
            source_parameters,
            site_parameters,
            distances,
            [IM.PGA, IM.pSA],
            periods,
        )

    assert list(results) == ["pSA/empirical/NSHM2022"]
    psa = results["pSA/empirical/NSHM2022"]
    assert set(psa.data_vars) == {"mean", "std_Total"}
    assert psa["mean"].dims == ("station", "period")
    assert list(psa.station.values) == ["A", "B", "C"]
    np.testing.assert_array_equal(psa.period.values, periods)
    assert np.isfinite(psa.to_array()).all()
    # Ground motion decays with distance.
    assert (psa["mean"].diff("station") < 0).all()
