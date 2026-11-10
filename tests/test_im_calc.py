"""Tests for workflow.scripts.im_calc."""

import functools

import numpy as np
import pytest
import xarray as xr

from IM.ims import IM
from qcore import coordinates
from source_modelling import sources
from source_modelling.magnitude_scaling import BoldM
from workflow import realisations
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


def test_source_parameters_moment_average_the_width() -> None:
    # Two vertical planes, 10 km and 20 km wide. The wider plane is 1.5
    # magnitude units larger (~178x the moment), so the averaged width sits
    # just under 20 km rather than at the 15 km arithmetic mean.
    def plane(northing: float, width: float) -> sources.Plane:
        return sources.Plane.from_centroid_strike_dip(
            coordinates.nztm_to_wgs_depth(np.array([northing, 1_570_000.0, 12_000.0])),
            dip=90.0,
            length=20.0,
            width=width,
            strike_nztm=0.0,
        )

    source_config = realisations.SourceConfig(
        source_geometries={
            "narrow": plane(5_150_000.0, 10.0),
            "wide": plane(5_200_000.0, 20.0),
        }
    )
    magnitudes = realisations.Magnitudes(
        magnitudes={"narrow": BoldM(6.0), "wide": BoldM(7.5)}
    )
    moments = magnitudes.moments
    expected_width = (10.0 * moments["narrow"] + 20.0 * moments["wide"]) / (
        moments["narrow"] + moments["wide"]
    )

    source_parameters = im_calc.calculate_source_parameters(
        source_config,
        magnitudes,
        realisations.Rakes(rakes={"narrow": 90.0, "wide": 90.0}),
        np.array([-43.5, 172.6, 10_000.0]),
    )

    assert source_parameters.avg_width == pytest.approx(expected_width, rel=1e-3)
    assert 19.0 < source_parameters.avg_width < 20.0


def test_empirical_inputs_supply_the_width() -> None:
    # Supplying `width` stops oq_wrapper filling it in with its ASK14
    # magnitude-dip estimate.
    source_parameters = im_calc.SourceParameters(
        mag=7.0,
        avg_rake=90.0,
        avg_dip=45.0,
        avg_ztor=0.0,
        avg_zbot=15.0,
        avg_width=21.2,
        hypo_depth=10.0,
    )
    station = xr.DataArray([1.0], dims="station")
    site_parameters = im_calc.SiteParameters(
        vs30=400.0 * station, z1pt0=0.1 * station, z2pt5=1.0 * station
    )
    distances = im_calc.Distances(
        rrup=10.0 * station, rjb=10.0 * station, hyp=10.0 * station, epi=10.0 * station
    )

    inputs = im_calc.empirical_inputs(source_parameters, site_parameters, distances)

    assert inputs["width"].item() == pytest.approx(21.2)
