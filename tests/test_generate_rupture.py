"""Tests for `generate-rupture`, which draws a rupture in the NZCVM velocity model."""

from pathlib import Path

import numpy as np
import shapely
from nzcvm.config.layers.backus import BackusAveragedLayerConfig
from test_rupture_input import _fault, _write_realisation

from qcore import coordinates
from source_modelling import srf
from workflow import velocity_model
from workflow.realisations import NZCVMSettings
from workflow.scripts import generate_rupture, rupture_input


def test_medium_converts_units_and_reads_once() -> None:
    """Shear speed and density at the same positions cost one query, in km/s and
    g/cm^3, with positions passed on as NZTM metres."""
    calls = []

    def query(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> dict[str, np.ndarray]:
        calls.append((x, y, z))
        return {"vs": np.full(x.shape, 3500.0), "rho": np.full(x.shape, 2700.0)}

    medium = generate_rupture.nzcvm_medium(query)
    positions = np.array([[[1570.0, 5180.0, 10.0]]])
    np.testing.assert_allclose(medium.rigidity_pa(positions), 2.7 * 3.5**2 * 1e9)
    assert len(calls) == 1
    np.testing.assert_allclose(np.stack(calls[0], axis=-1), positions * 1000.0)
    medium.shear_speed_km_s(positions + 1.0)
    assert len(calls) == 2


def test_srf_carries_the_velocity_model(
    tmp_path: Path, nzcvm_settings: NZCVMSettings
) -> None:
    """Every SRF point has the shear speed and density NZCVM gives at it."""
    realisation_ffp = tmp_path / "realisation.json"
    # Two faults in NZCVM's synthetic world, so the rupture jumps between them.
    _write_realisation(
        realisation_ffp,
        {
            "sw": _fault([(-43.50, 172.20), (-43.55, 172.30)], 60.0, 200.0),
            "ne": _fault([(-43.58, 172.35), (-43.63, 172.45)], 45.0, 30.0),
        },
        {"sw": None, "ne": "sw"},
    )
    nzcvm_settings.write_to_realisation(realisation_ffp)
    rupture_ffp = tmp_path / "rupture.toml"
    rupture_input.generate_rupture_input(
        realisation_ffp, tmp_path / "faults.geojson", rupture_ffp
    )
    srf_ffp = tmp_path / "realisation.srf"
    generate_rupture.generate_rupture(realisation_ffp, rupture_ffp, srf_ffp)

    points = srf.read_srf(srf_ffp).points
    northing, easting, depth = coordinates.wgs_depth_to_nztm(
        points[["lat", "lon", "dep"]].to_numpy() * [1.0, 1.0, 1000.0]
    ).T
    expected = velocity_model.point_query(
        nzcvm_settings,
        shapely.MultiPoint(np.c_[easting, northing]).convex_hull.buffer(1000.0),
        exclude=(BackusAveragedLayerConfig,),
    )(easting, northing, depth)
    # The SRF holds cm/s and g/cm^3, to a few significant figures.
    np.testing.assert_allclose(points["vs"] / 100.0, expected["vs"], rtol=1e-3)
    np.testing.assert_allclose(points["den"], expected["rho"] / 1000.0, rtol=1e-3)
    assert points["vs"].nunique() > 1
