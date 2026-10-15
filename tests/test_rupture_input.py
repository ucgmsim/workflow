"""Tests for `create-rupture-input`, which writes the rupture generator's input."""

import dataclasses
import io
import json
import tomllib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from qcore import coordinates
from rupture_generator import config as rupture_config
from rupture_generator.geometry import geometry_from_geojson
from rupture_generator.rupture.generator import generate
from rupture_generator.rupture.source import magnitude_from_moment
from source_modelling import sources
from workflow import defaults
from workflow.realisations import (
    Magnitudes,
    Rakes,
    RealisationMetadata,
    RuptureGenerationConfig,
    RupturePropagationConfig,
    RuptureVelocity,
    Seeds,
    SourceConfig,
    VelocityModel1D,
)
from workflow.scripts import rupture_input

VERSION = defaults.DefaultsVersion.v26_7_1Hz


def _fault(
    trace: list[tuple[float, float]], dip: float, dip_dir: float
) -> sources.Fault:
    """A fault hung from a (latitude, longitude) trace, as nshm2022-to-realisation builds them."""
    return sources.Fault.from_trace_points(
        np.array(trace), 0.0, 12.0, dip, dip_dir=dip_dir
    )


# A bent fault dipping south-west, and one given in the other order and dipping
# north-east, which source_modelling turns round so it dips to the right of its trace.
SOUTH_WEST = _fault([(-42.0, 172.0), (-42.05, 172.1), (-42.08, 172.2)], 60.0, 200.0)
NORTH_EAST = _fault([(-42.1, 172.25), (-42.15, 172.35)], 45.0, 30.0)
VERTICAL = _fault([(-42.0, 172.0), (-42.05, 172.1)], 90.0, 0.0)


@pytest.mark.parametrize(
    "fault", [SOUTH_WEST, NORTH_EAST, VERTICAL], ids=["sw", "ne", "vertical"]
)
def test_section_rebuilds_the_realisation_corners(fault: sources.Fault) -> None:
    """The generator hangs the section where source_modelling hung the fault."""
    feature = rupture_input.fault_section("fault", fault)
    charts = geometry_from_geojson(
        io.StringIO(json.dumps({"type": "FeatureCollection", "features": [feature]})),
        "EPSG:2193",
    )
    nodes = charts["fault"].nodes  # (top/bottom, trace point, east/north/depth) km
    trace = np.vstack([fault.planes[0].bounds[0]] + [p.bounds[1] for p in fault.planes])
    bottom = np.vstack(
        [fault.planes[0].bounds[3]] + [p.bounds[2] for p in fault.planes]
    )
    # The trace, in the same order, is exact. The bottom edge moves by metres, because
    # each plane here turns the dip direction into a bearing at its own start and the
    # generator hangs them all from the first plane's.
    for row, expected, atol_m in zip(nodes, (trace, bottom), (1e-3, 10.0)):
        # bounds are (northing, easting, depth) in metres.
        np.testing.assert_allclose(row[:, 0] * 1000.0, expected[:, 1], atol=atol_m)
        np.testing.assert_allclose(row[:, 1] * 1000.0, expected[:, 0], atol=atol_m)
        np.testing.assert_allclose(row[:, 2] * 1000.0, expected[:, 2], atol=1e-3)


@pytest.mark.parametrize("defaults_version", list(defaults.DefaultsVersion))
def test_defaults_are_rupture_generator_tables(
    defaults_version: defaults.DefaultsVersion,
) -> None:
    """Every defaults version's tables pass the generator's strict reader."""
    rupture_generation = RuptureGenerationConfig.read_from_defaults(defaults_version)
    decoded = rupture_config._DECODER.decode(
        {
            "seed": 1,
            "geometry": {"path": "f.geojson", "crs": "EPSG:2193", "spacing_km": 1.0},
            "hypocentre": {"segment": "a", "strike_fraction": 0.5, "dip_fraction": 0.5},
            "medium": {
                "bottom_depth_km": [1.0],
                "shear_speed_km_s": [3.0],
                "density_g_cm3": [2.7],
            },
            "source": {
                "type": "finite",
                "magnitude": 6.0,
                "rake_deg": 90.0,
                "correlation": rupture_generation.correlation,
            },
            **{
                table: getattr(rupture_generation, table)
                for table in ("slip", "rise", "rake", "perturbation", "pulse")
            },
        }
    )
    assert decoded.perturbation.blend_sigma == 4.0


def _write_realisation(
    path: Path, faults: dict[str, sources.Fault], tree: dict[str, str | None]
) -> None:
    RealisationMetadata(
        name="test", version="1", tag="test", defaults_version=VERSION
    ).write_to_realisation(path)
    SourceConfig(source_geometries=faults).write_to_realisation(path)
    RupturePropagationConfig(
        rupture_causality_tree=tree, jump_points={}, hypocentre=np.array([0.3, 0.6])
    ).write_to_realisation(path)
    Magnitudes(
        magnitudes={name: 6.0 + 0.1 * i for i, name in enumerate(faults)}
    ).write_to_realisation(path)
    Rakes(rakes={name: 90.0 for name in faults}).write_to_realisation(path)
    Seeds.random_seeds().write_to_realisation(path)
    rupture_generation = RuptureGenerationConfig.read_from_defaults(VERSION)
    # Coarse enough to draw in a test.
    dataclasses.replace(rupture_generation, spacing_km=1.0).write_to_realisation(path)


@pytest.mark.parametrize("initial", ["sw", "ne"])
def test_generated_rupture_matches_realisation(tmp_path: Path, initial: str) -> None:
    """The input draws a rupture with the realisation's moments and hypocentre."""
    realisation_ffp = tmp_path / "realisation.json"
    other = "ne" if initial == "sw" else "sw"
    faults = {"sw": SOUTH_WEST, "ne": NORTH_EAST}
    _write_realisation(realisation_ffp, faults, {initial: None, other: initial})
    geometry_path = tmp_path / "rupture" / "faults.geojson"
    rupture_path = tmp_path / "rupture.toml"
    geometry_path.parent.mkdir()

    rupture_input.generate_rupture_input(realisation_ffp, geometry_path, rupture_path)

    document = tomllib.loads(rupture_path.read_text())
    assert document["geometry"]["path"] == "rupture/faults.geojson"
    assert document["propagation"] == {
        "type": "predetermined",
        "parents": {other: initial},
    }
    assert document["seed"] == Seeds.read_from_realisation(realisation_ffp).genslip_seed

    scenario = rupture_config.build(rupture_config.load(rupture_path))
    drawn = generate(
        scenario.realisation,
        scenario.medium,
        scenario.sources,
        scenario.model,
        seed=scenario.seed,
        jump_propagation_model=scenario.jump_model,
    )
    # The generator refuses slip whose moment strays from its source's, so each fault's
    # source carrying the realisation's magnitude is each fault drawn at it.
    magnitudes = Magnitudes.read_from_realisation(realisation_ffp).magnitudes
    assert set(drawn) == set(faults)
    for name, source in scenario.sources.items():
        assert magnitude_from_moment(source.moment_nm) == pytest.approx(
            magnitudes[name]
        )

    # The front leaves from the realisation's hypocentre.
    segment = drawn[initial]
    centres = segment.geometry.centres
    first = np.unravel_index(np.argmin(segment.onset_s), segment.onset_s.shape)
    hypocentre = coordinates.wgs_depth_to_nztm(
        faults[initial].fault_coordinates_to_wgs_depth_coordinates(np.array([0.3, 0.6]))
    )
    east, north, depth = centres[first] * 1000.0
    assert np.hypot(east - hypocentre[1], north - hypocentre[0]) < 1000.0
    assert depth == pytest.approx(hypocentre[2], abs=1000.0)


def test_point_source_is_refused(tmp_path: Path) -> None:
    """A point source has no finite fault for the generator to draw."""
    point = sources.Point.from_lat_lon_depth(
        np.array([-42.0, 172.0, 5000.0]),
        length_m=1000.0,
        width_m=1000.0,
        strike=0.0,
        dip=90.0,
        dip_dir=0.0,
    )
    with pytest.raises(TypeError, match="realisation-to-srf"):
        rupture_input.build_fault_system(SourceConfig(source_geometries={"p": point}))


def test_slip_dependent_rupture_speed_is_refused(tmp_path: Path) -> None:
    """genslip's rvfrac_slip_sig has no counterpart in the generator."""
    rupture_velocity = RuptureVelocity.read_from_defaults(VERSION)
    with pytest.raises(ValueError, match="rvfrac_slip_sig"):
        rupture_input.build_rupture(
            RuptureGenerationConfig.read_from_defaults(VERSION),
            RupturePropagationConfig(
                rupture_causality_tree={"a": None},
                jump_points={},
                hypocentre=np.array([0.5, 0.5]),
            ),
            Magnitudes(magnitudes={"a": 6.0}),
            Rakes(rakes={"a": 90.0}),
            VelocityModel1D(
                model=pd.DataFrame(
                    {
                        "thickness": [1.0],
                        "Vp": [5.0],
                        "Vs": [3.0],
                        "rho": [2.7],
                        "Qp": [100.0],
                        "Qs": [50.0],
                    }
                )
            ),
            dataclasses.replace(rupture_velocity, rvfrac_slip_sig=0.1),
            Seeds.random_seeds(),
            Path("faults.geojson"),
        )
