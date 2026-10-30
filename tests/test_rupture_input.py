"""Tests for `create-rupture-input`, which writes the rupture generator's input."""

import dataclasses
import io
import json
import subprocess
import sys
import tomllib
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from qcore import coordinates
from rupture_generator import config as rupture_config
from rupture_generator.geometry import geometry_from_geojson
from rupture_generator.rupture.generator import first_arrival, generate
from rupture_generator.rupture.source import magnitude_from_moment
from source_modelling import sources
from source_modelling.magnitude_scaling import BoldM
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
    trace: list[tuple[float, float]],
    dip: float,
    dip_dir: float,
    bottom_km: float = 12.0,
) -> sources.Fault:
    """A fault hung from a (latitude, longitude) trace, as nshm2022-to-realisation builds them."""
    return sources.Fault.from_trace_points(
        np.array(trace), 0.0, bottom_km, dip, dip_dir=dip_dir
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
    """Every defaults version's tables pass the generator's strict reader.

    The correlation lengths are the generator's own defaults: Graves & Pitarka's hybrid
    for slip, and Mai & Beroza (2002) everywhere for rake and the perturbation.
    """
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
            },
            **{
                table: getattr(rupture_generation, table)
                for table in ("slip", "rise", "rake", "perturbation", "pulse")
            },
            "propagation": {"type": "sampled", "model": rupture_generation.jump},
        }
    )
    assert decoded.perturbation.blend_sigma == 4.0
    assert decoded.slip.noise == rupture_config.GP_HYBRID
    assert decoded.rake.noise == rupture_config.STATIONARY
    assert decoded.perturbation.noise == rupture_config.STATIONARY


def _write_realisation(
    path: Path,
    faults: dict[str, sources.IsSource],
    tree: dict[str, str | None],
) -> None:
    RealisationMetadata(
        name="test", version="1", tag="test", defaults_version=VERSION
    ).write_to_realisation(path)
    SourceConfig(source_geometries=faults).write_to_realisation(path)
    RupturePropagationConfig(
        rupture_causality_tree=tree, jump_points={}, hypocentre=np.array([0.3, 0.6])
    ).write_to_realisation(path)
    Magnitudes(
        magnitudes={name: BoldM(6.0 + 0.1 * i) for i, name in enumerate(faults)}
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
    faults: dict[str, sources.IsSource] = {"sw": SOUTH_WEST, "ne": NORTH_EAST}
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
        "model": RuptureGenerationConfig.read_from_defaults(VERSION).jump,
    }
    # A hypocentre at 7 km leaves the deep tapers where they are.
    assert "segments" not in document
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


def _point(depth_m: float, dip: float = 60.0) -> sources.Point:
    return sources.Point.from_lat_lon_depth(
        np.array([-42.0, 172.0, depth_m]),
        length_m=3000.0,
        width_m=2000.0,
        strike=40.0,
        dip=dip,
        dip_dir=130.0,
    )


@pytest.mark.parametrize("dip", [60.0, 90.0])
def test_point_source_is_drawn_on_its_patch(dip: float) -> None:
    """A point source becomes the plane it stands for, centred on it."""
    point = _point(5000.0, dip)
    plane = rupture_input.point_plane(point)
    assert plane.length == pytest.approx(3.0)
    assert plane.width == pytest.approx(2.0)
    assert plane.strike == pytest.approx(40.0, abs=0.1)
    assert plane.dip == pytest.approx(dip)
    assert plane.top_m == pytest.approx(point.top_m)
    np.testing.assert_allclose(
        coordinates.wgs_depth_to_nztm(plane.centroid), point.bounds, atol=1.0
    )


def test_shallow_point_source_patch_stays_underground() -> None:
    """A patch that would break the surface is lowered until its top is at it."""
    plane = rupture_input.point_plane(_point(500.0))
    assert plane.top_m == pytest.approx(0.0, abs=1e-6)
    assert plane.width == pytest.approx(2.0)


def test_point_source_realisation_draws(tmp_path: Path) -> None:
    """A point-source realisation writes input the generator draws."""
    realisation_ffp = tmp_path / "realisation.json"
    _write_realisation(realisation_ffp, {"point": _point(8000.0)}, {"point": None})
    rupture_path = tmp_path / "rupture.toml"
    rupture_input.generate_rupture_input(
        realisation_ffp, tmp_path / "faults.geojson", rupture_path
    )
    scenario = rupture_config.build(rupture_config.load(rupture_path))
    drawn = generate(
        scenario.realisation,
        scenario.medium,
        scenario.sources,
        scenario.model,
        seed=scenario.seed,
    )
    assert set(drawn) == {"point"}
    assert "propagation" not in tomllib.loads(rupture_path.read_text())


PROFILE = {
    "type": "profile",
    "depth_km": [5.0, 8.0, 15.0, 20.0],
    "values": [2, 1, 1, 2],
}


@pytest.mark.parametrize(
    ("profile", "depth_km", "expected"),
    [
        (PROFILE, 10.0, None),
        (PROFILE, 15.0, None),
        (PROFILE, 18.0, [5.0, 8.0, 18.0, 23.0]),
        ({"type": "constant", "value": 1.0}, 18.0, None),
        ({"type": "profile", "depth_km": [1.0, 2.0], "values": [1, 2]}, 18.0, None),
    ],
)
def test_deepened_taper(
    profile: dict, depth_km: float, expected: list[float] | None
) -> None:
    """The deep taper starts no shallower than the hypocentre, and keeps its width."""
    deepened = rupture_input.deepened_taper(profile, depth_km)
    if expected is None:
        assert deepened is None
    else:
        assert deepened == profile | {"depth_km": expected}


def test_deep_hypocentre_moves_initial_fault_tapers() -> None:
    """Only the fault the rupture starts on has its deep tapers moved."""
    deep = _fault([(-42.0, 172.0), (-42.05, 172.1)], 90.0, 0.0, bottom_km=30.0)
    rupture_velocity = RuptureVelocity.read_from_defaults(VERSION)
    rupture = rupture_input.build_rupture(
        RuptureGenerationConfig.read_from_defaults(VERSION),
        {"deep": deep, "other": NORTH_EAST},
        RupturePropagationConfig(
            rupture_causality_tree={"deep": None, "other": "deep"},
            jump_points={},
            # 24 km down a vertical fault 30 km wide.
            hypocentre=np.array([0.5, 0.8]),
        ),
        Magnitudes(magnitudes={"deep": BoldM(6.5), "other": BoldM(6.0)}),
        Rakes(rakes={"deep": 90.0, "other": 90.0}),
        VelocityModel1D.read_from_defaults(VERSION),
        rupture_velocity,
        Seeds.random_seeds(),
        Path("faults.geojson"),
    )
    segments = rupture["segments"]
    assert set(segments) == {"deep"}
    assert segments["deep"]["rise"]["factor"]["depth_km"] == pytest.approx(
        [5.0, 8.0, 24.0, 29.0]
    )
    assert segments["deep"]["speed"]["factor"]["depth_km"] == pytest.approx(
        [5.0, 8.0, 24.0, 29.0]
    )
    # The global tables are the defaults, for the other fault.
    assert rupture["rise"]["factor"]["depth_km"] == [5.0, 8.0, 15.0, 20.0]
    assert rupture["speed"]["factor"]["depth_km"] == [5.0, 8.0, 15.0, 20.0]


def test_slip_dependent_rupture_speed_is_refused(tmp_path: Path) -> None:
    """genslip's rvfrac_slip_sig has no counterpart in the generator."""
    rupture_velocity = RuptureVelocity.read_from_defaults(VERSION)
    with pytest.raises(ValueError, match="rvfrac_slip_sig"):
        rupture_input.build_rupture(
            RuptureGenerationConfig.read_from_defaults(VERSION),
            {"a": NORTH_EAST},
            RupturePropagationConfig(
                rupture_causality_tree={"a": None},
                jump_points={},
                hypocentre=np.array([0.5, 0.5]),
            ),
            Magnitudes(magnitudes={"a": BoldM(6.0)}),
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


def _two_fault_rupture(
    rupture_generation: RuptureGenerationConfig,
) -> dict:
    rupture_velocity = RuptureVelocity.read_from_defaults(VERSION)
    return rupture_input.build_rupture(
        rupture_generation,
        {"sw": SOUTH_WEST, "ne": NORTH_EAST},
        RupturePropagationConfig(
            rupture_causality_tree={"sw": None, "ne": "sw"},
            jump_points={},
            hypocentre=np.array([0.3, 0.6]),
        ),
        Magnitudes(magnitudes={"sw": BoldM(6.5), "ne": BoldM(6.0)}),
        Rakes(rakes={"sw": 90.0, "ne": 90.0}),
        VelocityModel1D.read_from_defaults(VERSION),
        rupture_velocity,
        Seeds.random_seeds(),
        Path("faults.geojson"),
    )


@pytest.mark.parametrize("propagation", ["predetermined", "sampled", "likeliest"])
def test_propagation_type_is_written(propagation: str) -> None:
    """Only `predetermined` states the realisation's tree; the others leave it out."""
    rupture_generation = dataclasses.replace(
        RuptureGenerationConfig.read_from_defaults(VERSION), propagation=propagation
    )
    rupture = _two_fault_rupture(rupture_generation)
    expected = {"type": propagation, "model": rupture_generation.jump}
    if propagation == "predetermined":
        expected["parents"] = {"ne": "sw"}
    assert rupture["propagation"] == expected
    # The hypocentre, the generator's root, is the realisation's either way.
    assert rupture["hypocentre"]["segment"] == "sw"
    rupture_config._DECODER.decode(rupture | {"seed": 1})


@pytest.mark.parametrize("propagation", ["predetermined", "sampled", "likeliest"])
def test_single_fault_has_no_propagation(propagation: str) -> None:
    """A single fault has no tree, whatever the propagation type."""
    rupture_generation = dataclasses.replace(
        RuptureGenerationConfig.read_from_defaults(VERSION), propagation=propagation
    )
    assert (
        rupture_input.propagation_table(
            rupture_generation,
            RupturePropagationConfig(
                rupture_causality_tree={"a": None},
                jump_points={},
                hypocentre=np.array([0.5, 0.5]),
            ),
            1,
        )
        is None
    )


@pytest.mark.parametrize("reach", ["sampled", "reference"])
def test_reach_passes_through(reach: str) -> None:
    """The jump table's `reach` reaches the generator's model unchanged."""
    defaults_ = RuptureGenerationConfig.read_from_defaults(VERSION)
    rupture_generation = dataclasses.replace(
        defaults_, jump=defaults_.jump | {"reach": reach}
    )
    rupture = _two_fault_rupture(rupture_generation)
    assert rupture["propagation"]["model"]["reach"] == reach
    decoded = rupture_config._DECODER.decode(rupture | {"seed": 1})
    assert decoded.propagation is not None
    assert decoded.propagation.model.reach == reach


def test_defaults_leave_propagation_and_reach_to_the_generator() -> None:
    """No defaults version sets either key, so each reads as before they existed."""
    for version in defaults.DefaultsVersion:
        rupture_generation = RuptureGenerationConfig.read_from_defaults(version)
        assert rupture_generation.propagation == "predetermined"
        assert "reach" not in rupture_generation.jump
        assert "propagation" not in rupture_generation.to_dict()


def test_old_realisation_reads_as_predetermined(tmp_path: Path) -> None:
    """A section written before `propagation` existed states the realisation's tree."""
    realisation_ffp = tmp_path / "realisation.json"
    _write_realisation(
        realisation_ffp, {"sw": SOUTH_WEST, "ne": NORTH_EAST}, {"sw": None, "ne": "sw"}
    )
    section = json.loads(realisation_ffp.read_text())["rupture_generation"]
    assert "propagation" not in section
    assert "reach" not in section["jump"]

    rupture_path = tmp_path / "rupture.toml"
    rupture_input.generate_rupture_input(
        realisation_ffp, tmp_path / "faults.geojson", rupture_path
    )
    propagation = tomllib.loads(rupture_path.read_text())["propagation"]
    assert propagation["type"] == "predetermined"
    assert propagation["parents"] == {"ne": "sw"}
    assert "reach" not in propagation["model"]


def test_chosen_propagation_round_trips(tmp_path: Path) -> None:
    """A non-default propagation is written to the realisation and read back."""
    realisation_ffp = tmp_path / "realisation.json"
    RealisationMetadata(
        name="test", version="1", tag="test", defaults_version=VERSION
    ).write_to_realisation(realisation_ffp)
    defaults_ = RuptureGenerationConfig.read_from_defaults(VERSION)
    chosen = dataclasses.replace(
        defaults_, propagation="likeliest", jump=defaults_.jump | {"reach": "reference"}
    )
    chosen.write_to_realisation(realisation_ffp)
    section = json.loads(realisation_ffp.read_text())["rupture_generation"]
    assert section["propagation"] == "likeliest"
    assert section["jump"]["reach"] == "reference"
    assert RuptureGenerationConfig.read_from_realisation(realisation_ffp) == chosen


def _write_multi_fault_input(
    tmp_path: Path, propagation: str, reach: str | None = None
) -> Path:
    """Write a two-fault realisation with these keys, and its generator input."""
    realisation_ffp = tmp_path / "realisation.json"
    # The realisation's tree starts on `sw`, which the generator roots its own at.
    _write_realisation(
        realisation_ffp, {"sw": SOUTH_WEST, "ne": NORTH_EAST}, {"sw": None, "ne": "sw"}
    )
    rupture_generation = RuptureGenerationConfig.read_from_realisation(realisation_ffp)
    jump = rupture_generation.jump | ({} if reach is None else {"reach": reach})
    dataclasses.replace(
        rupture_generation, propagation=propagation, jump=jump
    ).write_to_realisation(realisation_ffp)
    rupture_path = tmp_path / "rupture.toml"
    rupture_input.generate_rupture_input(
        realisation_ffp, tmp_path / "faults.geojson", rupture_path
    )
    return rupture_path


@pytest.mark.parametrize("propagation", ["sampled", "likeliest"])
def test_generator_draws_its_own_tree(tmp_path: Path, propagation: str) -> None:
    """`rupture-generator` draws the input end to end, with every fault in the SRF."""
    rupture_path = _write_multi_fault_input(tmp_path, propagation, "reference")
    srf_path = tmp_path / "realisation.srf"
    subprocess.run(
        [sys.executable, "-m", "rupture_generator.cli", rupture_path, srf_path],
        check=True,
    )
    assert srf_path.stat().st_size > 0

    scenario = rupture_config.build(rupture_config.load(rupture_path))
    # Two faults have one tree, rooted at the realisation's hypocentre.
    assert dict(scenario.realisation.tree) == {"sw": None, "ne": "sw"}


def _jumping_with(seed: int) -> Callable[..., Any]:
    """A jump rule drawing from its own generator, in place of the jump stream's."""

    def rule(*args: Any) -> Any:
        return first_arrival(*args[:-1], np.random.default_rng(seed))

    return rule


@pytest.mark.parametrize("reach", ["sampled", "reference"])
def test_reference_reach_starts_children_whatever_the_jump_stream(
    tmp_path: Path, reach: str
) -> None:
    """Under `reach = "reference"` the child's onsets don't depend on the jump's draws.

    Under `"sampled"` they do, so the comparison has something to fix.
    """
    scenario = rupture_config.build(
        rupture_config.load(_write_multi_fault_input(tmp_path, "likeliest", reach))
    )
    onsets = [
        generate(
            scenario.realisation,
            scenario.medium,
            scenario.sources,
            scenario.model,
            seed=scenario.seed,
            jump_rule=_jumping_with(jump_seed),
            jump_propagation_model=scenario.jump_model,
        )["ne"].onset_s
        for jump_seed in range(4)
    ]
    same = [np.array_equal(onsets[0], other) for other in onsets[1:]]
    assert all(same) if reach == "reference" else not all(same)
