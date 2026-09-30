"""Tests for `workflow.sw4`."""

import numpy as np
import pytest

from workflow import defaults, sw4
from workflow.realisations import (
    BroadbandParameters,
    Refinement,
    Refinements,
    SW4Command,
    SW4Parameters,
    SW4Resolution,
    VelocityModelParameters,
    find_command,
)


def sw4_parameters(**supergrid_parameters: float) -> SW4Parameters:
    """Build minimal SW4 parameters, with a `supergrid` command if given parameters."""
    commands = [SW4Command("grid", {"proj": "tmerc"})]
    if supergrid_parameters:
        commands.append(SW4Command("supergrid", dict(supergrid_parameters)))
    return SW4Parameters(verbose=2, printcycle=10, nz_min=12, commands=commands)


@pytest.mark.parametrize(
    "supergrid, resolution, expected",
    [
        pytest.param({"gp": 30}, 400.0, 12000.0, id="gp-scales-with-resolution"),
        pytest.param({"gp": 30}, 200.0, 6000.0, id="gp-scales-with-resolution-fine"),
        pytest.param({"width": 12000.0}, 400.0, 12000.0, id="width-is-fixed"),
        pytest.param({"width": 12000.0}, 200.0, 12000.0, id="width-is-fixed-fine"),
        pytest.param({"gp": 30, "width": 6000.0}, 400.0, 6000.0, id="width-over-gp"),
        pytest.param({}, 400.0, 12000.0, id="no-supergrid-command"),
        pytest.param({"dc": 0.02}, 400.0, 12000.0, id="sw4-default-gp"),
    ],
)
def test_supergrid_width(
    supergrid: dict[str, float], resolution: float, expected: float
) -> None:
    assert sw4.supergrid_width(sw4_parameters(**supergrid), resolution) == expected


@pytest.mark.parametrize(
    "resolution, expected", [(100.0, 500.0), (200.0, 1000.0), (400.0, 2000.0)]
)
def test_minimum_fault_buffer_is_the_stencil_margin(
    resolution: float, expected: float
) -> None:
    """The sponge sits outside the domain, so only `5h` of clearance is needed."""
    assert sw4.minimum_fault_buffer_m(resolution) == expected


def test_check_fault_buffer_boundary() -> None:
    parameters = sw4_parameters(gp=30)
    sw4.check_fault_buffer(2.0, parameters, 400.0)
    with pytest.raises(ValueError, match=r"fault_buffer.*2\.000 km"):
        sw4.check_fault_buffer(1.9, parameters, 400.0)


def test_default_fault_buffer_is_the_derived_minimum() -> None:
    """The default `fault_buffer` matches the minimum for the coarsest SW4 grid."""
    version = defaults.DefaultsVersion.v26_7_1Hz
    resolution = SW4Resolution.read_from_defaults(version)
    velocity_model = VelocityModelParameters.read_from_defaults(version)

    assert velocity_model.fault_buffer * 1000.0 == sw4.minimum_fault_buffer_m(
        resolution.coarsest_resolution
    )


@pytest.mark.parametrize(
    "version",
    [
        version
        for version in defaults.DefaultsVersion
        if version != defaults.DefaultsVersion.v26_7_1Hz
    ],
)
def test_root_fault_buffer_is_left_alone(version: defaults.DefaultsVersion) -> None:
    assert VelocityModelParameters.read_from_defaults(version).fault_buffer == 2.0


def test_default_prefilter_matches_hf_highpass() -> None:
    """The LF prefilter and the HF highpass are a matched pair at `flo`."""
    version = defaults.DefaultsVersion.v26_7_1Hz
    prefilter = find_command(
        SW4Parameters.read_from_defaults(version).commands, "prefilter"
    )
    assert prefilter is not None
    flo = BroadbandParameters.read_from_defaults(version).flo

    assert prefilter.parameters["order"] == 4
    assert prefilter.parameters["passes"] == 2
    assert prefilter.parameters["fc2"] == pytest.approx(
        flo / (np.sqrt(2) - 1) ** (1 / 8), rel=1e-5
    )


def test_check_lateral_gridpoints() -> None:
    parameters = sw4_parameters(width=12000.0)
    # 100 km domain padded to 124 km: 100 km of interior.
    sw4.check_lateral_gridpoints(124000.0, 124000.0, parameters, 400.0)
    # Boundary case: exactly 2 * 5 gridpoints of interior.
    sw4.check_lateral_gridpoints(28000.0, 28000.0, parameters, 400.0)
    with pytest.raises(ValueError, match="supergrid sponges"):
        sw4.check_lateral_gridpoints(27999.0, 28000.0, parameters, 400.0)
    with pytest.raises(ValueError, match="supergrid sponges"):
        sw4.check_lateral_gridpoints(28000.0, 27999.0, parameters, 400.0)


def test_absorbed_period() -> None:
    parameters = sw4_parameters(width=12000.0)
    assert sw4.absorbed_period(parameters, 400.0, 3.5) == pytest.approx(7.96, abs=5e-3)
    assert sw4.absorbed_period(parameters, 400.0, 3.5, 60.0) == pytest.approx(
        3.98, abs=5e-3
    )
    assert sw4.absorbed_period(parameters, 400.0, 3.5, 85.0) == pytest.approx(
        0.69, abs=5e-3
    )
    assert sw4.absorbed_period(parameters, 400.0, 3.5, 90.0) == pytest.approx(0.0)


@pytest.mark.parametrize(
    "elevation_min, elevation_max, expected",
    [
        pytest.param(0.0, 1800.0, 5400.0, id="coastal"),
        pytest.param(200.0, 2000.0, 5200.0, id="inland-needs-less"),
        pytest.param(-500.0, 2000.0, 8000.0, id="bathymetry-needs-more"),
    ],
)
def test_topography_zmax(
    elevation_min: float, elevation_max: float, expected: float
) -> None:
    """`zmax = tau_max + 3 (tau_max - tau_min)`, from the SW4 User Guide."""
    assert sw4.topography_zmax(elevation_min, elevation_max) == expected


def test_vs_profile_maps_curvilinear_depths_to_reference_depths() -> None:
    """A 1500 m peak over a 4500 m `zmax` stretches its column by 4 / 3."""
    profile = sw4.vs_profile(
        z=np.array([-1500.0, 1500.0, 6000.0]),
        tau=-1500.0,
        vs=np.array([1000.0, 2000.0, 3000.0]),
        vp=np.array([2000.0, 4000.0, 6000.0]),
        topography_zmax=4500.0,
        bin_size=1000.0,
        n_bins=8,
    )

    # The surface is at reference depth 0, and 1500 m (3000 m below the peak)
    # at 3000 * 3 / 4 = 2250 m. Below `zmax` the grid is Cartesian.
    assert np.flatnonzero(np.isfinite(profile.min_vs)).tolist() == [0, 2, 6]
    assert profile.min_vs[[0, 2, 6]] == pytest.approx([750.0, 1500.0, 3000.0])
    # Stretched cells are taller, never narrower, so the time step is unchanged.
    assert profile.max_wave_speed[2] == pytest.approx(
        np.sqrt(4000.0**2 + 2 * 2000.0**2)
    )


def uniform_profile(min_vs: list[float], bin_size: float = 100.0) -> sw4.VsProfile:
    """Build a profile with these minimum speeds and a fixed maximum."""
    return sw4.VsProfile(
        bin_size=bin_size,
        min_vs=np.array(min_vs),
        max_wave_speed=np.full(len(min_vs), 6000.0),
    )


RESOLUTION = SW4Resolution(
    finest_resolution=50.0,
    coarsest_resolution=200.0,
    minimum_ppw=8.0,
    max_frequency=2.0,
)
"""At 2 Hz and 8 points per wavelength, 100 m needs 1600 m/s and 200 m 3200 m/s."""


def test_size_refinements_waits_for_the_slowest_material_below() -> None:
    """A slow layer under a fast one holds the finer grid down through both.

    The 100 m layer waits for the 1000 m/s bin at 1000 m, and the 200 m layer
    starts at 2000 m but is pushed to 2400 m to hold 12 cells of 100 m.
    """
    profile = uniform_profile(
        [500.0] * 5 + [2000.0] * 5 + [1000.0] + [2000.0] * 9 + [4000.0] * 20
    )

    refinements = sw4.size_refinements(profile, RESOLUTION, depth_m=30_000.0, nz_min=12)

    assert refinements == [
        Refinement(resolution=50.0, bottom=1100.0),
        Refinement(resolution=100.0, bottom=2400.0),
        Refinement(resolution=200.0, bottom=30_000.0),
    ]


def test_size_refinements_stops_at_the_domain_bottom() -> None:
    """Layers that would start below the domain are dropped."""
    profile = uniform_profile([500.0] * 10 + [2000.0] * 90)

    refinements = sw4.size_refinements(profile, RESOLUTION, depth_m=5_000.0, nz_min=12)

    assert refinements == [
        Refinement(resolution=50.0, bottom=1000.0),
        Refinement(resolution=100.0, bottom=5_000.0),
    ]


def test_layer_ppw_and_time_steps() -> None:
    profile = uniform_profile([500.0] * 10 + [4000.0] * 10)
    refinements = [
        Refinement(resolution=50.0, bottom=1000.0),
        Refinement(resolution=200.0, bottom=2000.0),
    ]

    assert sw4.layer_ppw(profile, refinements, 2.0) == pytest.approx([5.0, 10.0])
    assert sw4.layer_time_steps(profile, refinements, cfl=1.2) == pytest.approx(
        [0.01, 0.04]
    )


def test_resolution_must_halve_to_the_coarsest() -> None:
    with pytest.raises(ValueError, match="power-of-two"):
        SW4Resolution(
            finest_resolution=50.0,
            coarsest_resolution=300.0,
            minimum_ppw=8.0,
            max_frequency=2.0,
        )
    assert RESOLUTION.resolutions == [50.0, 100.0, 200.0]


def test_default_resolution_matches_the_velocity_model() -> None:
    """SW4's ladder is the model's, at the frequency the broadband merges at."""
    version = defaults.DefaultsVersion.v26_7_1Hz
    resolution = SW4Resolution.read_from_defaults(version)
    refinements = Refinements.read_from_defaults(version)

    assert resolution.finest_resolution == refinements.refinements[0].resolution
    assert resolution.coarsest_resolution == refinements.unbounded_refinement_resolution
    assert (
        resolution.max_frequency == BroadbandParameters.read_from_defaults(version).flo
    )


def test_empty_bins_take_the_material_above_them() -> None:
    """A sparsely sampled slow layer still holds the finer grid down."""
    nan = float("nan")
    # One slow sample at 1000 m, then nothing until fast material at 2000 m.
    profile = uniform_profile([nan] * 10 + [500.0] + [nan] * 9 + [4000.0] * 20)

    refinements = sw4.size_refinements(profile, RESOLUTION, depth_m=30_000.0, nz_min=12)

    assert refinements[0] == Refinement(resolution=50.0, bottom=2000.0)
