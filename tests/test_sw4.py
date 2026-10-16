"""Tests for `workflow.sw4`."""

import numpy as np
import pytest

from workflow import defaults, sw4
from workflow.realisations import (
    BroadbandParameters,
    Refinements,
    SW4Command,
    SW4Parameters,
    VelocityModelParameters,
    find_command,
)

DEEPEST_SUPPORTED_DOMAIN_KM = 350.0
"""The deepest domain any realisation can ask for."""

SW4_VERSIONS = [
    defaults.DefaultsVersion.v26_7_0_25Hz,
    defaults.DefaultsVersion.v26_7_0_5Hz,
    defaults.DefaultsVersion.v26_7_1Hz,
]
"""The defaults versions configured for SW4."""


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
    "resolution, expected", [(100.0, 3500.0), (200.0, 7000.0), (400.0, 14000.0)]
)
def test_minimum_fault_buffer_is_additive(resolution: float, expected: float) -> None:
    """The minimum buffer is `sponge + 5h`."""
    assert sw4.minimum_fault_buffer_m(sw4_parameters(gp=30), resolution) == expected


def test_check_fault_buffer_boundary() -> None:
    parameters = sw4_parameters(gp=30)
    sw4.check_fault_buffer(14.0, parameters, 400.0)
    with pytest.raises(ValueError, match=r"fault_buffer.*14\.000 km"):
        sw4.check_fault_buffer(13.9, parameters, 400.0)


def test_coarsest_resolution_is_the_bottom_refinement() -> None:
    refinements = Refinements.read_from_defaults(defaults.DefaultsVersion.v26_7_1Hz)
    assert sw4.coarsest_resolution(refinements, 3.0) == 100.0
    assert sw4.coarsest_resolution(refinements, 20.0) == 200.0
    assert sw4.coarsest_resolution(refinements, 60.0) == 400.0
    assert sw4.coarsest_resolution(refinements, DEEPEST_SUPPORTED_DOMAIN_KM) == 400.0


def test_lower_frequency_coarsest_resolution_matches_1hz() -> None:
    """The lower-frequency versions share 26.7.1Hz's coarsest grid, so a domain
    sized for one clears the sponge in all of them."""
    for version in SW4_VERSIONS:
        refinements = Refinements.read_from_defaults(version)
        assert (
            sw4.coarsest_resolution(refinements, DEEPEST_SUPPORTED_DOMAIN_KM) == 400.0
        )
    lower = Refinements.read_from_defaults(defaults.DefaultsVersion.v26_7_0_25Hz)
    assert sw4.coarsest_resolution(lower, 3.0) == 400.0
    half = Refinements.read_from_defaults(defaults.DefaultsVersion.v26_7_0_5Hz)
    assert sw4.coarsest_resolution(half, 3.0) == 200.0


TARGET_POINTS_PER_WAVELENGTH = 8.0
"""The points per wavelength SW4 keeps below the nzcvm Vs floor."""

SURFACE_GRID_FREQUENCY_PRODUCT = 100.0
"""Grid spacing times frequency (m Hz) above the Vs floor: 26.7.1Hz's 100 m
surface grid, scaled by 1 / f for the lower-frequency versions."""


def vs_floor(version: defaults.DefaultsVersion) -> tuple[float, float]:
    """Read the nzcvm Vs floor the refinements are built around.

    Parameters
    ----------
    version : DefaultsVersion
        The defaults version.

    Returns
    -------
    elevation : float
        Depth at and below which the floor applies, in metres (positive down).
    vs_min : float
        The floor's minimum Vs, in m/s.
    """
    (clamp,) = (
        layer
        for layer in defaults.load_defaults(version)["nzcvm"]["layers"]
        if layer["type"] == "clamp"
    )
    (floor,) = clamp["clamps"]["vs"]["floors"]
    return floor["elevation"], floor["min"]


def spacing_at(refinements: Refinements, depth_m: float) -> float:
    """Find the grid spacing at a depth in the deepest supported domain.

    Parameters
    ----------
    refinements : Refinements
        The mesh refinements.
    depth_m : float
        The depth, in metres.

    Returns
    -------
    float
        The grid spacing of the layer containing `depth_m`, in metres.
    """
    return next(
        refinement.resolution
        for refinement in refinements.refinements_for_depth(DEEPEST_SUPPORTED_DOMAIN_KM)
        if refinement.bottom > depth_m
    )


@pytest.mark.parametrize("version", SW4_VERSIONS)
def test_refinements_coarsen_at_the_vs_floor(
    version: defaults.DefaultsVersion,
) -> None:
    """Every version coarsens at the nzcvm Vs floor, by one rule.

    With n points per wavelength a grid of spacing h resolves f where
    Vs >= n h f. Below the floor Vs is at least the floor's minimum, so the
    grid there is the coarsest that keeps 8 points per wavelength (capped at
    the shared coarsest grid, which the sponge and `fault_buffer` are sized
    for). Above it the grid is 26.7.1Hz's 100 m scaled by 1 / f. A version
    that refines deeper than the floor pays for grid points it does not need.
    """
    refinements = Refinements.read_from_defaults(version)
    flo = BroadbandParameters.read_from_defaults(version).flo
    elevation, vs_min = vs_floor(version)
    coarsest = refinements.unbounded_refinement_resolution

    above = spacing_at(refinements, elevation - 1.0)
    below = spacing_at(refinements, elevation)

    assert above == min(SURFACE_GRID_FREQUENCY_PRODUCT / flo, coarsest)
    assert below == min(vs_min / (TARGET_POINTS_PER_WAVELENGTH * flo), coarsest)
    assert vs_min / (below * flo) >= TARGET_POINTS_PER_WAVELENGTH


@pytest.mark.parametrize("version", SW4_VERSIONS)
def test_default_fault_buffer_is_the_derived_minimum(
    version: defaults.DefaultsVersion,
) -> None:
    """The default `fault_buffer` matches the minimum for the deepest domain."""
    sw4_params = SW4Parameters.read_from_defaults(version)
    refinements = Refinements.read_from_defaults(version)
    velocity_model = VelocityModelParameters.read_from_defaults(version)

    coarsest = sw4.coarsest_resolution(refinements, DEEPEST_SUPPORTED_DOMAIN_KM)
    assert velocity_model.fault_buffer * 1000.0 == sw4.minimum_fault_buffer_m(
        sw4_params, coarsest
    )


@pytest.mark.parametrize(
    "version",
    [version for version in defaults.DefaultsVersion if version not in SW4_VERSIONS],
)
def test_root_fault_buffer_is_left_alone(version: defaults.DefaultsVersion) -> None:
    assert VelocityModelParameters.read_from_defaults(version).fault_buffer == 2.0


@pytest.mark.parametrize("version", SW4_VERSIONS)
def test_default_prefilter_matches_hf_highpass(
    version: defaults.DefaultsVersion,
) -> None:
    """The LF prefilter and the HF highpass are a matched pair at `flo`."""
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


@pytest.mark.parametrize("version", SW4_VERSIONS)
def test_default_topography_mapping_is_fourth_order(
    version: defaults.DefaultsVersion,
) -> None:
    """The curvilinear mapping keeps SW4's 4th-order accuracy at the zeta break.

    SW4's mapping is only C^(order-1) at the break, so convergence is about
    min(order, 4): order 3 loses an order of accuracy wherever topography
    is non-flat.
    """
    topography = find_command(
        SW4Parameters.read_from_defaults(version).commands, "topography"
    )
    assert topography is not None
    assert topography.parameters["order"] >= 4


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
