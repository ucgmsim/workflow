"""Tests for `workflow.sw4`."""

import pytest

from workflow import defaults, sw4
from workflow.realisations import (
    Refinements,
    SW4Command,
    SW4Parameters,
    VelocityModelParameters,
)

DEEPEST_SUPPORTED_DOMAIN_KM = 350.0
"""The deepest domain any realisation can ask for."""


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


def test_default_fault_buffer_is_the_derived_minimum() -> None:
    """The default `fault_buffer` matches the minimum for the deepest domain."""
    version = defaults.DefaultsVersion.v26_7_1Hz
    sw4_params = SW4Parameters.read_from_defaults(version)
    refinements = Refinements.read_from_defaults(version)
    velocity_model = VelocityModelParameters.read_from_defaults(version)

    coarsest = sw4.coarsest_resolution(refinements, DEEPEST_SUPPORTED_DOMAIN_KM)
    assert velocity_model.fault_buffer * 1000.0 == sw4.minimum_fault_buffer_m(
        sw4_params, coarsest
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
