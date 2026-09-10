"""Tests for `workflow.sw4`, the shared supergrid geometry.

These are decision tests. The numbers here are the ones the rest of the SW4
pipeline is built on, and the point of pinning them is that a silent change to
any of them puts a source back inside the absorbing layer.
"""

import pytest

from workflow import defaults, sw4
from workflow.realisations import (
    Refinements,
    SW4Command,
    SW4Parameters,
    VelocityModelParameters,
)

DEEPEST_SUPPORTED_DOMAIN_KM = 350.0
"""The deepest domain any realisation can ask for.

One static `fault_buffer` default has to clear the widest sponge any run can
produce, and the sponge is widest on the coarsest grid, which is the one a very
deep domain gets.
"""


def sw4_parameters(**supergrid_parameters: float) -> SW4Parameters:
    """Build minimal SW4 parameters carrying a single `supergrid` command.

    Parameters
    ----------
    **supergrid_parameters : float
        Parameters for the `supergrid` command. Pass none to omit the command
        entirely.

    Returns
    -------
    SW4Parameters
        The parameters.
    """
    commands = [SW4Command("grid", {"proj": "tmerc"})]
    if supergrid_parameters:
        commands.append(SW4Command("supergrid", dict(supergrid_parameters)))
    return SW4Parameters(verbose=2, printcycle=10, nz_min=12, commands=commands)


def test_supergrid_width_from_gridpoints() -> None:
    """`gp=` is a thickness on the coarsest grid, so it scales with resolution."""
    parameters = sw4_parameters(gp=30)
    assert sw4.supergrid_width(parameters, 400.0) == 12000.0
    assert sw4.supergrid_width(parameters, 200.0) == 6000.0


def test_supergrid_width_from_metres() -> None:
    """`width=` is already metres and must not scale with resolution."""
    parameters = sw4_parameters(width=12000.0)
    assert sw4.supergrid_width(parameters, 400.0) == 12000.0
    assert sw4.supergrid_width(parameters, 200.0) == 12000.0


def test_supergrid_width_prefers_width_over_gridpoints() -> None:
    """SW4 rejects both together, but if they appear, `width=` is what it uses."""
    parameters = sw4_parameters(gp=30, width=6000.0)
    assert sw4.supergrid_width(parameters, 400.0) == 6000.0


def test_supergrid_width_falls_back_to_the_sw4_default() -> None:
    """With no `supergrid` command SW4 still builds a sponge, at its own default."""
    parameters = sw4_parameters()
    assert (
        sw4.supergrid_width(parameters, 400.0)
        == sw4.SW4_DEFAULT_SUPERGRID_GRIDPOINTS * 400.0
    )
    # An empty `supergrid` command is the same situation.
    assert sw4.supergrid_width(sw4_parameters(dc=0.02), 400.0) == 12000.0


def test_minimum_fault_buffer_is_additive() -> None:
    """`sponge + 5h`, not `k * sponge`.

    The two terms have different physical origins, so they must not be folded
    into a multiplier: a multiplicative margin collapses to nothing as the grid
    refines, even though the stencil still spans five points.
    """
    parameters = sw4_parameters(gp=30)
    assert sw4.minimum_fault_buffer_m(parameters, 400.0) == 14000.0
    assert sw4.minimum_fault_buffer_m(parameters, 200.0) == 7000.0

    # Stated explicitly, so the additive form cannot be refactored away.
    for resolution in (100.0, 200.0, 400.0):
        assert (
            sw4.minimum_fault_buffer_m(parameters, resolution)
            == (sw4.SW4_DEFAULT_SUPERGRID_GRIDPOINTS + sw4.STENCIL_MARGIN_GRIDPOINTS)
            * resolution
        )


def test_check_fault_buffer_boundary() -> None:
    """14.0 km is exactly enough on a 400 m grid; 13.9 km is not."""
    parameters = sw4_parameters(gp=30)
    sw4.check_fault_buffer(14.0, parameters, 400.0)
    with pytest.raises(ValueError, match="supergrid absorbing layer"):
        sw4.check_fault_buffer(13.9, parameters, 400.0)


def test_check_fault_buffer_message_names_the_remedy() -> None:
    """The error has to say what to change, not just that something is wrong."""
    with pytest.raises(ValueError) as error:
        sw4.check_fault_buffer(2.0, sw4_parameters(gp=30), 400.0)
    message = str(error.value)
    assert "fault_buffer" in message
    assert "14.000 km" in message


def test_coarsest_resolution_is_the_bottom_refinement() -> None:
    """The sponge is measured on SW4's `mGridSize[0]`, the deepest layer."""
    refinements = Refinements.read_from_defaults(defaults.DefaultsVersion.v26_7_1Hz)
    assert sw4.coarsest_resolution(refinements, 3.0) == 100.0
    assert sw4.coarsest_resolution(refinements, 20.0) == 200.0
    assert sw4.coarsest_resolution(refinements, 60.0) == 400.0
    assert sw4.coarsest_resolution(refinements, DEEPEST_SUPPORTED_DOMAIN_KM) == 400.0


def test_default_fault_buffer_is_the_derived_minimum() -> None:
    """The YAML holds the number, Python holds the derivation, this pins them.

    A static YAML cannot hold a derived value, so the only thing keeping
    `v26_7_1Hz`'s `fault_buffer` honest is this test. It is evaluated at the
    deepest supported domain, because one default has to clear the widest
    sponge any run can produce.
    """
    version = defaults.DefaultsVersion.v26_7_1Hz
    sw4_params = SW4Parameters.read_from_defaults(version)
    refinements = Refinements.read_from_defaults(version)
    velocity_model = VelocityModelParameters.read_from_defaults(version)

    coarsest = sw4.coarsest_resolution(refinements, DEEPEST_SUPPORTED_DOMAIN_KM)
    minimum = sw4.minimum_fault_buffer_m(sw4_params, coarsest)

    assert minimum == 14000.0
    assert velocity_model.fault_buffer * 1000.0 == minimum

    # And it must actually pass its own gate at every supported depth.
    for depth in (5.0, 25.0, 60.0, DEEPEST_SUPPORTED_DOMAIN_KM):
        sw4.check_fault_buffer(
            velocity_model.fault_buffer,
            sw4_params,
            sw4.coarsest_resolution(refinements, depth),
        )


def test_root_fault_buffer_is_left_alone() -> None:
    """The EMOD3D-only versions keep 2.0 km for CyberShake reproducibility."""
    for version in defaults.DefaultsVersion:
        if version == defaults.DefaultsVersion.v26_7_1Hz:
            continue
        assert VelocityModelParameters.read_from_defaults(version).fault_buffer == 2.0


def test_check_lateral_gridpoints() -> None:
    """A grid whose two sponges meet has no interior to simulate in."""
    parameters = sw4_parameters(width=12000.0)
    # 100 km domain padded to 124 km: 100 km of interior.
    sw4.check_lateral_gridpoints(124000.0, 124000.0, parameters, 400.0)
    # Exactly 2 * 5 gridpoints of interior is the boundary case.
    sw4.check_lateral_gridpoints(28000.0, 28000.0, parameters, 400.0)
    with pytest.raises(ValueError, match="supergrid sponges"):
        sw4.check_lateral_gridpoints(27999.0, 28000.0, parameters, 400.0)
    with pytest.raises(ValueError, match="supergrid sponges"):
        sw4.check_lateral_gridpoints(28000.0, 27999.0, parameters, 400.0)


def test_absorbed_period() -> None:
    """`T_max = W cos(theta) / (0.431 c)`, pinned against edits to the constant."""
    parameters = sw4_parameters(width=12000.0)
    assert sw4.absorbed_period(parameters, 400.0, 3.5) == pytest.approx(7.96, abs=5e-3)
    assert sw4.absorbed_period(parameters, 400.0, 3.5, 60.0) == pytest.approx(
        3.98, abs=5e-3
    )
    # Grazing incidence destroys absorption; a nominally adequate sponge is not
    # adequate for a wave running along it.
    assert sw4.absorbed_period(parameters, 400.0, 3.5, 85.0) == pytest.approx(
        0.69, abs=5e-3
    )
    assert sw4.absorbed_period(parameters, 400.0, 3.5, 90.0) == pytest.approx(0.0)


def test_adiabatic_coefficient() -> None:
    """`max|Psi0'| / 2pi` with `max|Psi0'| = 2772/1024`."""
    assert sw4.ADIABATIC_COEFFICIENT == pytest.approx(0.4308374, abs=1e-7)


def test_stencil_margin_matches_sw4s_own_margin() -> None:
    """`src_reach(3) + sgd_reach(2)` at 4th order, the order SW4 defaults to.

    This constant is duplicated in SW4's own source-in-sponge check. If one
    moves without the other, one of the two guards becomes wrong.
    """
    assert sw4.STENCIL_MARGIN_GRIDPOINTS == 5
