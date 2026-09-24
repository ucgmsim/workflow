"""SW4 grid geometry and checks.

Covers the refined grid and the supergrid (absorbing sponge). Inside the sponge
SW4 solves a damped equation, so sources and receivers there do not produce
valid ground motion.
"""

import math

from workflow.realisations import (
    DomainParameters,
    Refinements,
    SW4Parameters,
    find_command,
)

SW4_DEFAULT_SUPERGRID_GRIDPOINTS = 30
"""SW4's default supergrid thickness, in grid points (`sw4/src/EW.C`)."""

STENCIL_MARGIN_GRIDPOINTS = 5
"""Grid points of clearance required between a source and the sponge.

`src_reach(3) + sgd_reach(2)` at 4th order. Must match `margin_pts` in SW4's
source check.
"""

ADIABATIC_COEFFICIENT = (2772.0 / 1024.0) / (2.0 * math.pi)
"""`max|Psi0'| / (2 pi)` for SW4's supergrid stretching function."""

SUPERGRID_WIDTH_ATTRIBUTES = {
    "SGWIDTH": "supergrid_width",
    "SGWIDTHGP": "supergrid_width_gp",
}
"""Map from SW4's supergrid-width datasets to the attribute names `lf-to-xarray`
writes and `im-calc` copies to the IM root attrs."""

SUPERGRID_DEPTH_COORDINATES = {
    "SGDEPTH": "supergrid_depth",
    "SGDEPTHGP": "supergrid_depth_gp",
}
"""Map from SW4's per-station supergrid-depth datasets (metres and grid points)
to the station coordinates `lf-to-xarray` writes and `im-calc` reads."""


def supergrid_width(sw4_params: SW4Parameters, coarsest_resolution: float) -> float:
    """Compute the supergrid sponge width SW4 will use, in metres.

    `width=` takes precedence over `gp=`, as in SW4. `gp=` is measured on the
    coarsest grid.

    Parameters
    ----------
    sw4_params : SW4Parameters
        The SW4 parameters.
    coarsest_resolution : float
        The coarsest grid spacing in the run, in metres.

    Returns
    -------
    float
        The sponge width, in metres.
    """
    command = find_command(sw4_params.commands, "supergrid")
    parameters = command.parameters if command is not None else {}

    width = parameters.get("width")
    if width is not None:
        return float(width)

    gridpoints = parameters.get("gp")
    if gridpoints is None:
        gridpoints = SW4_DEFAULT_SUPERGRID_GRIDPOINTS

    return float(gridpoints) * coarsest_resolution


def coarsest_resolution(refinements: Refinements, depth_km: float) -> float:
    """Find the coarsest grid spacing SW4 will use for a domain, in metres.

    Parameters
    ----------
    refinements : Refinements
        The mesh refinements.
    depth_km : float
        The domain depth, in kilometres.

    Returns
    -------
    float
        The coarsest grid spacing, in metres.
    """
    return max(
        refinement.resolution
        for refinement in refinements.refinements_for_depth(depth_km)
    )


def gridpoints_from_domain(
    domain_parameters: DomainParameters, refinements: Refinements
) -> int:
    """Estimate the number of grid points in a refined domain.

    Parameters
    ----------
    domain_parameters : DomainParameters
        The domain to estimate for.
    refinements : Refinements
        The mesh refinements.

    Returns
    -------
    int
        The approximate number of grid points.
    """
    depth = domain_parameters.depth
    area = domain_parameters.domain.area * (1000**2)
    domain_refinements = refinements.refinements_for_depth(depth)
    top = 0.0
    gridpoints = 0
    for refinement in domain_refinements:
        volume = (refinement.bottom - top) * area
        gridpoints += int(volume // (refinement.resolution) ** 3)
        top = refinement.bottom
    return gridpoints


def minimum_fault_buffer_m(
    sw4_params: SW4Parameters, coarsest_resolution: float
) -> float:
    """Compute the smallest fault buffer that clears the supergrid sponge.

    This is the sponge width plus `STENCIL_MARGIN_GRIDPOINTS` grid points.

    Parameters
    ----------
    sw4_params : SW4Parameters
        The SW4 parameters.
    coarsest_resolution : float
        The coarsest grid spacing in the run, in metres.

    Returns
    -------
    float
        The minimum fault buffer, in metres.
    """
    return (
        supergrid_width(sw4_params, coarsest_resolution)
        + STENCIL_MARGIN_GRIDPOINTS * coarsest_resolution
    )


def check_fault_buffer(
    fault_buffer_km: float, sw4_params: SW4Parameters, coarsest_resolution: float
) -> None:
    """Check that a fault buffer keeps every source clear of the supergrid sponge.

    Parameters
    ----------
    fault_buffer_km : float
        The `velocity_model.fault_buffer` value, in kilometres.
    sw4_params : SW4Parameters
        The SW4 parameters.
    coarsest_resolution : float
        The coarsest grid spacing in the run, in metres.

    Raises
    ------
    ValueError
        If the buffer is smaller than `minimum_fault_buffer_m`.
    """
    minimum = minimum_fault_buffer_m(sw4_params, coarsest_resolution)
    if fault_buffer_km * 1000.0 < minimum:
        sponge = supergrid_width(sw4_params, coarsest_resolution)
        raise ValueError(
            f"The fault buffer of {fault_buffer_km:.3f} km is smaller than the "
            f"{minimum / 1000.0:.3f} km needed to keep sources out of the SW4 "
            f"supergrid absorbing layer. On a {coarsest_resolution:.0f} m "
            f"coarsest grid the sponge is {sponge / 1000.0:.3f} km wide, and a "
            f"source needs a further {STENCIL_MARGIN_GRIDPOINTS} grid points "
            f"({STENCIL_MARGIN_GRIDPOINTS * coarsest_resolution / 1000.0:.3f} km) "
            "of clearance for its own stencil and the dissipation operator. "
            "Inside the layer SW4 solves a damped, coordinate-stretched "
            "equation, so the result is not a ground motion. Raise "
            f"velocity_model.fault_buffer to at least {minimum / 1000.0:.3f} km, "
            "or narrow the supergrid."
        )


def check_lateral_gridpoints(
    x_m: float, y_m: float, sw4_params: SW4Parameters, coarsest_resolution: float
) -> None:
    """Check that a SW4 grid has a usable interior between its lateral sponges.

    Parameters
    ----------
    x_m, y_m : float
        The lateral extents of the SW4 grid, in metres, in SW4's own axis
        convention (`x` is north).
    sw4_params : SW4Parameters
        The SW4 parameters.
    coarsest_resolution : float
        The coarsest grid spacing in the run, in metres.

    Raises
    ------
    ValueError
        If either axis's interior is narrower than twice the stencil margin.
    """
    sponge = supergrid_width(sw4_params, coarsest_resolution)
    minimum_interior = 2 * STENCIL_MARGIN_GRIDPOINTS * coarsest_resolution

    for axis, extent in (("x", x_m), ("y", y_m)):
        interior = extent - 2 * sponge
        if interior < minimum_interior:
            raise ValueError(
                f"The SW4 grid's {axis} extent of {extent / 1000.0:.3f} km leaves "
                f"only {interior / 1000.0:.3f} km between its two "
                f"{sponge / 1000.0:.3f} km supergrid sponges, but at least "
                f"{minimum_interior / 1000.0:.3f} km "
                f"({2 * STENCIL_MARGIN_GRIDPOINTS} grid points on a "
                f"{coarsest_resolution:.0f} m grid) is needed for a usable "
                "interior. Widen the domain or narrow the supergrid."
            )


def absorbed_period(
    sw4_params: SW4Parameters,
    coarsest_resolution: float,
    vs_km_s: float,
    incidence_degrees: float = 0.0,
) -> float:
    """Compute the longest period the supergrid sponge can absorb, in seconds.

    This is `W cos(theta) / (ADIABATIC_COEFFICIENT * c)`.

    Parameters
    ----------
    sw4_params : SW4Parameters
        The SW4 parameters.
    coarsest_resolution : float
        The coarsest grid spacing in the run, in metres.
    vs_km_s : float
        The shear wave speed at the layer, in km/s.
    incidence_degrees : float, default 0.0
        The angle between the ray and the layer normal, in degrees.

    Returns
    -------
    float
        The longest absorbable period, in seconds.
    """
    width = supergrid_width(sw4_params, coarsest_resolution)
    speed = vs_km_s * 1000.0
    return (
        width
        * math.cos(math.radians(incidence_degrees))
        / (ADIABATIC_COEFFICIENT * speed)
    )
