"""SW4 supergrid geometry.

SW4 surrounds the simulation domain with a *supergrid* absorbing layer (a
"sponge"), inside which it deliberately solves a damped, coordinate-stretched
equation rather than the wave equation. Anything inside that layer — a source, a
receiver — is not a ground motion prediction.

This module is the single shared definition of the sponge's geometry for the
workflow. It lives here rather than in `workflow.domain` because
`workflow.domain`'s one function is EMOD3D-specific, and because three separate
consumers need these numbers:

- `workflow.scripts.sw4_template` pads the SW4 grid laterally and vertically by
  the sponge width, so the requested domain becomes the grid's *interior*;
- `workflow.scripts.nzvm_input_template` pads the *velocity model* by at least
  as much again, so SW4 never queries outside the sfile;
- `workflow.scripts.generate_domain` checks the fault buffer against the sponge
  before a domain is ever written.

Validation lives here and not in `workflow.schemas`: the module docstring of
`workflow.realisations` reserves the schemas for loose, field-level validation,
and these checks are cross-field and depend on the resolved refinements.
"""

import math

from workflow.realisations import Refinements, SW4Parameters, find_command

SW4_DEFAULT_SUPERGRID_GRIDPOINTS = 30
"""SW4's own default supergrid thickness, in grid points (`sw4/src/EW.C`).

Used when the realisation's SW4 commands set neither `gp=` nor `width=` on the
`supergrid` command, because that is what SW4 itself would then use.
"""

STENCIL_MARGIN_GRIDPOINTS = 5
"""Grid points of clearance required between a source and the sponge.

A source must be far enough from the sponge that neither its own stencil nor the
dissipation operator applied to its outermost point reaches into the region
where the stretching function is not the identity. At 4th order that is
`src_reach(3) + sgd_reach(2) = 5` grid points: the moment tensor spans
`ic-2..ic+3`, and `addsgd4` sweeps `+/-2` with `reach=1`. The two terms sum
rather than max, because they are consecutive stencils, not alternatives.

This number must stay equal to `margin_pts` at 4th order in SW4's own source
check.
"""

ADIABATIC_COEFFICIENT = (2772.0 / 1024.0) / (2.0 * math.pi)
"""The constant in the supergrid's adiabatic absorption criterion.

SW4's stretching function has `Psi0'(xi) = 2772 * xi**5 * (1 - xi)**5`, so
`max|Psi0'| = 2772/1024` at `xi = 0.5`. The layer absorbs a wave adiabatically
while `W * cos(theta) / lambda >> max|Psi0'| / (2 pi)`, which gives a longest
absorbable period of `W * cos(theta) / (ADIABATIC_COEFFICIENT * c)`.
"""


def supergrid_width(sw4_params: SW4Parameters, coarsest_resolution: float) -> float:
    """Compute the supergrid sponge width SW4 will use, in metres.

    SW4's `supergrid` command accepts either a thickness in grid points (`gp=`)
    or a width in metres (`width=`), and `CHECK_INPUT` in `parseInputFile.C`
    makes them mutually exclusive. When both are somehow present, `width=` wins,
    matching SW4's own precedence. When neither is given (or there is no
    `supergrid` command at all) SW4 falls back to its own default `gp`.

    The grid-point form is measured on SW4's *coarsest* grid (`mGridSize[0]`),
    which is the bottom refinement layer, because a single scalar sponge width
    serves every grid and every face.

    Parameters
    ----------
    sw4_params : SW4Parameters
        The SW4 parameters read from the realisation (or defaults).
    coarsest_resolution : float
        The coarsest grid spacing in the run, in metres. See
        `coarsest_resolution`.

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

    This is SW4's `mGridSize[0]`: the grid spacing of the deepest (and so
    coarsest) mesh refinement once the theoretical refinements have been
    resolved against the domain depth.

    Parameters
    ----------
    refinements : Refinements
        The theoretical mesh refinements from the realisation (or defaults).
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


def minimum_fault_buffer_m(
    sw4_params: SW4Parameters, coarsest_resolution: float
) -> float:
    """Compute the smallest fault buffer that clears the supergrid sponge.

    The buffer is **additive**, `sponge + STENCIL_MARGIN_GRIDPOINTS * h`, not a
    multiple of the sponge width. The two terms have different physical origins:
    the sponge is a geometric exclusion (SW4 solves a different PDE inside it),
    while the stencil margin is a discretisation margin that scales with the
    grid spacing. A multiplicative `1.2 * sponge` would collapse to a 1.2 km
    margin on a 200 m grid even though the stencil still spans five points.

    At SW4's default `gp=30` this is `(30 + 5) * h`, i.e. exactly 14 km on a
    400 m grid and 7 km on a 200 m grid.

    Parameters
    ----------
    sw4_params : SW4Parameters
        The SW4 parameters read from the realisation (or defaults).
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
        The `velocity_model.fault_buffer` value, in kilometres. The domain edge
        is guaranteed to be at least this far from every source.
    sw4_params : SW4Parameters
        The SW4 parameters read from the realisation (or defaults).
    coarsest_resolution : float
        The coarsest grid spacing in the run, in metres.

    Raises
    ------
    ValueError
        If the buffer is narrower than the sponge plus the stencil margin, so a
        source could sit inside the absorbing layer.
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

    Both lateral axes carry a sponge on each side, so the interior of an axis is
    the extent less twice the sponge width. That interior has to be wide enough
    for the stencil margin on each side, otherwise the two sponges effectively
    meet and there is nowhere in the grid where SW4 solves the wave equation
    undisturbed.

    Parameters
    ----------
    x_m, y_m : float
        The lateral extents of the SW4 grid, in metres, in SW4's own axis
        convention (`x` is north).
    sw4_params : SW4Parameters
        The SW4 parameters read from the realisation (or defaults).
    coarsest_resolution : float
        The coarsest grid spacing in the run, in metres.

    Raises
    ------
    ValueError
        If either axis has too little interior left between its two sponges.
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

    The supergrid absorbs adiabatically while the layer is long compared to a
    wavelength measured along the layer normal, i.e. while
    `W cos(theta) / lambda >> ADIABATIC_COEFFICIENT`. Setting that ratio to one
    gives the longest period the layer still absorbs:
    `W cos(theta) / (ADIABATIC_COEFFICIENT * c)`. Longer periods are reflected
    rather than absorbed, and with a source in or near the layer they ring.

    Parameters
    ----------
    sw4_params : SW4Parameters
        The SW4 parameters read from the realisation (or defaults).
    coarsest_resolution : float
        The coarsest grid spacing in the run, in metres.
    vs_km_s : float
        The shear wave speed at the layer, in km/s. The slowest material
        against the layer is the conservative choice.
    incidence_degrees : float, default 0.0
        The angle between the ray and the layer normal, in degrees. Grazing
        incidence degrades absorption by `cos(theta)`.

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
