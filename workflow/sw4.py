"""SW4 grid geometry and checks.

Covers the refined grid and the supergrid (absorbing sponge). Inside the sponge
SW4 solves a damped equation, so sources and receivers there do not produce
valid ground motion.
"""

import dataclasses
import math

import numpy as np
import numpy.typing as npt

from workflow.realisations import (
    Refinement,
    SW4Parameters,
    SW4Resolution,
    find_command,
)

SW4_DEFAULT_SUPERGRID_GRIDPOINTS = 30
"""SW4's default supergrid thickness, in grid points (`sw4/src/EW.C`)."""

SW4_DEFAULT_CFL = 1.3
"""SW4's default CFL number at 4th order (`mCFL` in `sw4/src/EW.C`)."""

TOPOGRAPHY_ZMAX_RELIEF_FACTOR = 3.0
"""Multiple of the topographic relief the curvilinear grid extends below it.

From the SW4 User Guide (Chapter 5): `zmax >= tau_max + 3 (tau_max - tau_min)`.
"""

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


def minimum_fault_buffer_m(coarsest_resolution: float) -> float:
    """Compute the smallest fault buffer that clears the supergrid sponge.

    `create-sw4-input` pads the domain by one sponge width on every lateral
    face, so the sponge lies wholly outside the domain and the buffer only
    needs the `STENCIL_MARGIN_GRIDPOINTS` of clearance beyond it.

    Parameters
    ----------
    coarsest_resolution : float
        The coarsest grid spacing in the run, in metres.

    Returns
    -------
    float
        The minimum fault buffer, in metres.
    """
    return STENCIL_MARGIN_GRIDPOINTS * coarsest_resolution


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
    minimum = minimum_fault_buffer_m(coarsest_resolution)
    if fault_buffer_km * 1000.0 < minimum:
        sponge = supergrid_width(sw4_params, coarsest_resolution)
        raise ValueError(
            f"The fault buffer of {fault_buffer_km:.3f} km is smaller than the "
            f"{minimum / 1000.0:.3f} km needed to keep sources clear of the SW4 "
            "supergrid absorbing layer. `create-sw4-input` places the "
            f"{sponge / 1000.0:.3f} km sponge outside the domain, but a source "
            f"still needs {STENCIL_MARGIN_GRIDPOINTS} grid points of clearance "
            f"on a {coarsest_resolution:.0f} m coarsest grid for its own "
            "stencil and the dissipation operator. Raise "
            f"velocity_model.fault_buffer to at least {minimum / 1000.0:.3f} km."
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


def topography_zmax(elevation_min: float, elevation_max: float) -> float:
    """Compute the depth the curvilinear grid should extend to, in metres.

    This is the SW4 User Guide's `zmax >= tau_max + 3 (tau_max - tau_min)`, in
    elevations (`tau = -e`). It depends on the lowest elevation as well as the
    highest: a domain entirely inland needs less curvilinear grid than one
    reaching the coast, and bathymetry needs more.

    Parameters
    ----------
    elevation_min, elevation_max : float
        The lowest and highest elevations of the top surface, in metres above
        sea level.

    Returns
    -------
    float
        The depth of the bottom of the curvilinear grid, in metres below sea
        level.
    """
    return -elevation_min + TOPOGRAPHY_ZMAX_RELIEF_FACTOR * (
        elevation_max - elevation_min
    )


@dataclasses.dataclass
class VsProfile:
    """The slowest and fastest material at each depth of SW4's reference grid.

    Depths are in SW4's reference coordinate. Inside the curvilinear grid, SW4
    scales the topography linearly to zero at `zmax`
    (`GridGeneratorGeneral::assignInterfaceSurfaces`), so a column with top
    surface `tau` holds the reference depth `r` at the physical depth
    `tau + r (zmax - tau) / zmax`, and every curvilinear cell in that column is
    stretched vertically by `(zmax - tau) / zmax`. Refinement interfaces sit at
    fixed reference depths, so this is the coordinate they are sized in.

    The speeds are already adjusted for that stretch, so they can be compared
    directly with the nominal grid spacing `h`: `min_vs` against the coarsest
    spacing in a cell, `max(h, stretch h)`, and `max_wave_speed` against the
    finest, `min(h, stretch h)`. SW4's own `minVs/h` printout ignores the
    stretch (`EW::compute_minvsoverh`).
    """

    bin_size: float
    """The height of each depth bin, in metres."""
    min_vs: npt.NDArray[np.float64]
    """The slowest effective Vs in each bin (m/s), NaN where a bin is empty."""
    max_wave_speed: npt.NDArray[np.float64]
    """The fastest effective `sqrt(Vp^2 + 2 Vs^2)` in each bin (m/s), the speed
    SW4's time step is limited by (`EW::computeDT`), NaN where a bin is empty."""

    @classmethod
    def empty(cls, bin_size: float, n_bins: int) -> "VsProfile":
        """Build a profile with no material in any bin.

        Parameters
        ----------
        bin_size : float
            The height of each depth bin, in metres.
        n_bins : int
            The number of bins.

        Returns
        -------
        VsProfile
            The empty profile, which `merge` leaves any other profile unchanged
            by.
        """
        return cls(
            bin_size=bin_size,
            min_vs=np.full(n_bins, np.nan),
            max_wave_speed=np.full(n_bins, np.nan),
        )

    def filled(self) -> "VsProfile":
        """Fill each empty bin from the nearest non-empty bins either side of it.

        SW4 interpolates linearly between the model's samples, and clamps below
        its last one (`MaterialSfile.C`), so the material in an empty bin lies
        between the samples above and below it. Each empty bin takes the
        extremes of the two.

        Returns
        -------
        VsProfile
            The profile with every bin filled, unless it is entirely empty.
        """
        present = np.isfinite(self.min_vs)
        index = np.arange(1, len(present) + 1)
        # Indices into the values padded with NaN at both ends, which `fmin`
        # and `fmax` ignore where there is no sample on one side.
        above = np.maximum.accumulate(np.where(present, index, 0))
        below = np.minimum.accumulate(np.where(present, index, len(present) + 1)[::-1])[
            ::-1
        ]

        def padded(values: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
            return np.concatenate(([np.nan], values, [np.nan]))

        min_vs = padded(self.min_vs)
        max_wave_speed = padded(self.max_wave_speed)
        return VsProfile(
            bin_size=self.bin_size,
            min_vs=np.fmin(min_vs[above], min_vs[below]),
            max_wave_speed=np.fmax(max_wave_speed[above], max_wave_speed[below]),
        )

    def merge(self, other: "VsProfile") -> "VsProfile":
        """Combine two profiles of the same bins, keeping the extremes of each.

        Parameters
        ----------
        other : VsProfile
            The profile to combine with.

        Returns
        -------
        VsProfile
            The combined profile.
        """
        return VsProfile(
            bin_size=self.bin_size,
            min_vs=np.fmin(self.min_vs, other.min_vs),
            max_wave_speed=np.fmax(self.max_wave_speed, other.max_wave_speed),
        )


def vs_profile(
    z: npt.ArrayLike,
    tau: npt.ArrayLike,
    vs: npt.ArrayLike,
    vp: npt.ArrayLike,
    topography_zmax: float,
    bin_size: float,
    n_bins: int,
) -> VsProfile:
    """Bin material samples by SW4 reference depth.

    Parameters
    ----------
    z : array-like
        The physical depth of each sample, in metres below sea level.
    tau : array-like
        The top surface depth of each sample's column, in metres below sea
        level (negative above it). Broadcast against `z`.
    vs, vp : array-like
        The S and P wave speeds at each sample, in m/s.
    topography_zmax : float
        The depth of the bottom of the curvilinear grid, in metres.
    bin_size : float
        The height of each depth bin, in metres.
    n_bins : int
        The number of bins. Deeper samples land in the last bin.

    Returns
    -------
    VsProfile
        The profile of these samples.
    """
    z = np.asarray(z, dtype=np.float64)
    tau = np.asarray(tau, dtype=np.float64)
    vs = np.asarray(vs, dtype=np.float32)
    vp = np.asarray(vp, dtype=np.float32)

    # The stretch is a property of the column, so compute it before it is
    # broadcast against depth.
    with np.errstate(divide="ignore", invalid="ignore"):
        column_stretch = (topography_zmax - tau) / topography_zmax
        curvilinear = z < topography_zmax
        reference = np.where(curvilinear, (z - tau) / column_stretch, z)
    labels = np.clip(reference // bin_size, 0, n_bins - 1).astype(np.intp).ravel()
    slowest = np.where(curvilinear, np.maximum(column_stretch, 1.0), 1.0)
    fastest = np.where(curvilinear, np.minimum(column_stretch, 1.0), 1.0)

    min_vs = np.full(n_bins, np.inf)
    np.minimum.at(min_vs, labels, (vs / slowest.astype(np.float32)).ravel())
    # Squared, so the square root is only taken of each bin's maximum.
    max_squared_speed = np.full(n_bins, -np.inf)
    np.maximum.at(
        max_squared_speed,
        labels,
        ((vp**2 + 2 * vs**2) / fastest.astype(np.float32) ** 2).ravel(),
    )

    empty = np.isinf(min_vs)
    return VsProfile(
        bin_size=bin_size,
        min_vs=np.where(empty, np.nan, min_vs),
        max_wave_speed=np.sqrt(np.where(empty, np.nan, max_squared_speed)),
    )


def size_refinements(
    profile: VsProfile, resolution: SW4Resolution, depth_m: float, nz_min: int
) -> list[Refinement]:
    """Size SW4's mesh refinements from the material in the velocity model.

    Each layer is twice the spacing of the one above it, and starts at the
    shallowest reference depth below which no material is too slow for it to
    keep `resolution.minimum_ppw` at `resolution.max_frequency`. Interfaces are
    rounded deeper onto the coarser grid, and every layer keeps `nz_min` cells,
    so a layer that would be thinner than that above the domain bottom is left
    out.

    Parameters
    ----------
    profile : VsProfile
        The velocity model's profile.
    resolution : SW4Resolution
        The resolution targets.
    depth_m : float
        The domain depth, in metres, excluding the bottom sponge.
    nz_min : int
        The fewest cells a layer may hold.

    Returns
    -------
    list of Refinement
        The layers from the surface down. The last layer's bottom is `depth_m`.
    """
    # The slowest material at or below each bin.
    floor = np.fmin.accumulate(profile.filled().min_vs[::-1])[::-1]
    floor = np.where(np.isnan(floor), np.inf, floor)

    refinements: list[Refinement] = []
    top = 0.0
    current, *coarser_resolutions = resolution.resolutions
    for coarser in coarser_resolutions:
        needed = resolution.minimum_ppw * resolution.max_frequency * coarser
        # `floor` never decreases with depth, so the bins it clears are a suffix.
        (cleared,) = np.nonzero(floor >= needed)
        if not cleared.size:
            # Nothing is fast enough for the coarser grid.
            break
        # Rounding up onto the coarser grid keeps every uncleared bin in the
        # finer layer.
        bottom = max(
            math.ceil(cleared[0] * profile.bin_size / coarser) * coarser,
            math.ceil((top + nz_min * current) / coarser) * coarser,
        )
        # A coarser layer too thin to hold `nz_min` cells above the domain
        # bottom is not worth starting.
        if bottom + nz_min * coarser > depth_m:
            break
        refinements.append(Refinement(resolution=current, bottom=bottom))
        top, current = bottom, coarser

    refinements.append(Refinement(resolution=current, bottom=depth_m))
    return refinements


# Each layer's slowest effective Vs and fastest effective wave speed over the
# bins whose tops lie inside it, NaN where the profile has no material there.
def _layer_extremes(
    profile: VsProfile, refinements: list[Refinement]
) -> list[tuple[Refinement, float, float]]:
    profile = profile.filled()
    extremes = []
    top = 0.0
    for refinement in refinements:
        bins = slice(
            math.ceil(top / profile.bin_size),
            math.ceil(refinement.bottom / profile.bin_size),
        )
        min_vs = profile.min_vs[bins]
        present = np.isfinite(min_vs)
        if present.any():
            slowest = float(min_vs[present].min())
            fastest = float(profile.max_wave_speed[bins][present].max())
        else:
            slowest = fastest = math.nan
        extremes.append((refinement, slowest, fastest))
        top = refinement.bottom
    return extremes


def layer_ppw(
    profile: VsProfile, refinements: list[Refinement], frequency: float
) -> list[float]:
    """Compute the points per shortest S wavelength each layer achieves.

    Unlike SW4's `minVs/h` printout, this accounts for the curvilinear stretch.

    Parameters
    ----------
    profile : VsProfile
        The velocity model's profile.
    refinements : list of Refinement
        The layers from the surface down.
    frequency : float
        The frequency to measure at, in Hz.

    Returns
    -------
    list of float
        Each layer's points per wavelength, NaN if the profile has no material
        in it.
    """
    return [
        min_vs / (refinement.resolution * frequency)
        for refinement, min_vs, _ in _layer_extremes(profile, refinements)
    ]


def layer_time_steps(
    profile: VsProfile, refinements: list[Refinement], cfl: float = SW4_DEFAULT_CFL
) -> list[float]:
    """Estimate the stable time step of each layer, in seconds.

    SW4 steps every grid at the smallest of these (`EW::computeDT`), so the
    layer with the smallest one sets the cost of the whole run. This ignores
    attenuation, which lowers SW4's time step slightly.

    Parameters
    ----------
    profile : VsProfile
        The velocity model's profile.
    refinements : list of Refinement
        The layers from the surface down.
    cfl : float, default SW4_DEFAULT_CFL
        The CFL number.

    Returns
    -------
    list of float
        Each layer's time step, NaN if the profile has no material in it.
    """
    return [
        cfl * refinement.resolution / fastest
        for refinement, _, fastest in _layer_extremes(profile, refinements)
    ]
