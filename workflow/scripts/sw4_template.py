"""SW4 Input Generation.

Description
-----------
Render the SW4 input file for a realisation: the grid, the refinement stack,
the source, the stations, and the commands in the realisation's `sw4` section.
The requested domain becomes the grid's interior, padded laterally by one
supergrid width on each side.

SW4's refinements are sized from the Vs in the velocity model, to the targets
in the realisation's `sw4_resolution` section. They are independent of the
velocity model's own `refinements`, which only set how finely it is sampled.

Inputs
------
1. A realisation file containing domain parameters,
2. A station file,
3. An SRF file,
4. An SW4 sfile velocity model.

Outputs
-------
An SW4 input file.
"""

import copy
import itertools
import math
import string
from collections.abc import Iterator
from pathlib import Path

import h5py
import numpy as np
import numpy.typing as npt
import typer
from nzcvm.formats import sfile

from qcore import cli
from workflow import log_utils, sw4
from workflow.realisations import (
    DomainParameters,
    RealisationMetadata,
    Refinement,
    SW4Command,
    SW4Parameters,
    SW4Resolution,
    VelocityModelParameters,
    find_command,
)

app = typer.Typer()

IMAGE_TIME_KEYS = frozenset({"time", "timeInterval", "cycle", "cycleInterval"})
"""Parameter keys that determine when an `imagehdf5` command fires. If none of
these are set, SW4 never emits the image, so we default to the simulation end time."""

PROFILE_READ_BYTES = 2**30
"""Bytes of `Cs` and `Cp` read at once when profiling the velocity model."""

PROFILE_BLOCK_ELEMENTS = 2**22
"""Samples binned at once when profiling the velocity model. Binning holds
about ten float64 temporaries per sample, so this is about 340 MB."""

SW4_TEMPLATE = string.Template("""
${fileio}

${grid}

${time}

${rupturehdf5}
${refinements}

${other_commands}

${sfile}
${rechdf5}
""")


def _azimuth_from_velocity_model(velocity_model: h5py.File) -> float:
    """Extract the azimuth value from a velocity model HDF5 file."""
    _, _, azimuth = velocity_model.attrs[sfile.ORIGIN_AZIM_ATTR]
    return float(azimuth)


# The depth of the bottom of the velocity model, in metres.
def _model_bottom_from_velocity_model(velocity_model: h5py.File) -> float:
    _, bottom = velocity_model.attrs[sfile.MIN_MAX_DEPTH_ATTR]
    return float(bottom)


# Split `range(n_rows)` into consecutive slices of at most `rows_per_block`.
def _row_blocks(n_rows: int, rows_per_block: int) -> Iterator[slice]:
    for start in range(0, n_rows, rows_per_block):
        yield slice(start, min(start + rows_per_block, n_rows))


# The lowest and highest elevations of the model's top surface, in metres.
def _elevation_range_from_velocity_model(
    velocity_model: h5py.File,
) -> tuple[float, float]:
    surface = velocity_model[sfile.SURFACE_GROUP]["z_values_0"]
    rows_per_block = max(1, PROFILE_BLOCK_ELEMENTS // surface.shape[1])
    # The sfile stores depths, positive down.
    shallowest, deepest = math.inf, -math.inf
    for rows in _row_blocks(surface.shape[0], rows_per_block):
        block = surface[rows]
        shallowest = min(shallowest, float(block.min()))
        deepest = max(deepest, float(block.max()))
    return -deepest, -shallowest


# Read rows of a finer grid's surface, subsampled onto a coarser grid sharing
# its corners, as SW4 assumes of an sfile's interfaces (`MaterialSfile.C`).
def _decimated_rows(
    surface: h5py.Dataset, shape: tuple[int, int], rows: slice
) -> npt.NDArray[np.float64]:
    fine_i, fine_j = surface.shape
    stride_i = (fine_i - 1) // max(shape[0] - 1, 1)
    stride_j = (fine_j - 1) // max(shape[1] - 1, 1)
    if (shape[0] - 1) * stride_i != fine_i - 1 or (
        shape[1] - 1
    ) * stride_j != fine_j - 1:
        raise ValueError(
            f"A {surface.shape} surface does not decimate onto a {shape} grid."
        )
    fine_rows = slice(rows.start * stride_i, (rows.stop - 1) * stride_i + 1, stride_i)
    return np.asarray(surface[fine_rows, ::stride_j], dtype=np.float64)


# How many rows of `Cs` and `Cp` to read together within `budget_bytes`. Rows
# are whole chunk rows where they fit, so no chunk is read twice. Where one
# row of chunks is over budget, HDF5 still decompresses whole chunks, so
# memory is then bounded by the chunk size.
def _profile_read_rows(dataset: h5py.Dataset, budget_bytes: int) -> int:
    ni, nj, nk = dataset.shape
    row_bytes = 2 * nj * nk * dataset.dtype.itemsize
    rows = max(1, budget_bytes // row_bytes)
    if dataset.chunks is not None and dataset.chunks[0] <= rows:
        rows -= rows % dataset.chunks[0]
    return min(rows, ni)


# Profile the model's slowest and fastest material by SW4 reference depth.
# This reads the sfile as SW4 does (`MaterialSfile.C`), a block of rows at a
# time, so it works for any sfile and for models larger than memory: `ngrids`
# grids, each spanning `z_values_{g}` to `z_values_{g + 1}` with its vertical
# points spaced evenly between them.
def _vs_profile_from_velocity_model(
    velocity_model: h5py.File, topography_zmax: float, bin_size: float
) -> sw4.VsProfile:
    material = velocity_model[sfile.MATERIAL_GROUP]
    interfaces = velocity_model[sfile.SURFACE_GROUP]
    surface = interfaces["z_values_0"]
    n_bins = math.ceil(_model_bottom_from_velocity_model(velocity_model) / bin_size) + 1

    profile = sw4.VsProfile.empty(bin_size, n_bins)
    for index in range(int(velocity_model.attrs[sfile.NGRIDS_ATTR])):
        grid = material[f"grid_{index}"]
        vs, vp = grid["Cs"], grid["Cp"]
        ni, nj, nk = vs.shape
        top_surface = interfaces[f"z_values_{index}"]
        bottom_surface = interfaces[f"z_values_{index + 1}"]
        fraction = np.linspace(0.0, 1.0, nk)

        bin_rows = max(1, PROFILE_BLOCK_ELEMENTS // (nj * nk))
        for read in _row_blocks(ni, _profile_read_rows(vs, PROFILE_READ_BYTES)):
            vs_block, vp_block = vs[read], vp[read]
            top = _decimated_rows(top_surface, (ni, nj), read)
            bottom = np.asarray(bottom_surface[read], dtype=np.float64)
            tau = _decimated_rows(surface, (ni, nj), read)
            for rows in _row_blocks(read.stop - read.start, bin_rows):
                z = (
                    top[rows, :, None]
                    + fraction * (bottom[rows] - top[rows])[..., None]
                )
                partial = sw4.vs_profile(
                    z,
                    tau[rows, :, None],
                    vs_block[rows],
                    vp_block[rows],
                    topography_zmax,
                    bin_size,
                    n_bins,
                )
                profile = profile.merge(partial)

    return profile


def _lateral_footprint_from_velocity_model(
    velocity_model: h5py.File,
) -> tuple[float, float]:
    """Measure the lateral footprint of a velocity model sfile, in metres."""
    material = velocity_model[sfile.MATERIAL_GROUP]
    grid_name = max(
        material, key=lambda name: material[name].attrs[sfile.HORIZONTAL_ATTR]
    )
    grid = material[grid_name]
    resolution = float(grid.attrs[sfile.HORIZONTAL_ATTR])
    # Every material component of a grid has the same shape, so the first one is
    # representative.
    nx, ny = next(iter(grid.values())).shape[:2]
    return (nx - 1) * resolution, (ny - 1) * resolution


# NOTE: despite being private, this function has a complete numpy docstring to help explain the outputs.
def _build_sw4_commands(
    sw4_params: SW4Parameters,
    x: float,
    y: float,
    z: float,
    dx: float,
    azimuth: float,
    lon: float,
    lat: float,
    velocity_model_name: str,
    velocity_model_directory: Path,
    topography_zmax: float,
    simulation_time: float,
) -> tuple[SW4Command, list[SW4Command]]:
    """Resolve SW4Parameters.commands, layering in runtime-computed values.

    Parameters
    ----------
    sw4_params : SW4Parameters
        The SW4 parameters read from the realisation (or defaults).
    x, y, z : float
        Simulation domain extents (metres).
    dx : float
        Grid spacing of the bottom refinement layer (metres).
    azimuth : float
        Grid azimuth, taken from the velocity model.
    lon, lat : float
        Grid origin coordinates.
    velocity_model_name : str
        Filename of the velocity model sfile.
    velocity_model_directory : Path
        Directory containing the velocity model sfile.
    topography_zmax : float
        Computed maximum topography depth.
    simulation_time : float
        Simulation duration (seconds), used as the default image output time.

    Returns
    -------
    tuple[SW4Command, list[SW4Command]]
        The resolved `grid` command, and every other resolved command.

    Raises
    ------
    ValueError
        If `sw4_params.commands` has no `grid` command.
    """
    commands = sw4_params.commands
    grid = find_command(commands, "grid")
    if grid is None:
        raise ValueError("SW4 configuration is missing a required 'grid' command")
    topography = find_command(commands, "topography")

    grid_command = grid.merged(x=x, y=y, z=z, h=dx, az=azimuth, lon=lon, lat=lat)
    other_commands = []
    for command in commands:
        if command is grid:
            continue
        if topography is not None and command is topography:
            other_commands.append(
                command.merged(
                    input="sfile",
                    zmax=topography_zmax,
                    file=f"{velocity_model_directory}/{velocity_model_name}",
                )
            )
        elif command.name == "imagehdf5" and not (
            IMAGE_TIME_KEYS & command.parameters.keys()
        ):
            other_commands.append(command.merged(time=simulation_time))
        else:
            other_commands.append(command)

    return grid_command, other_commands


def _adjust_for_topography(
    refinements: list[Refinement], topography_zmax: float, nzmin: int = 12
) -> tuple[list[Refinement], float]:
    """Deepen refinement layers so each holds at least `nzmin` cells."""

    # GOAL: We want to ensure every refinement has at least `nzmin` layers. We
    # do this because SW4 requires a minimum number of gridpoints in each
    # refinement. On a free surface this is achieved simply by ensuring each
    # input refinement is at least `nzmin` gridpoints from the one above it.
    # Topography complicates the picture. SW4 requires that the topography is
    # backed by curvilinear layers down to a certain depth (see the HACK below).
    # These layers we will call topographic layers. Usually these layers subsume
    # the refinements listed in the realisation, so that the bookkeeping is
    # roughly the same. The last topographic layer can cut between input
    # refinements to introduce an additional *implicit* curvilinear layer. So
    # even if each refinement is well separated, the implicit layer might be
    # too thin. We illustrate all of this diagrammatically here.
    #
    # Example: refinements of 100 m down to 5000 m and 200 m down to 25000 m,
    # with topography_zmax = 5400 m and nzmin = 12 (depths not to scale).
    #
    #         BEFORE                                 AFTER
    #
    #   ~~~~~~~~~~~~~~~~~~~ topography        ~~~~~~~~~~~~~~~~~~~ topography
    #   | curvilinear     |                   | curvilinear     |
    #   | h = 100 m       |                   | h = 100 m       |
    #   +-----------------+ z = 5000          +-----------------+ z = 5000
    #   | implicit, 200 m | 2 cells (< 12)    | implicit        |
    #   +=================+ z = 5400          | curvilinear     |
    #   | cartesian       |   topo zmax       | h = 200 m       | 12 cells
    #   | h = 200 m       |                   |                 |
    #   |                 |                   +=================+ z = 7400
    #   |                 |                   | cartesian       |   topo zmax
    #   |                 |                   | h = 200 m       |
    #   +-----------------+ z = 25000         +-----------------+ z = 25000
    #
    #   ---  input refinement boundary      ===  topographic boundary (zmax)
    #
    # The input refinements are untouched here; only topography_zmax moves. Had
    # topography_zmax instead landed just *above* an input boundary (e.g.
    # 4800 m), the thin layer would be the one below it, and it is the input
    # refinement (5000 m -> 6000 m) that gets pushed down instead.

    # The solution is to ensure that the input refinement intersecting the
    # topographic boundary is wide enough so that above and below the
    # topographic boundary we still have `nzmin` points. This seems challenging,
    # but the algorithm actually ends up being simple:
    #
    # 1. Introduce the topography bottom as an additional refinement with the
    # same resolution as the bottom of the topography; this accounts for the
    # implicit layer that SW4 inserts in its models.
    # 2. Walk over each refinement and ensure that each refinement is separated
    # by `nzmin` cells from the one above.
    # 3. Delete the implicit layer but record its bottom.

    # This bottom is where we should tell SW4 to terminate the topographic
    # layers. That closes the loop and ensures the model we build here matches
    # what SW4 constructs in its code.

    # Ensure no side effects
    refinements = copy.deepcopy(refinements)
    # By shallow copying the refinements again, we can record all the
    # refinements the user specified, but their depths will be automatically
    # updated by the loop below, which mutates the refinements they share. It
    # also means that the topography layer (which is added to the `refinements`
    # list but not `real_refinements`) is not returned at the end.
    real_refinements = refinements.copy()
    try:
        topography_resolution = min(
            (
                refinement
                for refinement in refinements
                if refinement.bottom > topography_zmax
            ),
            key=lambda r: r.bottom,
        ).resolution
    except ValueError as e:
        e.add_note(
            "This can happen if the simulation domain is too shallow for topography,"
            " or the refinements are not deep enough to capture the topographic"
            " extent. Raise the simulation depth, or increase the depth of the"
            " refinements."
        )
        raise
    topography = Refinement(bottom=topography_zmax, resolution=topography_resolution)
    refinements.append(topography)
    refinements.sort(key=lambda r: r.bottom)

    for above, below in itertools.pairwise(refinements):
        thickness = below.bottom - above.bottom
        nz = thickness // below.resolution
        cells_needed = nzmin - nz
        if cells_needed > 0:
            below.bottom += cells_needed * below.resolution

    topography_zmax = topography.bottom

    return real_refinements, topography_zmax


@cli.from_docstring(app)
def generate_sw4_input(
    realisation_ffp: Path,
    station_path: Path,
    srf_path: Path,
    velocity_model: Path,
    work_directory: Path,
    output_path: Path,
) -> None:
    """Generate an SW4 input file for a realisation.

    Parameters
    ----------
    realisation_ffp : Path
        Path to the realisation file.
    station_path : Path
        Path to the station file.
    srf_path : Path
        Path to the SRF file.
    velocity_model : Path
        Path to the velocity model file.
    work_directory : Path
        Path to the work directory.
    output_path : Path
        Path to the output SW4 file.
    """
    metadata = RealisationMetadata.read_from_realisation(realisation_ffp)
    domain_parameters = DomainParameters.read_from_realisation(realisation_ffp)
    sw4_params = SW4Parameters.read_from_realisation_or_defaults(
        realisation_ffp, metadata.defaults_version
    )
    velocity_model_parameters = (
        VelocityModelParameters.read_from_realisation_or_defaults(
            realisation_ffp, metadata.defaults_version
        )
    )
    resolution = SW4Resolution.read_from_realisation_or_defaults(
        realisation_ffp, metadata.defaults_version
    )
    logger = log_utils.get_logger(__name__)

    depth = domain_parameters.depth
    time = domain_parameters.duration

    with h5py.File(velocity_model, "r") as f:
        # The grid azimuth must match the velocity model's azimuth inside SW4.
        azimuth = _azimuth_from_velocity_model(f)
        sfile_zmax = _model_bottom_from_velocity_model(f)
        sfile_x, sfile_y = _lateral_footprint_from_velocity_model(f)
        elevation_min, elevation_max = _elevation_range_from_velocity_model(f)
        topography_zmax = sw4.topography_zmax(elevation_min, elevation_max)

        # NOTE: `_adjust_for_topography` can only deepen `topography_zmax`,
        # which lessens the curvilinear stretch, so profiling against this
        # shallower one is conservative.
        profile = _vs_profile_from_velocity_model(
            f, topography_zmax, bin_size=resolution.finest_resolution
        )

    refinements = sw4.size_refinements(
        profile, resolution, depth * 1000.0, nz_min=sw4_params.nz_min
    )

    refinements, topography_zmax = _adjust_for_topography(
        refinements, topography_zmax, nzmin=sw4_params.nz_min
    )
    coarsest_resolution = refinements[-1].resolution

    developer = find_command(sw4_params.commands, "developer")
    cfl = developer.parameters.get("cfl") if developer is not None else None
    cfl = sw4.SW4_DEFAULT_CFL if cfl is None else float(cfl)
    logger.info(
        "SW4 refinements sized from the velocity model",
        topography_zmax_m=topography_zmax,
        elevation_range_m=(elevation_min, elevation_max),
        layers=[
            {
                "resolution_m": refinement.resolution,
                "bottom_m": refinement.bottom,
                "ppw": ppw,
                "time_step_s": time_step,
            }
            for refinement, ppw, time_step in zip(
                refinements,
                sw4.layer_ppw(profile, refinements, resolution.max_frequency),
                sw4.layer_time_steps(profile, refinements, cfl),
            )
        ],
    )
    supergrid_width = sw4.supergrid_width(sw4_params, coarsest_resolution)

    sw4.check_fault_buffer(
        velocity_model_parameters.fault_buffer, sw4_params, coarsest_resolution
    )

    # Per the SW4 User Guide, the supergrid sponge (30 gridpoints by default) at
    # the bottom of the domain must be contained in the bottom refinement.
    refinements[-1].bottom += supergrid_width
    depth += supergrid_width / 1000.0

    # Pad the sw4 domain so that the supergrid is accounted for.
    supergrid_width_km = supergrid_width / 1000.0
    padded_domain = domain_parameters.domain.pad(
        pad_x=(supergrid_width_km, supergrid_width_km),
        pad_y=(supergrid_width_km, supergrid_width_km),
    )
    x = padded_domain.extent_x * 1000.0
    y = padded_domain.extent_y * 1000.0

    # In SW4, the domain always begins at the bottom-left corner (which is
    # corners[0] by construction).
    lat, lon = padded_domain.corners[0]

    # NOTE: In SW4 x = north, but in the workflow y = north.
    sw4.check_lateral_gridpoints(y, x, sw4_params, coarsest_resolution)

    if refinements[-1].bottom > sfile_zmax:
        raise ValueError("Bottom of domain exceeds velocity model bounds")

    # Double-check that the sfile domain is large enough for the domain we are simulating.
    if y > sfile_x or x > sfile_y:
        raise ValueError(
            f"The SW4 grid ({y / 1000.0:.3f} x {x / 1000.0:.3f} km, "
            "north x east, including its supergrid padding) is not contained in "
            f"the velocity model ({sfile_x / 1000.0:.3f} x "
            f"{sfile_y / 1000.0:.3f} km). Regenerate the velocity model with "
            "`create-nzvm-input`, which pads the model to cover the padded SW4 "
            "grid."
        )

    logger.info(
        "SW4 supergrid geometry",
        supergrid_width_m=supergrid_width,
        absorbed_period_normal_incidence_s=sw4.absorbed_period(
            sw4_params,
            coarsest_resolution,
            # NOTE: `s_wave_velocity` is in m/s, `absorbed_period` wants km/s.
            velocity_model_parameters.s_wave_velocity / 1000.0,
        ),
        absorbed_period_60_degrees_s=sw4.absorbed_period(
            sw4_params,
            coarsest_resolution,
            velocity_model_parameters.s_wave_velocity / 1000.0,
            incidence_degrees=60.0,
        ),
        fault_buffer_km=velocity_model_parameters.fault_buffer,
    )

    refinements_str = "\n".join(
        SW4Command("refinement", {"zmax": f"{refinement.bottom:.1f}"}).render()
        for refinement in refinements[
            :-1
        ]  # The last refinement layer is implicitly the bottom of the domain.
    )
    dx = refinements[-1].resolution

    velocity_model_directory = velocity_model.parent
    velocity_model_name = velocity_model.name
    # The topography-following adjustment in `_adjust_for_topography` can push
    # the bottom refinement deeper, increasing the total depth of the model. Here
    # we account for that by updating `depth` to reflect this change.
    depth = max(depth, refinements[-1].bottom / 1000.0)
    grid_command, other_commands = _build_sw4_commands(
        sw4_params,
        # NOTE: In SW4 x = north, but in the workflow y = north.
        x=y,
        y=x,
        z=depth * 1000.0,
        dx=dx,
        azimuth=azimuth,
        lon=lon,
        lat=lat,
        velocity_model_name=velocity_model_name,
        velocity_model_directory=velocity_model_directory,
        topography_zmax=topography_zmax,
        simulation_time=time,
    )

    low_frequency_output = work_directory / "out.h5"
    output_path.write_text(
        SW4_TEMPLATE.substitute(
            fileio=SW4Command(
                "fileio",
                {
                    "path": str(work_directory),
                    "verbose": sw4_params.verbose,
                    "printcycle": sw4_params.printcycle,
                },
            ).render(),
            grid=grid_command.render(),
            time=SW4Command("time", {"t": time}).render(),
            rupturehdf5=SW4Command("rupturehdf5", {"file": str(srf_path)}).render(),
            refinements=refinements_str,
            other_commands="\n".join(command.render() for command in other_commands),
            sfile=SW4Command(
                "sfile",
                {
                    "filename": velocity_model_name,
                    "directory": str(velocity_model_directory),
                },
            ).render(),
            rechdf5=SW4Command(
                "rechdf5",
                {
                    "infile": str(station_path),
                    "outfile": str(low_frequency_output.relative_to(work_directory)),
                },
            ).render(),
        )
    )
