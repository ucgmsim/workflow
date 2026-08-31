import copy
import itertools
import string
from pathlib import Path

import h5py
import typer
from nzcvm.formats import sfile

from qcore import cli
from workflow import log_utils, sw4
from workflow.realisations import (
    DomainParameters,
    RealisationMetadata,
    Refinement,
    Refinements,
    SW4Command,
    SW4Parameters,
    VelocityModelParameters,
    find_command,
)

app = typer.Typer()

IMAGE_TIME_KEYS = frozenset({"time", "timeInterval", "cycle", "cycleInterval"})
"""Parameter keys that determine when an `imagehdf5` command fires. If none of
these are set, SW4 never emits the image, so we default to the simulation end time."""

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


def azimuth_from_velocity_model(velocity_model: h5py.File) -> float:
    """
    Extract the azimuth value from a velocity model HDF5 file.

    Parameters
    ----------
    velocity_model : h5py.File
        HDF5 file object containing velocity model attributes.

    Returns
    -------
    float
        The azimuth value stored in the file's attribute.
    """
    _, _, azimuth = velocity_model.attrs[sfile.ORIGIN_AZIM_ATTR]
    return float(azimuth)


def topography_height_from_velocity_model(
    velocity_model: h5py.File,
) -> tuple[float, float]:
    """
    Extract the minimum topography height from a velocity model HDF5 file.

    Parameters
    ----------
    velocity_model : h5py.File
        HDF5 file object containing velocity model attributes.

    Returns
    -------
    float
        The minimum depth (topography height) stored in the file's attribute.
    """
    global_min, zmax = velocity_model.attrs[sfile.MIN_MAX_DEPTH_ATTR]
    return -float(global_min), float(zmax)


def lateral_footprint_from_velocity_model(
    velocity_model: h5py.File,
) -> tuple[float, float]:
    """Measure the lateral footprint of a velocity model sfile, in metres.

    An sfile does not record its own lateral extent, so it is recovered from the
    shape of a material grid and that grid's horizontal spacing. Every grid in an
    sfile covers the same footprint, so any of them would do; the coarsest is
    used because it is the smallest array to describe.

    The sfile format requires the outermost axis to be due north, which is SW4's
    `x`, so the returned pair is in SW4's axis convention and not the workflow's.

    Parameters
    ----------
    velocity_model : h5py.File
        HDF5 file object containing a velocity model sfile.

    Returns
    -------
    tuple[float, float]
        The extents of the model along SW4's `x` (north) and `y` (east) axes, in
        metres.
    """
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


def build_sw4_commands(
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

    grid_command = None
    other_commands = []
    for command in commands:
        if command is grid:
            grid_command = command.merged(
                x=x, y=y, z=z, h=dx, az=azimuth, lon=lon, lat=lat
            )
        elif topography is not None and command is topography:
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


def adjust_for_topography(
    refinements: list[Refinement], topography_zmax: float, nzmin: int = 12
) -> tuple[list[Refinement], float]:
    """Deepen refinement layers so each holds at least `nzmin` cells.

    Topography raises the top of the grid above z=0, which eats into the
    first refinement layer. SW4 needs every grid in the stack to be at
    least `nzmin` cells deep, so any layer left too thin is pushed down
    until it is. The topography surface is inserted as a boundary for the
    purpose of that count, but is not returned as a refinement.

    Parameters
    ----------
    refinements : list of Refinement
        The refinement layers, resolved for the domain depth.
    topography_zmax : float
        Depth of the lowest point of the topography (metres, positive
        down).
    nzmin : int, optional
        Minimum number of cells in any layer.

    Returns
    -------
    tuple
        The adjusted refinements, and the topography depth used.
    """
    # Ensure no side effects
    refinements = copy.deepcopy(refinements)
    # By shallow copying the refinements before modifying them this view into the refinements will only have the updated refinements, and not the topography and bottom.
    real_refinements = refinements.copy()
    topography_resolution = min(
        (
            refinement
            for refinement in refinements
            if refinement.bottom > topography_zmax
        ),
        key=lambda r: r.bottom,
    ).resolution
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
    """Generate SW4 template for realisation

    Parameters
    ----------
    realisation_ffp : Path
        Path to realisation file.
    station_path : Path
        Path to station file.
    srf_path : Path
        Path to srf file.
    velocity_model : Path
        Path to velocity model file.
    work_directory : Path
        Path to work directory.
    output_path : Path
        Path to output SW4 file.
    """
    metadata = RealisationMetadata.read_from_realisation(realisation_ffp)
    domain_parameters = DomainParameters.read_from_realisation(realisation_ffp)
    theoretical_refinements = Refinements.read_from_realisation_or_defaults(
        realisation_ffp, metadata.defaults_version
    )
    sw4_params = SW4Parameters.read_from_realisation_or_defaults(
        realisation_ffp, metadata.defaults_version
    )
    velocity_model_parameters = (
        VelocityModelParameters.read_from_realisation_or_defaults(
            realisation_ffp, metadata.defaults_version
        )
    )
    logger = log_utils.get_logger(__name__)

    with h5py.File(velocity_model, "r") as f:
        # Azimuth must be the same as the velocity model inside SW4
        azimuth = azimuth_from_velocity_model(f)
        topography_height, sfile_zmax = topography_height_from_velocity_model(f)
        sfile_x, sfile_y = lateral_footprint_from_velocity_model(f)

    # HACK: SW4 User Guide (Chapter 5) suggests
    # z_max >= -e_min + 3 (e_max - e_min) where e_min, e_max are the minimum
    # and maximum topography level of the velocity model we will assume that the
    # minimum elevation is zero (i.e. every simulation contains ocean, and there
    # is no ocean bathymetry).
    topography_zmax = 3 * topography_height

    depth = domain_parameters.depth
    time = domain_parameters.duration
    refinements = theoretical_refinements.refinements_for_depth(depth)

    refinements, topography_zmax = adjust_for_topography(
        refinements, topography_zmax, nzmin=sw4_params.nz_min
    )
    refinements = sorted(refinements, key=lambda r: r.bottom)
    # The sponge width is a single scalar in SW4, measured on the coarsest grid,
    # and it applies to every face of every grid.
    coarsest_resolution = refinements[-1].resolution
    supergrid_width = sw4.supergrid_width(sw4_params, coarsest_resolution)

    # Gate B: the last gate before an SW4 run. This catches hand-edited
    # realisations and those produced by `copy-domain-parameters` or
    # `import-realisation`, which never go through `generate-domain`.
    sw4.check_fault_buffer(
        velocity_model_parameters.fault_buffer, sw4_params, coarsest_resolution
    )

    # Per the SW4 User Guide, the supergrid sponge (30 gridpoints by default) at the bottom of the domain
    # must be contained in the bottom refinement.
    refinements[-1].bottom += supergrid_width
    depth += supergrid_width / 1000.0

    # The sponge is carved out of the grid, not out of the requested domain, so
    # the SW4 grid is the requested domain padded by one sponge width per lateral
    # face. Without this the lateral sponges eat into the region the domain was
    # sized to cover and a source can end up inside the absorbing layer.
    # `BoundingBox.pad` works in kilometres and pads along the box's own rotated
    # axes, so no trigonometry is needed here. The velocity model is padded by at
    # least as much in `create-nzvm-input`, so SW4 never queries outside the sfile.
    supergrid_width_km = supergrid_width / 1000.0
    padded_domain = domain_parameters.domain.pad(
        pad_x=(supergrid_width_km, supergrid_width_km),
        pad_y=(supergrid_width_km, supergrid_width_km),
    )
    x = padded_domain.extent_x * 1000.0
    y = padded_domain.extent_y * 1000.0

    # In SW4 the domain should always begin from the bottom-left (which is corners[0] by construction)
    lat, lon = padded_domain.corners[0]

    # NOTE: In SW4 x = north, but in the workflow y = north.
    sw4.check_lateral_gridpoints(y, x, sw4_params, coarsest_resolution)

    if refinements[-1].bottom > sfile_zmax:
        raise ValueError("Bottom of domain exceeds velocity model bounds")

    # The sfile and the SW4 grid are padded symmetrically about the same
    # centroid with the same azimuth, so comparing extents is a containment
    # check.
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
            # NOTE: `s_wave_velocity` is in m/s, `absorbed_period` wants km/s.
            velocity_model_parameters.s_wave_velocity / 1000.0,
            incidence_degrees=60.0,
        ),
        fault_buffer_km=velocity_model_parameters.fault_buffer,
    )

    refinements_str = "\n".join(
        SW4Command("refinement", {"zmax": f"{refinement.bottom:.1f}"}).render()
        for refinement in refinements[
            :-1
        ]  # Last refinement layer is implicitly the bottom of the domain
    )
    dx = refinements[-1].resolution

    velocity_model_directory = velocity_model.parent
    velocity_model_name = velocity_model.name

    grid_command, other_commands = build_sw4_commands(
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
                    "path": work_directory,
                    "verbose": sw4_params.verbose,
                    "printcycle": sw4_params.printcycle,
                },
            ).render(),
            grid=grid_command.render(),
            time=SW4Command("time", {"t": time}).render(),
            rupturehdf5=SW4Command("rupturehdf5", {"file": srf_path}).render(),
            refinements=refinements_str,
            other_commands="\n".join(command.render() for command in other_commands),
            sfile=SW4Command(
                "sfile",
                {
                    "filename": velocity_model_name,
                    "directory": velocity_model_directory,
                },
            ).render(),
            rechdf5=SW4Command(
                "rechdf5",
                {
                    "infile": station_path,
                    "outfile": low_frequency_output.relative_to(work_directory),
                },
            ).render(),
        )
    )
