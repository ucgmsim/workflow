import copy
import itertools
import string
from pathlib import Path

import h5py
import typer
from nzcvm.formats import sfile
from nzcvm.grids import sw4

from qcore import cli
from workflow.realisations import (
    DomainParameters,
    RealisationMetadata,
    Refinement,
    Refinements,
    SW4Command,
    SW4Parameters,
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

    x = domain_parameters.domain.extent_x * 1000.0
    y = domain_parameters.domain.extent_y * 1000.0
    with h5py.File(velocity_model, "r") as f:
        # Azimuth must be the same as the velocity model inside SW4
        azimuth = azimuth_from_velocity_model(f)
        topography_height, sfile_zmax = topography_height_from_velocity_model(f)

    # HACK: SW4 User Guide (Chapter 5) suggests
    # z_max >= -e_min + 3 (e_max - e_min) where e_min, e_max are the minimum
    # and maximum topography level of the velocity model we will assume that the
    # minimum elevation is zero (i.e. every simulation contains ocean, and there
    # is no ocean bathymetry).
    topography_zmax = 3 * topography_height

    # In SW4 the domain should always begin from the bottom-left (which is corners[0] by construction)
    lat, lon = domain_parameters.domain.corners[0]

    depth = domain_parameters.depth
    time = domain_parameters.duration
    refinements = theoretical_refinements.refinements_for_depth(depth)

    refinements, topography_zmax = adjust_for_topography(
        refinements, topography_zmax, nzmin=sw4_params.nz_min
    )
    refinements = sorted(refinements, key=lambda r: r.bottom)
    # Per the SW4 User Guide, the supergrid sponge (30 gridpoints by default) at the bottom of the domain
    # must be contained in the bottom refinement.
    supergrid_width = sw4_params.supergrid_padding * refinements[-1].resolution
    refinements[-1].bottom += supergrid_width
    depth += supergrid_width / 1000.0

    if refinements[-1].bottom > sfile_zmax:
        raise ValueError("Bottom of domain exceeds velocity model bounds")

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
