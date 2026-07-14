import copy
import itertools
from pathlib import Path

import h5py
import typer
from nzcvm.formats import sfile

from qcore import cli
from workflow.realisations import (
    DomainParameters,
    RealisationMetadata,
    Refinement,
    Refinements,
    SW4ImageOutput,
    SW4Parameters,
)

app = typer.Typer()
SW4_TEMPLATE = """
fileio path={work_directory} verbose={verbose} printcycle={printcycle}

supergrid gp={supergrid_gp}
time t={time}

grid x={x} y={y} z={z} h={dx} az={azimuth} lon={lon} lat={lat} proj={projection_type} ellps={projection_ellps} lon_p={projection_lon_p} lat_p={projection_lat_p} scale={projection_scale}
rupturehdf5 file={srf}

{refinement_str}

attenuation maxfreq={attenuation_maxfreq} phasefreq={attenuation_phasefreq} nmech={attenuation_nmech}

{image_output_str}

sfile filename={velocity_model_name} directory={velocity_model_directory}
topography input=sfile zmax={topography_zmax} order={topography_order} file={velocity_model_directory}/{velocity_model_name}
rechdf5 infile={station_file_name} outfile={station_output_file_path}

developer reporttiming={reporttiming} cfl={cfl} failonnan={failonnan}
"""


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


def build_image_output_lines(
    image_outputs: list[SW4ImageOutput], simulation_time: float
) -> str:
    lines = []
    for img in image_outputs:
        parts = [
            f"mode={img.mode}",
            f"{img.plane}={img.plane_value}",
            f"file={img.file}",
        ]
        if img.time is not None:
            parts.append(f"time={img.time}")
        if img.time_interval is not None:
            parts.append(f"timeInterval={img.time_interval}")
        if img.cycle is not None:
            parts.append(f"cycle={img.cycle}")
        if img.cycle_interval is not None:
            parts.append(f"cycleInterval={img.cycle_interval}")
        if (
            img.time is None
            and img.time_interval is None
            and img.cycle is None
            and img.cycle_interval is None
        ):
            parts.append(f"time={simulation_time}")
        parts.append(f"precision={img.precision}")
        lines.append(f"imagehdf5 {' '.join(parts)}")
    return "\n".join(lines)


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
        f"refinement zmax={refinement.bottom:.1f}"
        for refinement in refinements[
            :-1
        ]  # Last refinement layer is implicitly the bottom of the domain
    )
    dx = refinements[-1].resolution

    image_output_str = build_image_output_lines(sw4_params.image_outputs, time)

    low_frequency_output = work_directory / "out.h5"
    output_path.write_text(
        SW4_TEMPLATE.format(
            # NOTE: In SW4 x = north, but in the workflow y = north.
            x=y,
            y=x,
            z=depth * 1000.0,
            dx=dx,
            lat=lat,
            lon=lon,
            azimuth=azimuth,
            topography_zmax=topography_zmax,
            velocity_model_name=velocity_model.name,
            velocity_model_directory=velocity_model.parent,
            time=time,
            srf=srf_path,
            station_file_name=station_path,
            work_directory=work_directory,
            station_output_file_path=low_frequency_output.relative_to(work_directory),
            refinement_str=refinements_str,
            verbose=sw4_params.verbose,
            printcycle=sw4_params.printcycle,
            supergrid_gp=sw4_params.supergrid_gp,
            projection_ellps=sw4_params.projection_ellps,
            projection_lon_p=sw4_params.projection_lon_p,
            projection_lat_p=sw4_params.projection_lat_p,
            projection_scale=sw4_params.projection_scale,
            projection_type=sw4_params.projection_type,
            attenuation_maxfreq=sw4_params.attenuation_maxfreq,
            attenuation_phasefreq=sw4_params.attenuation_phasefreq,
            attenuation_nmech=sw4_params.attenuation_nmech,
            topography_order=sw4_params.topography_order,
            reporttiming=int(sw4_params.reporttiming),
            cfl=sw4_params.cfl,
            failonnan=int(sw4_params.failonnan),
            image_output_str=image_output_str,
        )
    )
