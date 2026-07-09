from pathlib import Path

import h5py
import typer
from nzcvm.formats import sfile

from qcore import cli
from workflow import domain
from workflow.realisations import DomainParameters

app = typer.Typer()
SW4_TEMPLATE = """
fileio path={work_directory} verbose=2 printcycle=10

supergrid gp=30
time t={time}

# NZTM equivalent projection without the false northing and easting (which are unsupported by SW4)
# NOTE: x=north, y=east, z=down
grid x={x} y={y} z={z} h={dx} az={azimuth} lon={lon} lat={lat} proj=tmerc ellps=GRS80 lon_p=173.0 lat_p=0.0 scale=0.9996
rupturehdf5 file={srf}

{refinement_str}

attenuation maxfreq=10.0 phasefreq=2.5 nmech=3

imagehdf5 mode=topo z=0 file=topo cycle=0 precision=float
imagehdf5 mode=grid z=0 file=grid cycle=0 precision=float

imagehdf5 mode=p   z=0 file=surf_vp  cycle=0 precision=float
imagehdf5 mode=s   z=0 file=surf_vs  cycle=0 precision=float
imagehdf5 mode=rho z=0 file=surf_rho cycle=0 precision=float

imagehdf5 mode=mag    z=0 file=surf_mag    timeInterval=0.5 precision=float
imagehdf5 mode=velmag z=0 file=surf_velmag timeInterval=0.5 precision=float
imagehdf5 mode=uz     z=0 file=surf_uz     timeInterval=0.5 precision=float

imagehdf5 mode=hmax z=0 file=surf_hmax time={time} precision=float
imagehdf5 mode=vmax z=0 file=surf_vmax time={time} precision=float

sfile filename={velocity_model_name} directory={velocity_model_directory}
topography input=sfile zmax={topography_zmax} order=3 file={velocity_model_directory}/{velocity_model_name}
rechdf5 infile={station_file_name} outfile={station_output_file_path}

developer reporttiming=1 ctol=1e-4 failonnan=1
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
    domain_parameters = DomainParameters.read_from_realisation(realisation_ffp)

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
    refinements = domain.domain_refinements(depth)

    refinements, topography_zmax = domain.adjust_for_topography(
        refinements, topography_zmax
    )
    refinements = sorted(refinements, key=lambda r: r.bottom)
    # Per the SW4 User Guide, the supergrid sponge (30 gridpoints by default) at the bottom of the domain
    # must be contained in the bottom refinement.
    supergrid_padding = 30
    supergrid_width = supergrid_padding * refinements[-1].resolution
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
        )
    )
