from pathlib import Path

import typer

from qcore import cli
from workflow import domain
from workflow.realisations import DomainParameters

app = typer.Typer()
SW4_TEMPLATE = """
fileio path={work_directory}

supergrid gp=30
time t={time}

# NZTM equivalent projection without the false northing and easting (which are unsupported by SW4)
# NOTE: x=north, y=east, z=down
grid x={x} y={y} z={z} h={dx} az={azimuth} lon={lon} lat={lat} proj=tmerc ellps=GRS80 lon_p=173.0 lat_p=0.0 scale=0.9996
rupturehdf5 file={srf}

{refinement_str}

attenuation maxfreq=1.0

sfile filename={velocity_model_name} directory={velocity_model_directory}
topography input=sfile zmax={topography_zmax} order=3 file={velocity_model_directory}/{velocity_model_name}
rechdf5 infile={station_file_name} outfile={station_output_file_path} writeEvery=1
"""


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
    azimuth = domain_parameters.domain.bearing
    # In SW4 the domain should always begin from the bottom-left (which is corners[0] by construction)
    lat, lon = domain_parameters.domain.corners[0]

    depth = domain_parameters.depth
    time = domain_parameters.duration
    refinements = domain.domain_refinements(depth)
    refinements_str = '\n'.join(
        f'refinement zmax={refinement.bottom:.1f}'
        for refinement in refinements[:-1]
    )
    bottom_refinement = refinements[-1]
    dx = bottom_refinement.resolution
    top_refinement = refinements[0]
    topography_zmax = top_refinement.bottom
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
            station_file_name=output_path,
            work_directory=work_directory,
            station_output_file_path=low_frequency_output,
            refinement_str=refinements_str
        )
    )
