from pathlib import Path

from workflow import domain
from workflow.realisations import DomainParameters

SW4_TEMPLATE = """
fileio path={work_directory}

supergrid gp=30
time t={time}

# NZTM equivalent projection without the false northing and easting (which are unsupported by SW4)
grid nx={nx} ny={ny} nz={nz} h={dx} az={azimuth} lon={lon} lat={lat} proj=tmerc datum=NZGD2000 lon_p=173.0 lat_p=0.0 scale=0.9996
rupturehdf5 file={srf}

attenuation maxfreq=1.0

sfile filename={velocity_model_name} directory={velocity_model_directory}
topography input=sfile zmax={topography_zmax} order=3 file={velocity_model_directory}/{velocity_model_name}
rechdf5 infile={station_file_name} outfile={station_output_file_path} writeEvery=1
"""


def generate_sw4_input(
    realisation_ffp: Path,
    station_path: Path,
    srf_path: Path,
    velocity_model: Path,
    work_directory: Path,
    output_path: Path,
) -> None:
    domain_parameters = DomainParameters.read_from_realisation(realisation_ffp)

    nx = domain_parameters.nx
    ny = domain_parameters.ny
    nz = domain_parameters.nz
    azimuth = domain_parameters.domain.great_circle_bearing
    lat, lon = domain_parameters.domain.origin

    depth = domain_parameters.depth
    time = domain_parameters.duration
    refinements = domain.domain_refinements(depth)
    top_refinement = refinements[0]
    dx = top_refinement.resolution
    topography_zmax = top_refinement.bottom

    output_path.write_text(
        SW4_TEMPLATE.format(
            # NOTE: In SW4 x = north, but in the workflow y = north.
            nx=ny,
            ny=nx,
            nz=nz,
            dx=dx,
            lat=lat,
            lon=lon,
            azimuth=azimuth,
            topography_zmax=topography_zmax,
            velocity_file_name=velocity_model.name,
            velocity_model_directory=velocity_model.parent,
            time=time,
            srf=srf_path,
            station_file_name=station_path,
            work_directory=work_directory,
            station_output_file_path=output_path,
        )
    )
