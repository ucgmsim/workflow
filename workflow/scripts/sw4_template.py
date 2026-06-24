from pathlib import Path

from workflow import domain
from workflow.realisations import DomainParameters

SW4_TEMPLATE = """
fileio path={work_directory}

supergrid gp=30
time t={time}

grid nx={nx} ny={ny} nz={nz} h={dx}
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
    output_path: Path,
    work_directory: Path,
) -> None:
    domain_parameters = DomainParameters.read_from_realisation(realisation_ffp)
    domain_parameters.extent_x / 
