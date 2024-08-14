"""
Generate Velocity Model Script.

This script generates a velocity model using NZVM based on parameters extracted from a realisation JSON file.

Usage:
------

To run the script, use the command line:
$ python generate_velocity_model.py <realisation_filepath> <velocity_model_output> [OPTIONS]

"""

import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Annotated

import typer

from workflow import realisations
from workflow.realisations import DomainParameters, VelocityModelParameters


def write_nzvm_config(
    domain_parameters: DomainParameters,
    velocity_model_parameters: VelocityModelParameters,
    output_path: Path,
    nzvm_config_path: Path,
) -> None:
    """Write NZVM configuration file.

    Parameters
    ----------
    domain_parameters : DomainParameters
        Domain parameters extracted from realisation JSON.
    velocity_model_parameters : VelocityModelParameters
        Velocity model parameters extracted from realisation JSON.
    output_path : Path
        Path to the output directory for generated velocity model files.
    nzvm_config_path : Path
        Path to the NZVM configuration file to be written.

    """
    with open(nzvm_config_path, mode="w", encoding="utf-8") as nzvm_file_handle:
        nzvm_file_handle.write(
            "\n".join(
                [
                    "CALL_TYPE=GENERATE_VELOCITY_MOD",
                    f"MODEL_VERSION={velocity_model_parameters.version}",
                    f"OUTPUT_DIR={output_path}",
                    f"ORIGIN_LAT={domain_parameters.domain.origin[0]}",
                    f"ORIGIN_LON={domain_parameters.domain.origin[1]}",
                    f"ORIGIN_ROT={domain_parameters.domain.bearing}",
                    f"EXTENT_X={domain_parameters.domain.extent_x}",
                    f"EXTENT_Y={domain_parameters.domain.extent_y}",
                    "EXTENT_ZMIN=0",  # TODO: CHANGE THIS
                    f"EXTENT_ZMAX={domain_parameters.depth}",
                    f"EXTENT_Z_SPACING={domain_parameters.resolution}",
                    f"EXTENT_LATLON_SPACING={domain_parameters.resolution}",
                    f"MIN_VS={velocity_model_parameters.min_vs}",
                    f"TOPO_TYPE={velocity_model_parameters.topo_type}",
                    "",
                ]
            )
        )


def run_nzvm(nzvm_binary_ffp: Path, nzvm_config_ffp: Path, num_threads: int) -> None:
    """Run NZVM executable with specified configuration.

    Parameters
    ----------
    nzvm_binary_ffp : Path
        Path to the NZVM binary executable.
    nzvm_config_ffp : Path
        Path to the NZVM configuration file.
    num_threads : int
        Number of threads to use for velocity model generation.
    """
    environment = os.environ.copy()
    environment["OMP_NUM_THREADS"] = str(num_threads)
    subprocess.run(
        [str(nzvm_binary_ffp), str(nzvm_config_ffp)],
        cwd=nzvm_binary_ffp.parent,
        env=environment,
        stdout=sys.stdout,
        stderr=sys.stderr,
        check=True,
    )


def generate_velocity_model(
    realisation_filepath: Annotated[
        Path,
        typer.Argument(
            help="Path to realisation JSON file.",
            readable=True,
            exists=True,
            dir_okay=False,
        ),
    ],
    velocity_model_output: Annotated[
        Path,
        typer.Argument(
            help="Path to output velocity model directory.",
            writable=True,
            file_okay=False,
            exists=False,
        ),
    ],
    velocity_model_bin_path: Annotated[
        Path, typer.Option(help="Path to NZVM binary.", exists=True, readable=True)
    ] = "/Velocity-Model/NZVM",
    scratch_directory: Annotated[
        Path,
        typer.Option(
            help="Directory to intermediate output files to.",
            exists=False,
            writable=True,
            file_okay=False,
        ),
    ] = "/out",
    num_threads: Annotated[
        int,
        typer.Option(
            help="Number of threads to use for velocity model generation (-1 for inferred thread count).",
            min=-1,
        ),
    ] = -1,
) -> None:
    """Generate a velocity model for a seismic realisation using NZVM.

    This program reads parameters from a JSON file describing the seismic
    domain and velocity model specifications, generates a configuration file
    for NZVM (New Zealand Velocity Model), runs NZVM to generate the velocity
    model, and saves the output to a specified directory.

    Example usage:
    $ generate_velocity_model path/to/realisation.json /path/to/save/velocity_model
    """
    domain_parameters = DomainParameters.read_from_realisation(realisation_filepath)
    velocity_model_parameters = VelocityModelParameters.read_from_realisation(realisation_filepath)
    nzvm_config_path = scratch_directory / "nzvm.cfg"
    velocity_model_intermediate_path = scratch_directory / "Velocity_Model"

    write_nzvm_config(
        domain_parameters,
        velocity_model_parameters,
        velocity_model_intermediate_path,
        nzvm_config_path,
    )
    run_nzvm(velocity_model_bin_path, nzvm_config_path, num_threads)
    shutil.copytree(
        velocity_model_intermediate_path / "Velocity_Model", velocity_model_output
    )


def main():
    typer.run(generate_velocity_model)


if __name__ == "__main__":
    main()
