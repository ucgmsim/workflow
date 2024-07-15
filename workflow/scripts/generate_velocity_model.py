import os
import shutil
import subprocess
from pathlib import Path
from typing import Annotated

import typer

from workflow import realisations
from workflow.realisations import DomainParameters, VelocityModelParameters


def write_nzvm_config(
    domain_parameters: DomainParameters,
    velocity_model_parameters: VelocityModelParameters,
    nzvm_config_path: Path,
    output_path: Path,
) -> None:
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
                ]
            )
        )


def run_nzvm(nzvm_binary_ffp: Path, nzvm_config_ffp: Path, num_threads: int) -> None:
    environment = os.environ.copy()
    environment["OMP_NUM_THREADS"] = str(num_threads)
    subprocess.run(
        [str(nzvm_binary_ffp), str(nzvm_config_ffp)],
        cwd=nzvm_binary_ffp.parent,
        env=environment,
        capture_output=True,
    )


def copy_nzvm_files(scratch_directory: Path, velocity_model_output: Path) -> None:
    generated_velocity_model_files = scratch_directory / "Velocity_Model"
    shutil.copytree(generated_velocity_model_files, velocity_model_output)


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
        ),
    ],
    velocity_model_bin_path: Annotated[
        Path, typer.Option(help="Path to NZVM binary.", exists=True, readable=True)
    ] = "/Velocity-Model/NZVM",
    scratch_directory: Annotated[
        Path,
        typer.Option(
            help="Directory to intermediate output files to.",
            exists=True,
            writable=True,
            file_okay=False,
        ),
    ] = "/out/Velocity_Model",
) -> None:
    """Generate velocity model for a realisation."""
    domain_parameters: DomainParameters = realisations.read_config_from_realisation(
        DomainParameters, realisation_filepath
    )
    velocity_model_parameters: VelocityModelParameters = (
        realisations.read_config_from_realisation(
            VelocityModelParameters, realisation_filepath
        )
    )
    nzvm_config_path = scratch_directory / "nzvm.cfg"
    write_nzvm_config(
        domain_parameters,
        velocity_model_parameters,
        nzvm_config_path,
        output_path=scratch_directory,
    )
    run_nzvm(velocity_model_bin_path, nzvm_config_path)
    copy_nzvm_files(scratch_directory, velocity_model_output)


def main():
    typer.run(generate_velocity_model)


if __name__ == "__main__":
    main()
