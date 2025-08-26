"""Velocity Model Generation.

Description
-----------
Generate a velocity model for a domain.

Inputs
------
A realisation file containing:

1. Domain parameters,
2. Velocity model parameters.

Outputs
-------
A directory consisting of [velocity model files](https://wiki.canterbury.ac.nz/display/QuakeCore/File+Formats+Used+In+Ground+Motion+Simulation#FileFormatsUsedInGroundMotionSimulation-VelocityModelFiles).

Environment
-----------
Can be run in the Cybershake container. Can also be run from your own computer using the `generate-velocity-model` command which is installed after running `pip install workflow@git+https://github.com/ucgmsim/workflow`. If you are executing on your own computer you also need to specify the `NZVM` path (`--velocity-model-bin-path`) and the work directory (`--work-directory`).

Usage
-----
`generate-velocity-model [OPTIONS] REALISATION_FFP VELOCITY_MODEL_OUTPUT`

For More Help
-------------
See the output of `generate-velocity-model --help`.

Visualisation
-------------
The velocity modelling repository contains some tools to plot velocity models. See `velocity_modelling.scripts.plot_velocity_model`.
"""

import os
import shutil
import subprocess
from pathlib import Path
from typing import Annotated, Optional

import typer

from qcore import cli
from workflow import log_utils, realisations, utils
from workflow.realisations import (
    DomainParameters,
    RealisationMetadata,
    VelocityModelParameters,
)

app = typer.Typer()


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
                    f"ORIGIN_ROT={domain_parameters.domain.great_circle_bearing}",
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


def run_nzvm(
    nzvm_binary_ffp: Path, nzvm_config_ffp: Path, num_threads: int | None
) -> None:
    """Run NZVM executable with specified configuration.

    Parameters
    ----------
    nzvm_binary_ffp : Path
        Path to the NZVM binary executable.
    nzvm_config_ffp : Path
        Path to the NZVM configuration file.
    num_threads : int or None
        Number of threads to use for velocity model generation. Use
        None for inferred thread count.
    """
    environment = os.environ.copy()

    if num_threads is None:
        environment["OMP_NUM_THREADS"] = str(utils.get_available_cores())
    elif num_threads:
        environment["OMP_NUM_THREADS"] = str(num_threads)

    subprocess.check_call(
        [str(nzvm_binary_ffp), str(nzvm_config_ffp.resolve())],
        cwd=nzvm_binary_ffp.parent,
        env=environment,
        stderr=subprocess.STDOUT,
    )


def run_nzcvm(nzvm_config_ffp: Path) -> None:
    """Run NZCVM executable with specified configuration.

    Parameters
    ----------
    nzvm_config_ffp : Path
        Path to the NZVM-format configuration file.
    """
    from velocity_modelling.scripts import nzcvm

    nzcvm.generate_velocity_model(nzvm_config_ffp)


@cli.from_docstring(app)
@log_utils.log_call()
def generate_velocity_model(
    realisation_ffp: Annotated[
        Path, typer.Argument(readable=True, exists=True, dir_okay=False)
    ],
    velocity_model_output: Annotated[
        Path, typer.Argument(writable=True, file_okay=False, exists=False)
    ],
    velocity_model_bin_path: Annotated[
        Path | None, typer.Option(exists=True, readable=True)
    ] = None,
    work_directory: Annotated[
        Path, typer.Option(exists=False, writable=True, file_okay=False)
    ] = Path("/out"),
    use_nzcvm: Annotated[bool, typer.Option()] = False,
    num_threads: Annotated[Optional[int], typer.Option(min=1)] = None,
) -> None:
    """
    Generate a velocity model for a seismic realisation using NZVM.

    This function generates a configuration file
    for the velocity model binary (NZVM), runs NZVM to produce the velocity
    model, and saves the output to the specified directory.

    Parameters
    ----------
    realisation_ffp : Path
        Path to the JSON file containing the seismic realisation parameters.
    velocity_model_output : Path
        Path to the directory where the generated velocity model will be saved.
    velocity_model_bin_path : Path, optional
        Path to the NZVM binary.
    work_directory : Path, optional
        Directory for intermediate output files.
    use_nzcvm : bool, optional
        If True, use the NZCVM Python package instead of the NZVM binary. Default is False.
    num_threads : int or None, optional
        Number of threads to use for velocity model generation. Use None for inferred thread count.

    Returns
    -------
    None
        The function does not return any value. It writes the generated velocity model to the specified output directory.
    """
    if not use_nzcvm and not velocity_model_bin_path:
        raise ValueError(
            "If not using nzcvm, you must specify the path to the NZVM binary."
        )

    domain_parameters = DomainParameters.read_from_realisation(realisation_ffp)
    metadata = RealisationMetadata.read_from_realisation(realisation_ffp)
    velocity_model_parameters = (
        VelocityModelParameters.read_from_realisation_or_defaults(
            realisation_ffp, metadata.defaults_version
        )
    )
    nzvm_config_path = work_directory / "nzvm.cfg"
    velocity_model_intermediate_path = work_directory / "Velocity_Model"

    write_nzvm_config(
        domain_parameters,
        velocity_model_parameters,
        velocity_model_intermediate_path,
        nzvm_config_path,
    )
    if use_nzcvm:
        run_nzcvm(nzvm_config_path)
    else:
        run_nzvm(velocity_model_bin_path, nzvm_config_path, num_threads)
    shutil.copytree(
        velocity_model_intermediate_path / "Velocity_Model", velocity_model_output
    )
    realisations.append_log_entry(realisation_ffp)
