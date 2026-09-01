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
Can be run in the Cybershake container. Can also be run from your own computer using the `generate-velocity-model` command which is installed after running `pip install workflow@git+https://github.com/ucgmsim/workflow`.
If you are executing on your own computer you also need to specify the `NZVM` path (`--velocity-model-bin-path`) and the work directory (`--work-directory`).

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
from typing import Annotated

import typer

from qcore import cli
from velocity_modelling.constants import WriteFormat
from velocity_modelling.scripts import generate_3d_model
from workflow import log_utils, realisations, utils
from workflow.realisations import (
    DomainParameters,
    RealisationMetadata,
    Resolution,
    VelocityModelParameters,
)

app = typer.Typer()


def write_nzvm_config(
    resolution: Resolution,
    domain_parameters: DomainParameters,
    velocity_model_parameters: VelocityModelParameters,
    output_path: Path,
    nzvm_config_path: Path,
) -> None:
    """Write NZVM configuration file.

    Parameters
    ----------
    resolution : Resolution
        Resolution parameters extracted from realisation JSON.
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
                    f"EXTENT_Z_SPACING={resolution.resolution}",
                    f"EXTENT_LATLON_SPACING={resolution.resolution}",
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


def run_nzcvm(
    nzvm_config_ffp: Path,
    work_directory: Path,
    num_threads: int | None,
) -> None:
    """Generate velocity model with New Zealand Community Velocity Model.

    Parameters
    ----------
    nzvm_config_ffp : Path
        Path to NZVM config to generate from
    work_directory : Path
        Working directory to output HDF5 to
    num_threads : int | None
        Number of threads to use (default is inferred by
        `utils.get_available_cores`)
    """

    num_threads = num_threads or utils.get_available_cores()
    generate_3d_model.generate_3d_model(
        nzvm_config_ffp,
        out_dir=work_directory,
        output_format=WriteFormat.HDF5.name,
        np_workers=num_threads,
    )


@cli.from_docstring(app)
@log_utils.log_call()
def generate_velocity_model(
    realisation_ffp: Annotated[
        Path, typer.Argument(readable=True, exists=True, dir_okay=False)
    ],
    velocity_model_output: Annotated[
        Path | None, typer.Argument(writable=True, file_okay=False)
    ] = None,
    velocity_model_bin_path: Annotated[
        Path | None, typer.Option(exists=True, readable=True)
    ] = None,
    work_directory: Annotated[
        Path, typer.Option(exists=False, writable=True, file_okay=False)
    ] = Path("/out"),
    use_nzcvm: Annotated[bool, typer.Option()] = False,
    num_threads: Annotated[int | None, typer.Option(min=1)] = None,
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
    velocity_model_output : Path, optional
        Path to the directory where the generated velocity model will be saved.
        Required when using the NZVM binary (``--no-use-nzcvm``). Not used
        when ``--use-nzcvm`` is set; EMOD3D conversion is handled by the
        separate ``convert-vm-hdf5-to-emod3d`` command in that case.
    velocity_model_bin_path : Path, optional
        Path to the NZVM binary.
    work_directory : Path, optional
        Directory for intermediate output files.
    use_nzcvm : bool, optional
        If True, use the NZCVM Python package instead of the NZVM binary.
        The velocity model is written as HDF5 to ``work_directory``; use
        ``convert-vm-hdf5-to-emod3d`` afterwards to produce EMOD3D files.
        Default is False.
    num_threads : int or None, optional
        Number of threads to use for velocity model generation. Use None for inferred thread count.

    Returns
    -------
    None
        The function does not return any value. It writes the generated velocity model to the specified output directory.
    """
    domain_parameters = DomainParameters.read_from_realisation(realisation_ffp)
    metadata = RealisationMetadata.read_from_realisation(realisation_ffp)
    velocity_model_parameters = (
        VelocityModelParameters.read_from_realisation_or_defaults(
            realisation_ffp, metadata.defaults_version
        )
    )
    resolution = Resolution.read_from_realisation_or_defaults(
        realisation_ffp, metadata.defaults_version
    )
    nzvm_config_path = work_directory / "nzvm.cfg"
    velocity_model_intermediate_path = work_directory / "Velocity_Model"

    write_nzvm_config(
        resolution,
        domain_parameters,
        velocity_model_parameters,
        velocity_model_intermediate_path,
        nzvm_config_path,
    )

    if use_nzcvm:
        run_nzcvm(
            nzvm_config_path,
            work_directory,
            num_threads,
        )
    elif velocity_model_bin_path:
        if velocity_model_output is None:
            raise typer.BadParameter(
                "VELOCITY_MODEL_OUTPUT is required when using the NZVM binary."
            )
        run_nzvm(velocity_model_bin_path, nzvm_config_path, num_threads)
        shutil.rmtree(velocity_model_output, ignore_errors=True)
        shutil.move(
            velocity_model_intermediate_path / "Velocity_Model", velocity_model_output
        )
    else:
        raise ValueError(
            "If not using nzcvm, you must specify the path to the NZVM binary."
        )

    realisations.append_log_entry(realisation_ffp)
