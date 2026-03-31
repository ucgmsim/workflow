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
from velocity_modelling.constants import WriteFormat
from velocity_modelling.scripts import generate_3d_model
from velocity_modelling.tools import convert_hdf5_to_emod3d
from workflow import log_utils, realisations, utils
from workflow.realisations import (
    DomainParameters,
    RealisationMetadata,
    Resolution,
    VelocityModelParameters,
)

app = typer.Typer()
generate_hdf5_app = typer.Typer()
convert_hdf5_app = typer.Typer()


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


def run_nzcvm_generate(
    nzvm_config_ffp: Path,
    work_directory: Path,
    num_threads: int | None,
) -> None:
    """Generate HDF5 velocity model only (no EMOD3D conversion).

    Parameters
    ----------
    nzvm_config_ffp : Path
        Path to NZVM config to generate from.
    work_directory : Path
        Working directory to output HDF5 to.
    num_threads : int or None
        Number of threads to use (default is inferred by
        `utils.get_available_cores`).
    """
    num_threads = num_threads or utils.get_available_cores()
    generate_3d_model.generate_3d_model(
        nzvm_config_ffp,
        out_dir=work_directory,
        output_format=WriteFormat.HDF5.name,
        np_workers=num_threads,
    )


def run_nzcvm_convert(
    work_directory: Path,
    velocity_model_intermediate_path: Path,
) -> None:
    """Convert existing HDF5 velocity model to EMOD3D binary format.

    Parameters
    ----------
    work_directory : Path
        Working directory containing the HDF5 file (velocity_model.h5).
    velocity_model_intermediate_path : Path
        Output directory for EMOD3D files.
    """
    os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
    hdf5_output_file = work_directory / "velocity_model.h5"
    if not hdf5_output_file.exists():
        raise FileNotFoundError(
            f"HDF5 file not found: {hdf5_output_file}. "
            "Run the generate step first."
        )
    convert_hdf5_to_emod3d.convert_hdf5_to_emod3d(
        hdf5_output_file, velocity_model_intermediate_path
    )


def run_nzcvm(
    nzvm_config_ffp: Path,
    work_directory: Path,
    velocity_model_intermediate_path: Path,
    num_threads: int | None,
) -> None:
    """Generate velocity model with New Zealand Community Velocity Model.

    Parameters
    ----------
    nzvm_config_ffp : Path
        Path to NZVM config to generate from
    work_directory : Path
        Working directory to output HDF5 to
    velocity_model_intermediate_path : Path
        Output directory for EMOD3D files
    num_threads : int | None
        Number of threads to use (default is inferred by
        `utils.get_available_cores`)
    """
    run_nzcvm_generate(nzvm_config_ffp, work_directory, num_threads)
    run_nzcvm_convert(work_directory, velocity_model_intermediate_path)


def _prepare_nzvm_config(
    realisation_ffp: Path,
    work_directory: Path,
) -> tuple[Path, Path]:
    """Read realisation parameters and write NZVM config file.

    Parameters
    ----------
    realisation_ffp : Path
        Path to the JSON realisation file.
    work_directory : Path
        Directory for intermediate output files.

    Returns
    -------
    nzvm_config_path : Path
        Path to the written NZVM config file.
    velocity_model_intermediate_path : Path
        Path to the intermediate velocity model directory.
    """
    domain_parameters = DomainParameters.read_from_realisation(realisation_ffp)
    metadata = RealisationMetadata.read_from_realisation(realisation_ffp)
    velocity_model_parameters = VelocityModelParameters.read_from_realisation_or_defaults(
        realisation_ffp, metadata.defaults_version
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
    return nzvm_config_path, velocity_model_intermediate_path


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
    nzvm_config_path, velocity_model_intermediate_path = _prepare_nzvm_config(
        realisation_ffp, work_directory
    )

    if use_nzcvm:
        run_nzcvm(
            nzvm_config_path,
            work_directory,
            velocity_model_intermediate_path,
            num_threads,
        )
        shutil.rmtree(velocity_model_output, ignore_errors=True)
        shutil.move(velocity_model_intermediate_path, velocity_model_output)
    elif velocity_model_bin_path:
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


@cli.from_docstring(generate_hdf5_app)
@log_utils.log_call()
def generate_velocity_model_hdf5(
    realisation_ffp: Annotated[
        Path, typer.Argument(readable=True, exists=True, dir_okay=False)
    ],
    work_directory: Annotated[
        Path, typer.Option(exists=False, writable=True, file_okay=False)
    ] = Path("/out"),
    num_threads: Annotated[Optional[int], typer.Option(min=1)] = None,
) -> None:
    """Generate HDF5 velocity model only (step 1 of 2 for Cylc split workflow).

    Reads realisation.json, writes nzvm.cfg, then runs the parallel NZCVM
    generation to produce velocity_model.h5 in work_directory.
    Does NOT perform EMOD3D conversion. Run convert-velocity-model-hdf5 next.

    Parameters
    ----------
    realisation_ffp : Path
        Path to the JSON realisation file.
    work_directory : Path
        Directory for intermediate output files (velocity_model.h5 written here).
    num_threads : int or None, optional
        Number of threads for parallel generation.
    """
    work_directory.mkdir(parents=True, exist_ok=True)
    nzvm_config_path, _ = _prepare_nzvm_config(realisation_ffp, work_directory)
    run_nzcvm_generate(nzvm_config_path, work_directory, num_threads)


@cli.from_docstring(convert_hdf5_app)
@log_utils.log_call()
def convert_velocity_model_hdf5(
    realisation_ffp: Annotated[
        Path, typer.Argument(readable=True, exists=True, dir_okay=False)
    ],
    velocity_model_output: Annotated[
        Path, typer.Argument(writable=True, file_okay=False)
    ],
    work_directory: Annotated[
        Path, typer.Option(exists=True, writable=True, file_okay=False)
    ] = Path("/out"),
) -> None:
    """Convert HDF5 velocity model to EMOD3D format (step 2 of 2 for Cylc split workflow).

    Reads velocity_model.h5 from work_directory (written by generate-velocity-model-hdf5),
    converts it to EMOD3D binary format, and moves the result to velocity_model_output.

    Parameters
    ----------
    realisation_ffp : Path
        Path to the JSON realisation file.
    velocity_model_output : Path
        Final output directory for EMOD3D binary files.
    work_directory : Path
        Directory containing velocity_model.h5 from the generate step.
    """
    velocity_model_intermediate_path = work_directory / "Velocity_Model"
    run_nzcvm_convert(work_directory, velocity_model_intermediate_path)
    shutil.rmtree(velocity_model_output, ignore_errors=True)
    shutil.move(velocity_model_intermediate_path, velocity_model_output)
    realisations.append_log_entry(realisation_ffp)
