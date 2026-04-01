"""Convert Velocity Model HDF5 to EMOD3D Format.

Description
-----------
Convert a velocity model in HDF5 format to EMOD3D binary format. This is used
as a separate step after ``generate-velocity-model`` to
decouple the parallel HDF5 generation (high CPU) from the conversion step.

Inputs
------
1. A realisation file (for logging),
2. A work directory containing ``velocity_model.h5`` written by
   ``generate-velocity-model``.

Outputs
-------
EMOD3D binary velocity model files written directly to ``velocity_model_output``.

Environment
-----------
Can be run in the Cybershake container using the ``convert-vm-hdf5-to-emod3d``
command, which is installed after running
``pip install workflow@git+https://github.com/ucgmsim/workflow``.

Usage
-----
``convert-vm-hdf5-to-emod3d [OPTIONS] REALISATION_FFP VELOCITY_MODEL_OUTPUT``

For More Help
-------------
See the output of ``convert-vm-hdf5-to-emod3d --help``.
"""

import os
from pathlib import Path
from typing import Annotated

import typer

from qcore import cli
from velocity_modelling.tools import convert_hdf5_to_emod3d
from workflow import log_utils, realisations

app = typer.Typer()


@cli.from_docstring(app)
@log_utils.log_call()
def convert_vm_hdf5_to_emod3d(
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
    """Convert HDF5 velocity model to EMOD3D binary format.

    Reads velocity_model.h5 from work_directory (produced by
    ``generate-velocity-model``) and converts it to EMOD3D
    binary format, writing the result directly to velocity_model_output.

    Parameters
    ----------
    realisation_ffp : Path
        Path to the JSON realisation file.
    velocity_model_output : Path
        Output directory for EMOD3D binary files.
    work_directory : Path
        Directory containing velocity_model.h5 from the generate step.
        Default is /out.
    """
    os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
    hdf5_output_file = work_directory / "velocity_model.h5"

    if not hdf5_output_file.is_file():
        typer.echo(
            f"Expected HDF5 velocity model file not found: {hdf5_output_file}",
            err=True,
        )
        raise typer.Exit(code=1)

    if not hdf5_output_file.is_file():
        typer.echo(
            f"Expected HDF5 velocity model file not found: {hdf5_output_file}",
            err=True,
        )
        raise typer.Exit(code=1)

    convert_hdf5_to_emod3d.convert_hdf5_to_emod3d(
        hdf5_output_file, velocity_model_output
    )
    realisations.append_log_entry(realisation_ffp)
