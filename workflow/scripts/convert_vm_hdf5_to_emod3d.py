import os
import shutil
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
    """Convert HDF5 velocity model to EMOD3D format (step 2 of 2 for split Cylc workflow).

    Reads velocity_model.h5 from work_directory (written by
    generate-velocity-model-hdf5), converts it to EMOD3D binary format, and
    moves the result to velocity_model_output.

    Parameters
    ----------
    realisation_ffp : Path
        Path to the JSON realisation file.
    velocity_model_output : Path
        Final output directory for EMOD3D binary files.
    work_directory : Path
        Directory containing velocity_model.h5 from the generate step.
    """
    os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
    hdf5_output_file = work_directory / "velocity_model.h5"
    velocity_model_intermediate_path = work_directory / "Velocity_Model"
    convert_hdf5_to_emod3d.convert_hdf5_to_emod3d(
        hdf5_output_file, velocity_model_intermediate_path
    )
    shutil.rmtree(velocity_model_output, ignore_errors=True)
    shutil.move(velocity_model_intermediate_path, velocity_model_output)
    realisations.append_log_entry(realisation_ffp)

