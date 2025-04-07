"""Low Frequency Merge.

Description
-----------
Merge EMOD3D seis files into a single xarray dataset.

Inputs
------
1. A realisation file containing EMOD3D parameters.
2. A directory containing EMOD3D output bin files (e.g., *-seis.e3d* files).

Outputs
-------
1. An HDF5 file containing the merged EMOD3D output bin files.

Environment
-----------
Can be run in the cybershake container. Can also be run from your own computer using the `merge-lf` command which is installed after running `pip install workflow@git+https://github.com/ucgmsim/workflow`.

Usage
-----
`merge-lf [OPTIONS] REALISATION_FFP EMOD3D_DIRECTORY OUT_FILE`

For More Help
-------------
See the output of `merge-lf --help`.
"""

from pathlib import Path
from typing import Annotated

import typer

from qcore import cli, timeseries
from workflow import log_utils
from workflow.realisations import EMOD3DParameters

app = typer.Typer()


@cli.from_docstring(app)
def merge_lf(
    realisation_ffp: Annotated[
        Path, typer.Argument(exists=True, dir_okay=False, readable=False)
    ],
    emod3d_outbin: Annotated[
        Path, typer.Argument(exists=True, file_okay=False, readable=True)
    ],
    output_ffp: Annotated[Path, typer.Argument(dir_okay=False, writable=True)],
):
    """Merge EMOD3D seis files into a single xarray dataset.

    Parameters
    ----------
    realisation_ffp : Path
        Path to the realisation file. Must contain EMOD3D parameters.
    emod3d_outbin : Path
        Path to EMOD3D output bin directory (contains *-seis.e3d* files).
    output_ffp : Path
        Path to output file (will be an HDF5 file).

    Examples
    --------
    >>> merge_lf(
        ...     emod3d_outbin="LF/OutBin",
        ...     output_ffp="merged_lf.h5",
        ... )
    >>> # The above code would merge all EMOD3D output bin files in the directory 'LF/OutBin'
    >>> # and save it as 'merged_lf.h5'.
    """
    emod3d_parameters = EMOD3DParameters.read_from_realisation(realisation_ffp)
    dataset = timeseries.read_lfseis_directory(
        emod3d_outbin,
        emod3d_parameters.ts_start,
    )
    dataset.attrs |= emod3d_parameters.to_dict()
    dataset.to_netcdf(output_ffp, engine="h5netcdf", mode="w")
