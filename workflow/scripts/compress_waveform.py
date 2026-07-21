"""Compress Waveform.

Description
-----------
Compress a broadband waveform HDF5 file using FLAC compression.

Inputs
------
1. A broadband waveform file (HDF5/NetCDF4 format, output of ``bb-sim``).

Outputs
-------
A compressed waveform file in HDF5 format with FlacArray-encoded waveform data.

Environment
-----------
Can be run in the cybershake container. Can also be run from your own
computer using the ``compress-waveform`` command which is installed after running
``pip install workflow@git+https://github.com/ucgmsim/workflow``.

Usage
-----
``compress-waveform WAVEFORM_FFP OUTPUT_FFP``

For More Help
-------------
See the output of ``compress-waveform --help``.
"""

from pathlib import Path
from typing import Annotated

import flacarray.hdf5
import h5py
import typer
import xarray as xr

from qcore import cli
from workflow import log_utils

app = typer.Typer()


@cli.from_docstring(app)
@log_utils.log_call()
def compress_waveform(
    waveform_ffp: Annotated[Path, typer.Argument(dir_okay=False, exists=True)],
    output_ffp: Annotated[Path, typer.Argument(dir_okay=False, writable=True)],
    level: Annotated[int, typer.Option(min=0, max=8)] = 5,
    precision: Annotated[int, typer.Option(min=1)] = 4,
) -> None:
    """Compress a broadband waveform file using FLAC.

    Parameters
    ----------
    waveform_ffp : Path
        Path to the input broadband waveform file (HDF5/NetCDF4).
    output_ffp : Path
        Path to the output compressed HDF5 file.
    level : int, optional
        FLAC compression level (0-8). Higher values compress more but
        are slower. Defaults to 5.
    precision : int, optional
        FLAC precision level (in significant digits of input data). Higher values compress less but
        have more precision. Defaults to 4.
    """
    with (
        xr.open_dataset(waveform_ffp, engine="h5netcdf") as broadband,
    ):
        broadband.drop_vars("waveform").to_netcdf(output_ffp, engine="h5netcdf")
        with h5py.File(output_ffp, "a") as hdf:
            group = hdf.create_group("_flac_compressed_waveform")
            group.attrs["flac_array"] = True
            group.attrs["name"] = "waveform"
            group.attrs["shape"] = broadband.waveform.shape
            group.attrs["dims"] = broadband.waveform.dims
            group.attrs["dtype"] = str(broadband.waveform.dtype)

            flacarray.hdf5.write_array(
                broadband.waveform.values,
                group,
                precision=precision,
                level=level,
                use_threads=True,
            )


if __name__ == "__main__":
    app()
