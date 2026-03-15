"""Compress Waveform.

Description
-----------
Compress a broadband waveform HDF5 file using FlacArray compression.

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

import h5py
import numpy as np
import typer
import xarray as xr
from flacarray import FlacArray

from qcore import cli
from workflow import log_utils

app = typer.Typer()


@cli.from_docstring(app)
@log_utils.log_call()
def compress_waveform(
    waveform_ffp: Annotated[Path, typer.Argument(dir_okay=False, exists=True)],
    output_ffp: Annotated[Path, typer.Argument(dir_okay=False, writable=True)],
    precision: Annotated[int, typer.Option(min=1)] = 4,
    level: Annotated[int, typer.Option(min=0, max=8)] = 5,
) -> None:
    """Compress a broadband waveform file using FlacArray.

    Parameters
    ----------
    waveform_ffp : Path
        Path to the input broadband waveform file (HDF5/NetCDF4).
    output_ffp : Path
        Path to the output compressed HDF5 file.
    precision : int, optional
        Number of significant decimal digits to retain in the
        compression. Higher values produce more accurate but larger
        files. Defaults to 4.
    level : int, optional
        FLAC compression level (0-8). Higher values compress more but
        are slower. Defaults to 5.
    """
    broadband = xr.open_dataset(waveform_ffp)

    waveform_data = broadband["waveform"].values
    flac_waveform = FlacArray.from_array(
        waveform_data, precision=precision, level=level
    )

    with h5py.File(output_ffp, "w") as hdf:
        flac_waveform.write_hdf5(hdf.create_group("waveform"))

        for coord_name in broadband.coords:
            coord_data = broadband.coords[coord_name].values
            if np.issubdtype(coord_data.dtype, np.str_):
                coord_data = coord_data.astype(bytes)
            hdf.create_dataset(f"coords/{coord_name}", data=coord_data)

        for attr_name, attr_value in broadband.attrs.items():
            hdf.attrs[attr_name] = attr_value

        hdf.attrs["waveform_dims"] = list(broadband["waveform"].dims)

    broadband.close()


if __name__ == "__main__":
    app()
