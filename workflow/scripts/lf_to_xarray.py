#!/usr/bin/env python
"""Low-frequency output merger.

Description
-----------
Merges low-frequency outputs into one xarray dataset.

Inputs
------
1. A low-frequency output directory.

Outputs
-------
1. A combined LF waveform output containing ground acceleration data for each station.

Environment
-----------
Can be run in the cybershake container. Can also be run from your own computer using the `lf-to-xarray` command which is installed after running `pip install workflow@git+https://github.com/ucgmsim/workflow`.

Usage
-----
`lf-to-xarray [OPTIONS] OUTBIN_DIRECTORY OUTPUT_FFP`

For More Help
-------------
See the output of `lf-to-xarray --help`.
"""

from pathlib import Path

import h5py
import typer
import xarray as xr

from qcore import cli, timeseries
from workflow import log_utils

app = typer.Typer()


def convert_sw4_station_recording(station_file: Path) -> xr.Dataset:
    global_npts = None
    with h5py.File(station_file, "r") as f:
        for group in f:
            if "NPTS" not in group:
                continue
            npts = int(group["NPTS"][0])
            if npts is not None and npts != global_npts:
                raise ValueError(f"SW4 output is corrupted: {npts=} but {global_npts=}")


@cli.from_docstring(app)
@log_utils.log_call()
def convert_lf_to_xarray_dataset(lfseis_directory: Path, output_ffp: Path) -> None:
    """Merge low-frequency outputs into an xarray dataset.

    Parameters
    ----------
    lfseis_directory : Path
        Directory containing station seismogram outputs.
    output_ffp : Path
        Path to write the xarray dataset
    """
    lf_dataset = timeseries.read_lfseis_directory(lfseis_directory)
    lf_dataset.to_netcdf(output_ffp, engine="h5netcdf")


if __name__ == "__main__":
    app()
