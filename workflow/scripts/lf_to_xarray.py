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

from enum import StrEnum, auto
from pathlib import Path

import dask.array as da
import h5py
import numpy as np
import typer
import xarray as xr

from qcore import cli, timeseries
from workflow import log_utils

app = typer.Typer()

CMS = 100.0
# Unit to convert m/s to cm/s


def convert_sw4_station_recording(handle: h5py.File) -> xr.Dataset:
    """Convert SW4 station recording to an xarray dataset.

    Parameters
    ----------
    handle: h5py.File
        Handle for HDF5 file.

    Returns
    -------
    xr.Dataset
        xarray dataset constructed from file.

    Raises
    ------
    RuntimeError
        If the HDF5 file is not in the format expected for an SW4 recording file
        (see Section 12.9 of the SW4 User Guide).
    """
    global_npts = None

    dt = np.float32(handle["DELTA"][()])
    xs = []
    ys = []
    zs = []
    stations = []
    attributes = {"start_sec": 0.0, "dt": dt}
    for station_name, group in handle.items():
        if "NPTS" not in group:
            continue
        npts = int(group["NPTS"][()])
        if global_npts is not None and npts != global_npts:
            raise RuntimeError(f"SW4 output is corrupted: {npts=} but {global_npts=}")
        global_npts = npts
        # Dask arrays here ensure that data is read chunkwise from the HDF5 file
        # without putting it all in-memory.
        x = da.from_array(group["X"], chunks=-1)
        xs.append(x * CMS)
        y = da.from_array(group["Y"], chunks=-1)
        ys.append(y * CMS)
        z = da.from_array(group["Z"], chunks=-1)
        zs.append(z * CMS)
        stations.append(station_name)
    if global_npts is None:
        raise RuntimeError(
            "No valid station recordings found in file. Are you sure this is an SW4 station file? Use `h5ls` to check the file structure."
        )
    time = np.arange(global_npts) * dt
    x = da.stack(xs, axis=0)
    y = da.stack(ys, axis=0)
    z = da.stack(zs, axis=0)
    waveform = da.stack((x, y, z), axis=0)
    return xr.Dataset(
        {"waveform": (("component", "station", "time"), waveform)},
        coords=dict(component=["x", "y", "z"], station=stations, time=time),
        attrs=attributes,
    )


class Format(StrEnum):
    """Input low frequency file format."""

    SW4 = auto()
    """SW4 HDF5 station recording."""
    EMOD3D = auto()
    """EMOD3D LFSeis directory."""


@cli.from_docstring(app)
@log_utils.log_call()
def convert_lf_to_xarray_dataset(
    low_frequency_path: Path, output_ffp: Path, format: Format = Format.EMOD3D
) -> None:
    """Merge low-frequency outputs into an xarray dataset.

    Parameters
    ----------
    lfseis_directory : Path
        Directory containing station seismogram outputs.
    output_ffp : Path
        Path to write the xarray dataset
    """
    match format:
        case Format.EMOD3D if low_frequency_path.is_dir():
            lf_dataset = timeseries.read_lfseis_directory(low_frequency_path)
            lf_dataset.to_netcdf(output_ffp, engine="h5netcdf")
        case Format.EMOD3D:
            raise ValueError("EMOD3D format requires directory to LFSeis files")
        case Format.SW4 if low_frequency_path.is_file():
            with h5py.File(low_frequency_path, "r") as f:
                lf_dataset = convert_sw4_station_recording(f)
                lf_dataset = lf_dataset.chunk(
                    {"component": "auto", "station": "auto", "time": -1}
                )
                lf_dataset.to_netcdf(output_ffp, engine="h5netcdf")
        case Format.SW4:
            raise ValueError("SW4 format requires station recording file.")


if __name__ == "__main__":
    app()
