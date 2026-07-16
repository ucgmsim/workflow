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
from typing import Annotated

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

TARGET_CHUNK_BYTES = 128 * 2**20
# Target size of a dask chunk (all components for a batch of stations).


def _read_station_batch(
    stations: xr.DataArray,
    sw4_ffp: Path,
    time: xr.DataArray,
    component: xr.DataArray,
) -> xr.DataArray:
    """Read waveforms for a batch of stations from an SW4 recording file.

    Parameters
    ----------
    sw4_ffp : Path
        Path to the SW4 HDF5 station recording file.
    station_names : list[str]
        Names of the station groups to read.

    Returns
    -------
    np.ndarray
        Waveforms in cm/s with shape (3, len(station_names), npts),
        components ordered x, y, z.
    """
    waveforms = np.empty((len(component), len(stations), len(time)), dtype=np.float32)
    with h5py.File(sw4_ffp, "r") as handle:
        for i, station_name in enumerate(stations):
            group = handle[station_name.item()]
            x_key = "EW" if "EW" in group else "X"
            y_key = "NS" if x_key == "EW" else "Y"
            waveforms[0, i] = group[x_key][:]
            waveforms[1, i] = group[y_key][:]
            waveforms[2, i] = group["UP"][:]

    return xr.DataArray(
        waveforms,
        dims=["component", "station", "time"],
        coords=dict(time=time, component=component, station=stations.values),
    )


def read_station_metadata(sw4_ffp: Path) -> xr.Dataset:
    """Initialise an xarray dataset using metadata read from the station recording file.

    Parameters
    ----------
    sw4_ffp: Path
        Path to SW4 recording file.

    Returns
    -------
    xr.Dataset
        Xarray dataset with initialised coordinate arrays and attributes.

    Raises
    ------
    RuntimeError
        If the HDF5 file is not in the format expected for an SW4 recording file
        (see Section 12.9 of the SW4 User Guide).
    """
    global_npts = None
    stations = []
    latitudes = []
    longitudes = []

    with h5py.File(sw4_ffp, "r") as handle:
        dt = np.float32(handle["DELTA"][:].squeeze())
        for station_name, group in handle.items():
            if "NPTS" not in group:
                continue
            npts = int(group["NPTS"][:].squeeze())
            if global_npts is not None and npts != global_npts:
                raise RuntimeError(
                    f"SW4 output is corrupted: {npts=} but {global_npts=}"
                )
            global_npts = npts
            stations.append(station_name)

            latitude, longitude, _ = group["STLA,STLO,STDP"][:]
            latitudes.append(latitude)
            longitudes.append(longitude)

    if global_npts is None:
        raise RuntimeError(
            "No valid station recordings found in file. Are you sure this is an SW4 station file? Use `h5ls` to check the file structure."
        )

    time = np.arange(global_npts) * dt
    return xr.Dataset(
        dict(
            latitude=("station", latitudes),
            longitude=("station", longitudes),
        ),
        coords=dict(station=stations, component=["x", "y", "z"], time=time),
        attrs=dict(dt=dt, nt=global_npts),
    )


def _template_waveform(dset: xr.Dataset) -> xr.DataArray:
    return xr.DataArray(
        da.empty(
            (
                len(dset.coords["component"]),
                len(dset.coords["station"]),
                len(dset.coords["time"]),
            ),
            dtype=np.float32,
        ),
        dims=["component", "station", "time"],
        coords=dset.coords,
    )


def convert_sw4_station_recording(sw4_ffp: Path) -> xr.Dataset:
    """Convert SW4 station recording to an xarray dataset.

    Parameters
    ----------
    sw4_ffp : Path
        Path to the SW4 HDF5 station recording file.

    Returns
    -------
    xr.Dataset
        An xarray dataset lazily constructed from the HDF5 file. Waveform data
        is read in batches of stations when the dataset is computed or written.
    """
    dset = read_station_metadata(sw4_ffp)
    batch_size = max(
        1,
        TARGET_CHUNK_BYTES
        // (
            len(dset.coords["component"])
            * len(dset.coords["time"])
            * np.float32().itemsize
        ),
    )
    chunked_stations = xr.DataArray(dset["station"].values).chunk(batch_size)
    waveform = xr.map_blocks(
        _read_station_batch,
        chunked_stations,
        kwargs=dict(time=dset["time"], component=dset["component"], sw4_ffp=sw4_ffp),
        template=_template_waveform(dset),
    )

    waveform = (waveform * CMS).differentiate("time")
    dset["waveform"] = waveform
    dset.attrs["units"] = "cm/s^2"

    return dset


class Format(StrEnum):
    """Input low frequency file format."""

    SW4 = auto()
    """SW4 HDF5 station recording."""
    EMOD3D = auto()
    """EMOD3D LFSeis directory."""


@cli.from_docstring(app)
@log_utils.log_call()
def convert_lf_to_xarray_dataset(
    low_frequency_path: Annotated[Path, typer.Argument(exists=True)],
    output_ffp: Annotated[Path, typer.Argument(writable=True, dir_okay=False)],
    format: Format = Format.EMOD3D,
) -> None:
    """Merge low-frequency outputs into an xarray dataset.

    Parameters
    ----------
    low_frequency_path : Path
        Directory containing station seismogram outputs.
    output_ffp : Path
        Path to write the xarray dataset
    format : Format, optional
        Format for the low-frequency inputs (EMOD3D or SW4). If format is SW4,
        the low frequency path should be an HDF5 file in the SW4 station format
        (Section 12.9 of the SW4 User Guide). If format is instead EMOD3D, the
        low frequency path should be a directory containing LFSeis files.
        Defaults to EMOD3D.
    """
    match format:
        case Format.EMOD3D if low_frequency_path.is_dir():
            lf_dataset = timeseries.read_lfseis_directory(low_frequency_path)
            lf_dataset.to_netcdf(output_ffp, engine="h5netcdf")
        case Format.EMOD3D:
            raise ValueError("EMOD3D format requires directory containing LFSeis files")
        case Format.SW4 if low_frequency_path.is_file():
            lf_dataset = convert_sw4_station_recording(low_frequency_path)
            lf_dataset.to_netcdf(output_ffp, engine="h5netcdf")
        case Format.SW4:
            raise ValueError("SW4 format requires station recording file.")


if __name__ == "__main__":
    app()
