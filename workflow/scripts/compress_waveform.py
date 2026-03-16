"""Compress Waveform.

Description
-----------
Compress a broadband waveform HDF5 file using FlacArray compression with
int32 rescaling and component-delta encoding for efficient storage.

The waveform data is rescaled to fill the safe range of a signed 32-bit
integer and delta encoded along the component axis (to exploit
inter-component correlation) before FLAC compression.  FLAC's built-in
linear prediction handles temporal smoothness internally, so no explicit
time-axis delta encoding is needed.  All coordinates and attributes from
the input xarray dataset are preserved so the compressed file can be
decompressed back to a complete xarray Dataset.

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

import functools
from pathlib import Path
from typing import Annotated

import flacarray
import h5py
import numpy as np
import typer
import xarray as xr

from qcore import cli
from workflow import log_utils

app = typer.Typer()


def _write_coords(
    hdf: h5py.File,
    broadband: xr.Dataset,
) -> None:
    """Write all coordinate variables to an HDF5 group.

    Parameters
    ----------
    hdf : h5py.File
        The open HDF5 file to write to.
    broadband : xr.Dataset
        The source xarray dataset.
    """
    coords_grp = hdf.create_group("coords")
    for coord_name in broadband.coords:
        coord_data = broadband.coords[coord_name].values
        if coord_data.dtype.kind == "U":
            coord_data = coord_data.astype(bytes)
        dset = coords_grp.create_dataset(coord_name, data=coord_data)
        dset.attrs["dims"] = list(broadband.coords[coord_name].dims)


def _read_coords(hdf: h5py.File) -> dict[str, tuple[tuple[str, ...], np.ndarray]]:
    """Read coordinate variables from an HDF5 group.

    Parameters
    ----------
    hdf : h5py.File
        The open HDF5 file to read from.

    Returns
    -------
    dict
        A mapping of coordinate names to (dims, values) tuples.
    """
    coords: dict[str, tuple[tuple[str, ...], np.ndarray]] = {}
    for coord_name in hdf["coords"]:
        dset = hdf["coords"][coord_name]
        data = dset[:]
        if data.dtype.kind == "S":
            data = data.astype(str)
        dims = tuple(dset.attrs["dims"])
        coords[coord_name] = (dims, data)
    return coords


@cli.from_docstring(app)
@log_utils.log_call()
def compress_waveform(
    waveform_ffp: Annotated[Path, typer.Argument(dir_okay=False, exists=True)],
    output_ffp: Annotated[Path, typer.Argument(dir_okay=False, writable=True)],
    level: Annotated[int, typer.Option(min=0, max=8)] = 5,
    precision: Annotated[int, typer.Option(min=1)] = 4,
) -> None:
    """Compress a broadband waveform file using FlacArray.

    The waveform is chunked by station so that each parallel dask task
    processes complete component-triples with full timeseries.  Within
    each chunk the data is rescaled to a safe sub-range of the signed
    32-bit integer type and delta encoded along the component axis
    before FLAC compression.

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
        FLAC precision level (in significant digits of input data). Higher values compress more but
        are lose more precision. Defaults to 4.
    """
    with (
        h5py.File(output_ffp, "w") as hdf,
        xr.open_dataset(waveform_ffp, engine="h5netcdf") as broadband,
    ):
        flacarray.hdf5.write_array(  # type: ignore[possibly-missing-attribute]
            broadband.waveform.values,
            hdf,
            precision=precision,
            level=level,
            use_threads=True,
        )
        _write_coords(hdf, broadband)

        for attr_name, attr_value in broadband.attrs.items():
            hdf.attrs[attr_name] = attr_value

        hdf.attrs["waveform_dims"] = list(broadband["waveform"].dims)


def decompress_waveform(compressed_ffp: Path) -> xr.Dataset:
    """Decompress a FlacArray-compressed waveform file to an xarray Dataset.

    Parameters
    ----------
    compressed_ffp : Path
        Path to the compressed HDF5 file produced by ``compress_waveform``.

    Returns
    -------
    xr.Dataset
        The decompressed waveform dataset with all original coordinates
        and attributes restored.
    """
    with h5py.File(compressed_ffp, "r") as hdf:
        waveform = flacarray.hdf5.read_array(hdf, use_threads=True)  # type: ignore[possibly-missing-attribute]

        dims = list(hdf.attrs["waveform_dims"])
        coords = _read_coords(hdf)
        attrs = {
            k: v
            for k, v in hdf.attrs.items()
            if k not in ("waveform_dims", "scale_factor", "waveform_dtype")
        }

    return xr.Dataset(
        {"waveform": (dims, waveform)},
        coords=coords,
        attrs=attrs,
    )


if __name__ == "__main__":
    app()
