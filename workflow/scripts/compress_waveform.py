"""Compress Waveform.

Description
-----------
Compress a broadband waveform HDF5 file using FlacArray compression with
int16 rescaling and delta encoding for efficient storage.

The waveform data is rescaled to fit the full range of a signed 16-bit
integer ([-32768, 32767]) and delta encoded along both the component
axis (to exploit inter-component correlation) and the time axis (to
exploit temporal smoothness) before FLAC compression.  This produces
small integer residuals that FLAC's Rice coding compresses very
efficiently.  All coordinates and attributes from the input xarray
dataset are preserved so the compressed file can be decompressed back
to a complete xarray Dataset.

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

import h5py
import numpy as np
import typer
import xarray as xr
from flacarray import FlacArray

from qcore import cli
from workflow import log_utils

app = typer.Typer()

_INT16_MAX = np.iinfo(np.int16).max


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


def _encode_chunk(ds_chunk: xr.Dataset, scale_factor: float) -> xr.Dataset:
    """Encode a waveform chunk with int16 scaling and delta encoding.

    Each chunk is expected to contain all three components and the full
    timeseries for a subset of stations.  The encoding pipeline is:

    1. Scale the float waveform to the int16 range using *scale_factor*.
    2. Delta-encode along the component axis (x, y, z are strongly
       correlated in seismic data, so differences are small).
    3. Delta-encode along the time axis (smooth waveforms produce
       near-zero residuals).

    Parameters
    ----------
    ds_chunk : xr.Dataset
        A chunk of the broadband dataset.
    scale_factor : float
        Global scale factor that maps the maximum amplitude to the
        int16 range.

    Returns
    -------
    xr.Dataset
        Encoded chunk with int32 waveform values.
    """
    waveform = ds_chunk["waveform"].values

    # Scale to fill the int16 range [-32768, 32767].
    scaled = np.round(waveform / scale_factor).astype(np.int32)

    # Component-wise delta: [x, y−x, z−y].
    # Components share strong seismic correlation, so their differences
    # have much smaller variance.
    comp_delta = np.diff(scaled, axis=0, prepend=np.zeros_like(scaled[:1]))

    # Time-wise delta: d[n] = s[n] − s[n−1].
    # Smooth waveforms produce near-zero residuals that compress well.
    time_delta = np.diff(
        comp_delta, axis=-1, prepend=np.zeros_like(comp_delta[..., :1])
    )

    return xr.Dataset(
        {"waveform": (ds_chunk["waveform"].dims, time_delta)},
        coords=ds_chunk.coords,
    )


@cli.from_docstring(app)
@log_utils.log_call()
def compress_waveform(
    waveform_ffp: Annotated[Path, typer.Argument(dir_okay=False, exists=True)],
    output_ffp: Annotated[Path, typer.Argument(dir_okay=False, writable=True)],
    level: Annotated[int, typer.Option(min=0, max=8)] = 5,
) -> None:
    """Compress a broadband waveform file using FlacArray.

    The waveform is chunked by station so that each parallel dask task
    processes complete component-triples with full timeseries.  Within
    each chunk the data is rescaled to the signed 16-bit integer range
    and delta encoded along both the component and time axes before
    FLAC compression.

    Parameters
    ----------
    waveform_ffp : Path
        Path to the input broadband waveform file (HDF5/NetCDF4).
    output_ffp : Path
        Path to the output compressed HDF5 file.
    level : int, optional
        FLAC compression level (0-8). Higher values compress more but
        are slower. Defaults to 5.
    """
    # Chunk by station: each chunk gets all components and full timeseries
    # for a subset of stations, allowing parallel encoding via dask.
    broadband = xr.open_dataset(waveform_ffp, chunks={"station": "auto"})

    waveform_dtype = str(broadband["waveform"].dtype)

    # Compute the global scale factor.  This is a reduction that dask
    # evaluates lazily until .compute() is called.
    max_abs = float(abs(broadband["waveform"]).max().compute())
    scale_factor = max_abs / _INT16_MAX if max_abs > 0 else 1.0

    # Encode each station-chunk in parallel: scale → component delta → time delta.
    encoded = broadband.map_blocks(
        functools.partial(_encode_chunk, scale_factor=scale_factor)
    )

    # Materialise the encoded int32 array and compress losslessly with FLAC.
    encoded_data: np.ndarray = encoded["waveform"].compute().values
    flac_waveform = FlacArray.from_array(encoded_data, level=level)

    with h5py.File(output_ffp, "w") as hdf:
        flac_waveform.write_hdf5(hdf.create_group("waveform"))
        _write_coords(hdf, broadband)

        for attr_name, attr_value in broadband.attrs.items():
            hdf.attrs[attr_name] = attr_value

        hdf.attrs["waveform_dims"] = list(broadband["waveform"].dims)
        hdf.attrs["scale_factor"] = scale_factor
        hdf.attrs["waveform_dtype"] = waveform_dtype

    broadband.close()


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
        flac_waveform = FlacArray.read_hdf5(hdf["waveform"])
        delta = flac_waveform.to_array()

        # Undo time delta encoding: cumulative sum along time axis.
        comp_delta = np.cumsum(delta, axis=-1)

        # Undo component delta encoding: cumulative sum along component axis.
        scaled = np.cumsum(comp_delta, axis=0)

        # Rescale back to the original floating-point range.
        scale_factor = hdf.attrs["scale_factor"]
        waveform_dtype = np.dtype(str(hdf.attrs["waveform_dtype"]))
        scale_factor_typed = waveform_dtype.type(scale_factor)
        waveform = scaled.astype(waveform_dtype) * scale_factor_typed

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
