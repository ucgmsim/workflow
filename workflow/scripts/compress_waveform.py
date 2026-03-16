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

import h5py
import numpy as np
import typer
import xarray as xr
from flacarray import FlacArray

from qcore import cli
from workflow import log_utils

app = typer.Typer()

# Maximum scaled value.  After component-delta encoding the worst-case
# integer value is 2× the scaled maximum.  Using 2^23 − 1 (matching the
# float32 mantissa width) gives effectively lossless round-trip precision
# for single-precision data while keeping all intermediate delta values
# well within int32 range for FlacArray lossless compression.
_SCALE_LIMIT = (1 << 23) - 1


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
    """Encode a waveform chunk with int32 scaling and component-delta encoding.

    Each chunk is expected to contain all three components and the full
    timeseries for a subset of stations.  The encoding pipeline is:

    1. Scale the float waveform to the int32 safe range using
       *scale_factor*.
    2. Delta-encode along the component axis (x, y, z are strongly
       correlated in seismic data, so differences are small).

    FLAC's built-in linear prediction already exploits temporal
    smoothness, so no explicit time-axis delta is applied here.

    Parameters
    ----------
    ds_chunk : xr.Dataset
        A chunk of the broadband dataset.
    scale_factor : float
        Global scale factor that maps the maximum amplitude to the
        int32 safe range.

    Returns
    -------
    xr.Dataset
        Encoded chunk with int32 waveform values.
    """
    waveform = ds_chunk["waveform"].values

    # Scale to fill the int32 safe range.
    scaled = np.round(waveform / scale_factor).astype(np.int32)

    # Component-wise delta: [x, y−x, z−y].
    # Components share strong seismic correlation, so their differences
    # have much smaller variance.
    comp_delta = np.diff(scaled, axis=0, prepend=np.zeros_like(scaled[:1]))

    return xr.Dataset(
        {"waveform": (ds_chunk["waveform"].dims, comp_delta)},
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
    """
    # Chunk by station: each chunk gets all components and full timeseries
    # for a subset of stations, allowing parallel encoding via dask.
    broadband = xr.open_dataset(waveform_ffp, chunks={"station": "auto"})

    waveform_dtype = str(broadband["waveform"].dtype)

    # Compute the global scale factor.  This is a reduction that dask
    # evaluates lazily until .compute() is called.
    max_abs = float(abs(broadband["waveform"]).max().compute())
    scale_factor = max_abs / _SCALE_LIMIT if max_abs > 0 else 1.0

    # Encode each station-chunk in parallel: scale → component delta.
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
        comp_delta = flac_waveform.to_array()

        # Undo component delta encoding: cumulative sum along component axis.
        # Use int64 to guarantee no overflow on all platforms.
        scaled = np.cumsum(comp_delta, axis=0, dtype=np.int64)

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
