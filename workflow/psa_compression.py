"""Lossy, invertible compression for full 180-angle pSA RotD curves.

`IM.ims.pseudo_spectral_acceleration(..., full_rotd180=True)` returns a
`rotd180` variable of shape `(station, period, angle=180)`: pSA (g) at every
integer rotation angle for every period and station. Stored as plain floats
this is roughly 180x the size of the RotD0/50/100 summary statistics, so it
is compressed before being written to disk.

The scheme: `ln(pSA)` is quantized to a fixed step -- a uniform relative
error bound, since a step on `ln(pSA)` is a step in relative terms -- and
then first-differenced along the `angle` axis. The directional curve is very
smooth in theta, so the quantized differences are almost all 0 or +/-1, which
a byte-shuffled gzip packs down substantially.

Deliberately built on HDF5's *native* gzip + shuffle filters rather than a
third-party dynamic filter plugin (e.g. `hdf5plugin`'s zstd, which compresses
better): this process always imports `netCDF4` before anything else (see
`workflow/scripts/im_calc.py`'s header comment), and `netCDF4` bundles HDF5
1.14.6 while `h5py` bundles HDF5 2.0.0. With both loaded in one process, a
dynamically-loaded filter plugin compiled against one of those ABIs reliably
fails with `H5Z__prelude_callback(): error during user callback` when applied
through the other. Gzip and shuffle are built into every HDF5 build, so this
sidesteps that conflict entirely.
"""

import numpy as np
import xarray as xr

DEFAULT_REL_STEP = 0.02
"""Default relative-error step. Bounds the max relative error at ~rel_step/2."""


def encode_psa_rotd180(
    rotd180: xr.DataArray, rel_step: float = DEFAULT_REL_STEP
) -> xr.DataArray:
    """Quantize and delta-encode a full 180-angle pSA RotD curve.

    Parameters
    ----------
    rotd180 : xr.DataArray
        pSA (g), with an `angle` dimension of size 180 (degrees, 0..179).
        May be lazy (dask-backed); the encoding stays lazy in that case.
    rel_step : float, optional
        Relative-error step size on pSA. The resulting max relative error is
        bounded by `expm1(log1p(rel_step) / 2)`, e.g. ~1% for the 2% default.

    Returns
    -------
    xr.DataArray
        int16-encoded array, same shape and dims as `rotd180`. The first
        angle holds the quantized absolute level; the rest hold first
        differences along `angle`. Decode with `decode_psa_rotd180`. Carries
        `ln_step`, `rel_step`, and `max_relative_error_bound` attrs recording
        how it was encoded.
    """
    ln_step = float(np.log1p(rel_step))
    # `.round()` (not `np.round()`, which silently forces a dask-backed
    # DataArray to compute instead of dispatching to `dask.array.round`).
    quantized = (np.log(np.maximum(rotd180, 1e-30)) / ln_step).round()
    level = quantized.isel(angle=slice(0, 1))
    delta = quantized.diff("angle")
    encoded = xr.concat([level, delta], dim="angle").astype(np.int16)
    encoded.attrs = {
        "description": (
            "First differences along the angle axis of "
            "round(ln(pSA[g]) / ln_step). Recover pSA (g) with "
            "workflow.psa_compression.decode_psa_rotd180, or manually as "
            "exp(cumsum(value, axis='angle') * ln_step)."
        ),
        "ln_step": ln_step,
        "rel_step": rel_step,
        "max_relative_error_bound": float(np.expm1(ln_step / 2)),
    }
    return encoded


def decode_psa_rotd180(encoded: xr.DataArray) -> xr.DataArray:
    """Invert `encode_psa_rotd180`.

    Parameters
    ----------
    encoded : xr.DataArray
        int16 array produced by `encode_psa_rotd180`, carrying an `ln_step`
        attribute.

    Returns
    -------
    xr.DataArray
        pSA (g), same shape and dims as `encoded`.
    """
    ln_step = encoded.attrs["ln_step"]
    quantized = encoded.astype(np.int64).cumsum("angle")
    decoded = np.exp(quantized * ln_step)
    decoded.attrs["units"] = "g"
    return decoded


def rotd180_netcdf_encoding(encoded: xr.DataArray, complevel: int = 4) -> dict:
    """Build the `to_netcdf` encoding dict for an encoded `rotd180` variable.

    Applies HDF5's native byte-shuffle + gzip filters (the encoded values are
    almost all small integers, so shuffling bytes before gzip compresses
    substantially better than gzip alone), chunked so that every period and
    angle for a given station chunk lands in the same HDF5 chunk -- the delta
    encoding only pays off when whole angle sweeps compress together.

    `complevel` defaults to 4, not gzip's max of 9. Writing a chunk through a
    filter is a single call into the HDF5 C library, which is not reentrant
    across threads and so is serialized under h5py's global lock -- meaning
    wall-clock time for this step is dask-thread-count-independent and scales
    directly with the per-chunk gzip cost. Levels above ~4-6 buy a small extra
    size reduction for a much larger, non-parallelisable time cost; measured
    on realistic delta-encoded data, level 9 took several times longer than
    level 4 for a size difference in the low single-digit percent.

    Parameters
    ----------
    encoded : xr.DataArray
        int16 array produced by `encode_psa_rotd180`, with dims
        `(station, period, angle)`.
    complevel : int, optional
        gzip compression level (1-9). Higher is smaller but slower, and the
        write is not parallel across dask workers -- see above.

    Returns
    -------
    dict
        Encoding options for `xr.Dataset.to_netcdf`/`xr.DataTree.to_netcdf`
        (pass as `encoding={group_path: {"rotd180": rotd180_netcdf_encoding(...)}}`).
    """
    # `.variable.chunks` (the dask array's own per-axis chunks) rather than
    # `.chunksizes` (an xarray property that cross-validates every attached
    # coordinate too): `latitude`/`longitude` ride along as station-dimension
    # coordinates but get their own, independently-sized "auto" dask chunking
    # upstream, so `.chunksizes` raises "inconsistent chunks along dimension
    # station" even though the data variable itself is perfectly consistent.
    variable_chunks = encoded.variable.chunks
    station_chunk = (
        int(variable_chunks[encoded.get_axis_num("station")][0])
        if variable_chunks is not None
        else encoded.sizes["station"]
    )
    chunksizes = (station_chunk, encoded.sizes["period"], encoded.sizes["angle"])
    return {
        "zlib": True,
        "complevel": complevel,
        "shuffle": True,
        "chunksizes": chunksizes,
    }
