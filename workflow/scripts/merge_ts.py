#!/usr/bin/env python3
"""Merge EMOD3D Timeslices.

Description
-----------
Merge the output timeslice files of EMOD3D into a Zarr store.

Inputs
------
1. A directory containing EMOD3D timeslice files.

Outputs
-------
1. A merged output zarr store.

Environment
-----------
Can be run in the cybershake container. Can also be run from your own computer using the `merge-ts` command which is installed after running `pip install workflow@git+https://github.com/ucgmsim/workflow`.

Usage
-----
`merge-ts XYTS_DIRECTORY output.zarr`

For More Help
-------------
See the output of `merge-ts --help`.
"""

from numcodecs.zfpy import ZFPY

import dataclasses
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated
from collections.abc import Hashable

import dask.array as da
import numpy as np
import tqdm
import typer
import xarray as xr
from dask.diagnostics import ProgressBar
import zarr
import zfpy

from qcore import cli, coordinates, xyts

app = typer.Typer()


def read_component_xyts_files(
    xyts_directory: Path, glob_pattern: str
) -> list[xyts.XYTSFile]:
    """Read XYTS headers from component XYTS directory.

    Parameters
    ----------
    xyts_directory : Path
        The directory containing e3d files.
    glob_pattern : str
        The glob pattern to search for xyts files.

    Returns
    -------
    list[xyts.XYTSFile]
        A list of XYTS files with parsed metadata.
    """
    return [
        xyts.XYTSFile(
            xyts_file_path, proc_local_file=True, meta_only=True, round_dt=False
        )
        for xyts_file_path in xyts_directory.glob(glob_pattern)
    ]


WaveformArray = da.Array
CoordinateArray = np.ndarray[tuple[int, int], np.dtype[np.float64]]
TimeArray = np.ndarray[tuple[int], np.dtype[np.float64]]


@dataclass
class WaveformData:
    """Waveform data object"""

    x_start: int
    """Global x-start of waveform data."""
    x_end: int
    """Global x-end of waveform data."""
    y_start: int
    """Global y-start of waveform data."""
    y_end: int
    """Global y-end of waveform data."""
    data: WaveformArray
    """Waveform data (Dask array)."""
    z_start: int | None = None
    """Global z-start of waveform data."""
    z_end: int | None = None
    """Global z-end of waveform data."""


XYTS_PROC_HEADER_SIZE = 72


def read_waveform_data(xyts_file: xyts.XYTSFile) -> WaveformData:
    """Read waveform data from an XYTS file using Dask (lazy evaluation).

    Parameters
    ----------
    xyts_file : xyts.XYTSFile
        The XYTS file to read from.

    Returns
    -------
    WaveformData
        The extracted waveform data metadata and a lazy Dask array.

    Raises
    ------
    ValueError
        If the XYTS file is not a local XYTS file (output of EMOD3D).
        Local XYTS files will have non-None ``local_nx`` and
        ``local_ny`` attributes.
    """
    nt = xyts_file.nt
    components = len(xyts_file.comps)
    ny = xyts_file.local_ny
    nx = xyts_file.local_nx
    nz = getattr(xyts_file, "local_nz", None)

    if not (ny and nx):
        raise ValueError(
            "Encountered invalid XYTS component file (must have local ny and local nx both set)."
        )
    x0 = xyts_file.x0
    y0 = xyts_file.y0
    x1 = x0 + nx
    y1 = y0 + ny

    if nz is not None and nz > 0:
        z0 = getattr(xyts_file, "z0", 0)
        z1 = z0 + nz
        shape = (nt, components, nz, ny, nx)
    else:
        z0 = None
        z1 = None
        shape = (nt, components, ny, nx)

    lazy_data = da.from_array(
        np.memmap(
            xyts_file.xyts_path,
            dtype=np.float32,
            offset=XYTS_PROC_HEADER_SIZE,
            shape=shape,
            mode="r",
        ),
    )

    waveform_data = WaveformData(
        x_start=x0, y_start=y0, z_start=z0, x_end=x1, y_end=y1, z_end=z1, data=lazy_data
    )
    return waveform_data


@dataclass
class Metadata:
    """XYTS file metadata."""

    nx: int
    """Number of x gridpoints."""
    ny: int
    """Number of y gridpoints."""
    nt: int
    """Number of timesteps."""
    resolution: float
    """Spatial resolution (of simulation)."""
    dx: float
    """Spatial resolution (of XYTS file)."""
    dt: float
    """Temporal resolution."""
    mlon: float
    """Model origin longitude."""
    mlat: float
    """Model origin latitude."""
    mrot: float
    """Model rotation."""
    nz: int | None = None
    """Number of z gridpoints (if present)."""


def extract_metadata(xyts_file: xyts.XYTSFile) -> Metadata:
    """Extract metadata from an XYTS file.

    Parameters
    ----------
    xyts_file : xyts.XYTSFile
        The XYTS file to extract from.

    Returns
    -------
    Metadata
        The metadata extracted from the XYTS file.
    """
    nt = xyts_file.nt
    nx = xyts_file.nx
    ny = xyts_file.ny
    nz = getattr(xyts_file, "nz", None)
    resolution = xyts_file.hh
    dx = xyts_file.dx
    mlat = xyts_file.mlat
    mlon = xyts_file.mlon
    mrot = xyts_file.mrot
    dt = xyts_file.dt

    return Metadata(
        resolution=float(resolution),
        dx=float(dx),
        dt=float(dt),
        mlon=float(mlon),
        mlat=float(mlat),
        mrot=float(mrot),
        nx=int(nx),
        ny=int(ny),
        nt=int(nt),
        nz=int(nz) if nz is not None else None,
    )


def xyts_lat_lon_coordinates(
    metadata: Metadata,
) -> tuple[CoordinateArray, CoordinateArray]:
    """Generate the lat/lon coordinates corresponding to a model.

    Generates a ``lat`` and ``lon`` meshgrid such that ``waveform_data[i, j]``
    has latitude ``lat[i, j]`` and longitude ``lon[i, j]``.

    Parameters
    ----------
    metadata : Metadata
        The metadata describing a model (mrot, mlat, mlon, nx, ny,
        dx).

    Returns
    -------
    lat : array of float64
        The latitude coordinate meshgrid.
    lon : array of float64
        The longitude coordinate meshgrid.
    """
    proj = coordinates.SphericalProjection(
        mlon=metadata.mlon,
        mlat=metadata.mlat,
        mrot=metadata.mrot,
    )
    y, x = np.meshgrid(
        np.arange(metadata.ny, dtype=np.float64),
        np.arange(metadata.nx, dtype=np.float64),
        indexing="ij",
    )
    # dx = dy, so the following is ok.
    # Shift gridpoints so that they are origin centred.
    y = (y - metadata.ny / 2) * metadata.dx
    x = (x - metadata.nx / 2) * metadata.dx

    lat, lon = proj.inverse(x.flatten(), y.flatten()).T
    lat = lat.reshape(y.shape)
    lon = lon.reshape(y.shape)
    return lat, lon


def create_zarr_datastore(
    output: Path, dset: xr.Dataset, compress: set[Hashable], compressor: ZFPY
) -> None:
    for var_name, var_data in dset.data_vars.items():
        zarr.create_array(
            store=output / str(var_name),
            shape=var_data.shape,
            chunks="auto",
            dtype=var_data.dtype,
            serializer=compressor if var_name in compress else "auto",
            zarr_format=3,
            overwrite=True,
            fill_value=np.nan if np.issubdtype(var_data.dtype, np.floating) else 0,
            dimension_names=[str(dim) for dim in var_data.dims],
        )


@cli.from_docstring(app)
def merge_ts_zarr(
    component_xyts_directory: Annotated[
        Path,
        typer.Argument(
            dir_okay=True,
            file_okay=False,
            exists=True,
            readable=True,
        ),
    ],
    output: Annotated[
        Path,
        typer.Argument(dir_okay=True, writable=True),
    ],
    glob_pattern: str = "*xyts-*.e3d",
    scale: Annotated[float, typer.Option(min=0)] = 0.1,
) -> None:
    """Merge XYTS files into a Zarr store.

    Parameters
    ----------
    component_xyts_directory : Path
        The input xyts directory containing files to merge.
    output : Path
        The output zarr store path.
    glob_pattern : str, optional
        Set a custom glob pattern for merging the xyts files, by default "*xyts-*.e3d".
    scale : float, optional
        Set the scale for quantising XYTS outputs. Defaults to 0.1.
    complevel : int, optional
        Set the compression level for the output Zarr file. Range
        between 1-9 (9 being the highest level of compression).
        Defaults to 4.
    """
    component_xyts_files = read_component_xyts_files(
        component_xyts_directory, glob_pattern
    )
    if not component_xyts_files:
        raise FileNotFoundError(
            f"No files in '{component_xyts_directory}' match glob '{glob_pattern}'"
        )

    # XYTS files contain certain repeated metadata, so we can extract
    # a "sample" file for this common metadata.
    sample_xyts_file = component_xyts_files[0]
    metadata = extract_metadata(sample_xyts_file)
    bounds = np.iinfo(np.uint16)
    nan_value = bounds.max

    arrays = []

    for xyts_file in tqdm.tqdm(
        component_xyts_files, desc="Building Dask Graph", unit="files"
    ):
        local_data = read_waveform_data(xyts_file)
        magnitude = da.linalg.norm(local_data.data, axis=1)

        coords = {
            "time": np.arange(metadata.nt, dtype=np.float64) * metadata.dt,
            "y": np.arange(local_data.y_start, local_data.y_end),
            "x": np.arange(local_data.x_start, local_data.x_end),
        }

        if local_data.z_end is not None:
            coords["z"] = np.arange(local_data.z_start, local_data.z_end)
            dims = ("time", "z", "y", "x")
        else:
            dims = ("time", "y", "x")

        chunk_da = xr.DataArray(magnitude, dims=dims, coords=coords, name="waveform")
        arrays.append(chunk_da.to_dataset())

    merged_ds = xr.combine_by_coords(arrays, fill_value=nan_value)
    assert isinstance(merged_ds, xr.Dataset)
    merged_ds.attrs = dataclasses.asdict(metadata)
    lat, lon = xyts_lat_lon_coordinates(metadata)
    merged_ds["latitude"] = (("y", "x"), lat)
    merged_ds["longitude"] = (("y", "x"), lon)

    compressor = ZFPY(mode=zfpy.mode_fixed_accuracy, tolerance=scale)
    create_zarr_datastore(
        output, merged_ds, compress={"waveform"}, compressor=compressor
    )

    with ProgressBar():
        merged_ds.to_zarr(output, mode="a", zarr_format=3, compute=False)
        for var_name, var_data in merged_ds.data_vars.items():
            var_region = {dim: slice(None) for dim in var_data.dims}

            merged_ds[[var_name]].to_zarr(store=output, mode="a", region=var_region)


if __name__ == "__main__":
    app()
