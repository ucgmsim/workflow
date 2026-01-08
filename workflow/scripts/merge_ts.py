#!/usr/bin/env python3
"""Merge EMOD3D Timeslices.

Description
-----------
Merge the output timeslice files of EMOD3D.

Inputs
------
1. A directory containing EMOD3D timeslice files.

Outputs
-------
1. A merged output timeslice file.

Environment
-----------
Can be run in the cybershake container. Can also be run from your own computer using the `merge-ts` command which is installed after running `pip install workflow@git+https://github.com/ucgmsim/workflow`.

Usage
-----
`merge-ts XYTS_DIRECTORY output.h5`

For More Help
-------------
See the output of `merge-ts --help`.
"""

import dataclasses
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

import numpy as np
import tqdm
import typer
import xarray as xr

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


WaveformArray = np.ndarray[tuple[int, int, int, int], np.dtype[np.float32]]
QuantisedArray = np.ndarray[tuple[int, int, int], np.dtype[np.uint16]]
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
    """Waveform data."""


XYTS_PROC_HEADER_SIZE = 72


def read_waveform_data(xyts_file: xyts.XYTSFile) -> WaveformData:
    """Read waveform data from an XYTS file.

    Parameters
    ----------
    xyts_file : xyts.XYTSFile
        The XYTS file to read from.

    Returns
    -------
    WaveformData
        The extracted waveform data.

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
    if not (ny and nx):
        raise ValueError(
            "Encountered invalid XYTS component file (must have local ny and local nx both set)."
        )
    x0 = xyts_file.x0
    y0 = xyts_file.y0
    x1 = x0 + nx
    y1 = y0 + ny

    data = np.fromfile(
        xyts_file.xyts_path, dtype=np.float32, offset=XYTS_PROC_HEADER_SIZE
    ).reshape((nt, components, ny, nx))
    waveform_data = WaveformData(x_start=x0, y_start=y0, x_end=x1, y_end=y1, data=data)
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
    resolution = xyts_file.hh
    dx = xyts_file.dx
    mlat = xyts_file.mlat
    mlon = xyts_file.mlon
    mrot = xyts_file.mrot
    dt = xyts_file.dt
    # The casts here convert from numpy to python types
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


def create_xyts_dataset(
    data: QuantisedArray,
    lat: CoordinateArray,
    lon: CoordinateArray,
    time: TimeArray,
    metadata: Metadata,
) -> xr.Dataset:
    """Create an XYTS dataset from given waveform data, coordinate meshgrid, time and metadata.

    Parameters
    ----------
    data : (nt, ny, nx) array of uint16
        Quantised waveform data.
    lat : (ny, nx) array of float64
        Latitude meshgrid.
    lon : (ny, nx) array of float64
        Longitude meshgrid.
    time : (nt,) array of float64
        Time array.
    metadata : Metadata
        Metadata object.

    Returns
    -------
    xr.Dataset
        An xarray dataset with coordinates ``time``, ``y`` and ``x``
        indexing the waveform data, lat and lon arrays. Metadata
        populates the attributes.
    """
    (nt, ny, nx) = data.shape
    if metadata.nx != nx or metadata.ny != ny or metadata.nt != nt:
        raise ValueError(
            f"Metadata does not match data, {metadata.nx=}, {metadata.ny=}, {metadata.nt=} but data {nx=}, {ny=}, {nt=}"
        )
    elif lon.shape != (ny, nx):
        raise ValueError(
            f"Longitude shape incompatible, {lon.shape=} but {data.shape=}"
        )
    elif lat.shape != (ny, nx):
        raise ValueError(f"Latitude shape incompatible, {lat.shape=} but {data.shape=}")
    elif time.shape != (nt,):
        raise ValueError(
            f"Time shape incompatible, expected {(nt,)} but found {time.shape=}"
        )

    dset = xr.Dataset(
        {
            "waveform": (("time", "y", "x"), data),
        },
        coords={
            "time": ("time", time),
            "y": ("y", np.arange(metadata.ny)),
            "x": ("x", np.arange(metadata.nx)),
            "latitude": (("y", "x"), lat),
            "longitude": (("y", "x"), lon),
        },
        attrs=dataclasses.asdict(metadata),
    )

    return dset


def set_scale(dset: xr.Dataset, scale: float) -> None:
    """Set dataset scale properties.

    This function sets the appropriate netcdf properties and units to
    transparently read the uint16 quantised waveform values as 64-bit
    floating point arrays.

    Parameters
    ----------
    dset : xr.Dataset
        Dataset to update.
    scale : float
        Scale for waveform quantisation.
    """
    bounds = np.iinfo(np.uint16)
    max_bound = bounds.max
    dset["waveform"].attrs.update(
        {
            "scale_factor": scale,
            "add_offset": 0.0,
            "units": "cm/s",
            "_FillValue": max_bound,  # Reserve max bound for NaN values
        }
    )


def quantise_array(waveform_data: WaveformArray, scale: float) -> QuantisedArray:
    r"""
    Quantise a floating-point waveform array into 16-bit unsigned integers.

    The transformation follows the formula:
    $$output = \text{round}(\text{clip}(\frac{waveform\_data}{scale}, 0, 65534))$$

    Parameters
    ----------
    waveform_data : WaveformArray
        The input floating-point array. All values are expected to be >= 0.
    scale : float
        The quantisation step size (resolution). For example, a scale of 0.1
        means the output represents increments of 0.1 from the input.

    Returns
    -------
    QuantisedArray (uint16)
        The discrete representation of the waveform. Values are capped at
        65534 to reserve 65535 as a NaN indicator.
    """
    scaled = waveform_data / scale
    bounds = np.iinfo(np.uint16)
    max_bound = bounds.max
    np.nan_to_num(scaled, nan=max_bound, copy=False)
    np.clip(scaled, 0, max_bound - 1, out=scaled)
    np.round(scaled, out=scaled)
    return scaled.astype(np.uint16)


@cli.from_docstring(app)
def merge_ts_hdf5(
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
        typer.Argument(dir_okay=False, writable=True),
    ],
    glob_pattern: str = "*xyts-*.e3d",
    scale: Annotated[float, typer.Option(min=0)] = 0.1,
    complevel: Annotated[int, typer.Option(min=1, max=9)] = 4,
) -> None:
    """Merge XYTS files.

    Parameters
    ----------
    component_xyts_directory : Path
        The input xyts directory containing files to merge.
    output : Path
        The output xyts file.
    glob_pattern : str, optional
        Set a custom glob pattern for merging the xyts files, by default "*xyts-*.e3d".
    scale : float, optional
        Set the scale for quantising XYTS outputs. Defaults to 0.1.
    complevel : int, optional
        Set the compression level for the output HDF5 file. Range
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

    waveform_data = np.empty((metadata.nt, metadata.ny, metadata.nx), dtype=np.uint16)

    for xyts_file in tqdm.tqdm(component_xyts_files, unit="files"):
        local_data = read_waveform_data(xyts_file)
        magnitude = np.linalg.norm(local_data.data, axis=1)
        quantised = quantise_array(magnitude, scale)
        waveform_data[
            :,
            local_data.y_start : local_data.y_end,
            local_data.x_start : local_data.x_end,
        ] = quantised

    lat, lon = xyts_lat_lon_coordinates(metadata)
    time = np.arange(metadata.nt, dtype=np.float64) * metadata.dt
    dset = create_xyts_dataset(waveform_data, lat, lon, time, metadata)
    set_scale(dset, scale)

    dset.to_netcdf(
        output,
        engine="h5netcdf",
        encoding={
            "waveform": {
                "dtype": "uint16",
                "compression": "zlib",
                "complevel": complevel,
                "shuffle": True,
            }
        },
    )
