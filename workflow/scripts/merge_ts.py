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
import dataclasses
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any

import dask
import dask.array as da
import numpy as np
import tqdm
import typer
import xarray as xr
from tqdm.dask import TqdmCallback
from zarr.codecs import BloscCodec, Quantize

from qcore import cli, coordinates, xyts
from workflow import utils

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



XYTS_PROC_HEADER_SIZE = 72
def mmap_load_chunk(filename: Path, shape: tuple[int, ...], dtype: np.dtype, offset: int, sl: Any) -> np.ndarray:
    '''
    Memory map the given file with overall shape and dtype and return a slice
    specified by :code:`sl`.

    Parameters
    ----------

    filename : str
    shape : tuple
        Total shape of the data in the file
    dtype:
        NumPy dtype of the data in the file
    offset : int
        Skip :code:`offset` bytes from the beginning of the file.
    sl:
        Object that can be used for indexing or slicing a NumPy array to
        extract a chunk

    Returns
    -------

    numpy.memmap or numpy.ndarray
        View into memory map created by indexing with :code:`sl`,
        or NumPy ndarray in case no view can be created using :code:`sl`.
    '''
    data = np.memmap(filename, mode='r', shape=shape, dtype=dtype, offset=offset)
    return data[sl]


def mmap_dask_array(filename: Path, shape: tuple[int, ...], dtype: np.dtype, offset: int=0, blocksize: int=5) -> da.Array:
    '''
    Create a Dask array from raw binary data in :code:`filename`
    by memory mapping.

    This method is particularly effective if the file is already
    in the file system cache and if arbitrary smaller subsets are
    to be extracted from the Dask array without optimizing its
    chunking scheme.

    It may perform poorly on Windows if the file is not in the file
    system cache. On Linux it performs well under most circumstances.

    Parameters
    ----------

    filename : str
    shape : tuple
        Total shape of the data in the file
    dtype:
        NumPy dtype of the data in the file
    offset : int, optional
        Skip :code:`offset` bytes from the beginning of the file.
    blocksize : int, optional
        Chunk size for the outermost axis. The other axes remain unchunked.

    Returns
    -------

    dask.array.Array
        Dask array matching :code:`shape` and :code:`dtype`, backed by
        memory-mapped chunks.
    '''
    load = dask.delayed(mmap_load_chunk)
    chunks = []
    for index in range(0, shape[0], blocksize):
        # Truncate the last chunk if necessary
        chunk_size = min(blocksize, shape[0] - index)
        chunk = dask.array.from_delayed(
            load(
                filename,
                shape=shape,
                dtype=dtype,
                offset=offset,
                sl=slice(index, index + chunk_size)
            ),
            shape=(chunk_size, ) + shape[1:],
            dtype=dtype
        )
        chunks.append(chunk)
    return da.concatenate(chunks, axis=0)


def read_waveform_data(xyts_file: xyts.XYTSFile) -> xr.DataArray:
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
    
    coords = {'time': np.arange(0, nt), 'x': np.arange(x0, x1), 'y': np.arange(y0, y1), 'component': np.arange(components)}
    if nz is not None and nz > 0:
        z0 = getattr(xyts_file, "z0", 0)
        z1 = z0 + nz
        coords['z'] = np.arange(z0, z1)
        shape = (nt, components, nz, ny, nx)
        dims = ['time', 'component', 'nz', 'ny', 'nx']
    else:
        z0 = None
        z1 = None
        shape = (nt, components, ny, nx)
        dims = ['time', 'component', 'ny', 'nx']
        


    chunks = list(shape)
    chunks[0] = 1
    lazy_data = mmap_dask_array(
            xyts_file.xyts_path,
            dtype=np.float32,
            offset=XYTS_PROC_HEADER_SIZE,
            blocksize=10,
            shape=shape,
    )
    waveform_data = xr.DataArray(
        lazy_data,
        dims=dims,
        coords=coords,
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
    scale: int = 1,
    complevel: int = 5,
    dx: int = 1,
    dy: int = 1,
    dz: int = 1,
    n_threads: int = utils.get_available_cores() 
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

    arrays = []
    with TqdmCallback(), dask.config.set(scheduler="threads", num_workers=resolved_n_threads):
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


        filters = [
            Quantize(digits=scale, dtype='float32'),
        ]
        compressors = [
            BloscCodec(cname='zstd', clevel=complevel, shuffle='shuffle'),
        ]

        merged_ds = xr.combine_by_coords(arrays)
        
        selectors = dict()
        if dx != 1:
            selectors['x'] = range(0, metadata.nx, dx)
        if dy != 1:
            selectors['y'] = range(0, metadata.ny, dy)
        if dz != 1 and metadata.nz:
            selectors['z'] = range(0, metadata.nz, dz)

        if selectors:
            merged_ds = merged_ds.isel(selectors)
            
        assert isinstance(merged_ds, xr.Dataset)
        merged_ds.attrs = dataclasses.asdict(metadata)
        merged_ds = merged_ds.chunk(time=1, x=256, y=256, z=256)

        merged_ds.to_zarr(
            output,
            mode="w",
            zarr_format=3,
            encoding=dict(
                waveform=dict(
                    filters=filters,
                    compressors=compressors
                )
            )
        )


if __name__ == "__main__":
    app()
