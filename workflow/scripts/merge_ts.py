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
`merge-ts XYTS_DIRECTORY output.zarr --log-file my_job_monitor.log`

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
        Folder containing xyts.e3d files.
    
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
    data = np.memmap(filename, mode='r', shape=shape, dtype=dtype, offset=offset)
    return data[sl]


def mmap_dask_array(filename: Path, shape: tuple[int, ...], dtype: np.dtype, offset: int=0, blocksize: int=5) -> da.Array:
    load = dask.delayed(mmap_load_chunk)
    chunks = []
    for index in range(0, shape[0], blocksize):
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

TARGET_BLOCK_SIZE = 512 * 1024 * 1024

def read_waveform_data(xyts_file: xyts.XYTSFile) -> xr.DataArray:
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
        shape = (nt, components, ny, nz, nx)
        dims = ['time', 'component', 'y', 'z', 'x']
        
        blocksize = TARGET_BLOCK_SIZE // (4 * components * nz * ny * nx)
    else:
        z0 = None
        z1 = None
        shape = (nt, components, ny, nx)
        dims = ['time', 'component', 'y', 'x']
        blocksize = TARGET_BLOCK_SIZE // (4 * components * ny * nx)
        
    lazy_data = mmap_dask_array(
            xyts_file.xyts_path,
            dtype=np.float32,
            offset=XYTS_PROC_HEADER_SIZE,
            blocksize=blocksize,
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
    """Metadata dataclass for simulation data."""
    nx: int
    ny: int
    nt: int
    resolution: float
    dx: float
    dt: float
    mlon: float
    mlat: float
    mrot: float
    nz: int | None = None


def extract_metadata(xyts_file: xyts.XYTSFile) -> Metadata:
    """Extract the metadata from an XYTS file.

    Parameters
    ----------
    xyts_file : XYTSFile
        XYTS file handle to extract metadata from.

    Returns
    -------
    Metadata
        A metadata dataclass for the XYTS file.
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
    """Construct EMOD3D spatial coordinates for waveforms.

    Parameters
    ----------
    metadata : Metadata
        Metadata object describing domain boundaries.

    Returns
    -------
    lat : CoordinateArray
        The latitude coordinate for each point in the dataset.
    lon : CoordinateArray
        The longitude coordinate for each point in the dataset.
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
    scale: float = 0.1,
    complevel: int = 5,
    dx: int = 1,
    dy: int = 1,
    dz: int = 1,
    n_threads: int = utils.get_available_cores(),
) -> None:
    """Merge XYTS files into a Zarr store."""
    component_xyts_files = read_component_xyts_files(
        component_xyts_directory, glob_pattern
    )
    if not component_xyts_files:
        raise FileNotFoundError(
            f"No files in '{component_xyts_directory}' match glob '{glob_pattern}'"
        )

    sample_xyts_file = component_xyts_files[0]
    metadata = extract_metadata(sample_xyts_file)

    arrays = []
    with TqdmCallback(), dask.config.set(scheduler="threads", num_workers=n_threads):
        for xyts_file in tqdm.tqdm(
            component_xyts_files, desc="Building Dask Graph", unit="files"
        ):
            local_data = read_waveform_data(xyts_file)
            magnitude = np.sqrt((local_data ** 2).sum(dim='component'))
            arrays.append(magnitude.to_dataset(name='waveform'))

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

        lat, lon = xyts_lat_lon_coordinates(metadata)
        merged_ds['latitude'] = (('y', 'x'), lat)
        merged_ds['longitude'] = (('y', 'x'), lon)
        
        merged_ds.to_netcdf(output, engine='h5netcdf', encoding={'waveform': {
            'complevel': complevel,
            'dtype': 'uint16',
            'scale_factor': np.float32(scale),
            'add_offset': 0.0,
            '_FillValue': 65535,  # Use a valid uint16 integer for missing data
            'compression': 'zlib',
            'shuffle': True,
        }})



if __name__ == "__main__":
    app()
