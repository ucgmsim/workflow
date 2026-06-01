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
import logging
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any

import dask
import dask.array as da
import numpy as np
import psutil
import tqdm
import typer
import xarray as xr
from dask.distributed import Client, LocalCluster
from tqdm.dask import TqdmCallback
from zarr.codecs import BloscCodec, Quantize

from qcore import cli, coordinates, xyts
from workflow import utils

app = typer.Typer()

# Instantiate the logger
logger = logging.getLogger("merge_ts")


def _resource_monitor_worker(stop_event: threading.Event, interval: float = 5.0):
    """Background worker loop that logs system resources at regular intervals."""
    # Seed the CPU and Disk counters
    psutil.cpu_percent(interval=None)
    last_disk = psutil.disk_io_counters()
    last_time = time.time()

    while not stop_event.is_set():
        # Sleep in small increments so we can react quickly to the stop event
        elapsed = 0.0
        while elapsed < interval:
            if stop_event.is_set():
                return
            time.sleep(0.5)
            elapsed += 0.5

        try:
            current_time = time.time()
            time_delta = current_time - last_time
            
            cpu = psutil.cpu_percent(interval=None)
            mem = psutil.virtual_memory()
            current_disk = psutil.disk_io_counters()

            # Calculate actual speed (MB/s) since last interval
            if current_disk and last_disk:
                read_speed = ((current_disk.read_bytes - last_disk.read_bytes) / (1024 ** 2)) / time_delta
                write_speed = ((current_disk.write_bytes - last_disk.write_bytes) / (1024 ** 2)) / time_delta
                disk_str = f"DiskRead: {read_speed:.1f} MB/s, DiskWrite: {write_speed:.1f} MB/s"
            else:
                disk_str = "DiskActivity: N/A"

            mem_used_gb = mem.used / (1024 ** 3)
            mem_total_gb = mem.total / (1024 ** 3)

            logger.info(
                f"[RESOURCE HEARTBEAT] CPU: {cpu}% | "
                f"Mem: {mem.percent}% ({mem_used_gb:.1f}/{mem_total_gb:.1f} GB) | "
                f"{disk_str}"
            )

            # Update snapshots
            last_disk = current_disk
            last_time = current_time

        except Exception as e:
            logger.error(f"Error in resource monitor thread: {e}")


def read_component_xyts_files(
    xyts_directory: Path, glob_pattern: str
) -> list[xyts.XYTSFile]:
    """Read XYTS headers from component XYTS directory."""
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
    logger.debug(f"Reading chunk from {filename} at {sl}")
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
        shape = (nt, components, nz, ny, nx)
        dims = ['time', 'component', 'z', 'y', 'x']
        blocksize = 8
    else:
        z0 = None
        z1 = None
        shape = (nt, components, ny, nx)
        dims = ['time', 'component', 'y', 'x']
        blocksize = 2048
        
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
    scale: int = 1,
    complevel: int = 5,
    dx: int = 1,
    dy: int = 1,
    dz: int = 1,
    n_threads: int = utils.get_available_cores(),
    log_file: Annotated[
        Path | None,
        typer.Option(
            "--log-file",
            help="Path to the file where debug logs will be stored for monitoring.",
            dir_okay=False,
            writable=True,
        ),
    ] = None,
) -> None:
    """Merge XYTS files into a Zarr store."""
    monitor_thread = None
    stop_monitor = threading.Event()

    if log_file:
        logging.basicConfig(
            filename=log_file,
            level=logging.DEBUG,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            filemode="a",
        )
        logger.debug(f"Logging initialized. Writing to {log_file}")
        
        # Start background resource monitoring thread (checks every 5 seconds)
        monitor_thread = threading.Thread(
            target=_resource_monitor_worker, 
            args=(stop_monitor, 5.0), 
            daemon=True
        )
        monitor_thread.start()
        logger.debug("Background resource monitoring thread started.")

    try:
        logger.debug(f"Searching for files in '{component_xyts_directory}' with pattern '{glob_pattern}'")
        component_xyts_files = read_component_xyts_files(
            component_xyts_directory, glob_pattern
        )
        if not component_xyts_files:
            logger.error(f"No files matched pattern '{glob_pattern}' inside '{component_xyts_directory}'")
            raise FileNotFoundError(
                f"No files in '{component_xyts_directory}' match glob '{glob_pattern}'"
            )

        logger.debug(f"Found {len(component_xyts_files)} component XYTS files.")
        sample_xyts_file = component_xyts_files[0]
        metadata = extract_metadata(sample_xyts_file)

        arrays = []
        logger.debug(f"Building Dask Graph using {n_threads} worker threads...")
        cluster = LocalCluster(n_workers=n_threads, threads_per_worker=1, memory_limit=None)
        with Client(cluster):
            for xyts_file in tqdm.tqdm(
                component_xyts_files, desc="Building Dask Graph", unit="files"
            ):
                local_data = read_waveform_data(xyts_file)
                magnitude = np.sqrt((local_data ** 2).sum(dim='component'))
                arrays.append(magnitude.to_dataset(name='waveform'))

            filters = [
                Quantize(digits=scale, dtype='float32'),
            ]
            compressors = [
                BloscCodec(cname='zstd', clevel=complevel, shuffle='shuffle'),
            ]

            logger.debug("Combining datasets by coordinates...")
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
            
            logger.debug(f"Writing dataset to Zarr at {output}. (Dask execution compute phase active)")
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
            logger.debug("Zarr writing process successfully completed.")

    finally:
        # Guarantee the thread winds down cleanly even if the main block errors out
        if monitor_thread and monitor_thread.is_alive():
            logger.debug("Stopping system resource monitor thread...")
            stop_monitor.set()
            monitor_thread.join(timeout=2.0)


if __name__ == "__main__":
    app()
