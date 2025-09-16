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
`merge_ts XYTS_DIRECTORY XYTS_DIRECTORY/output.e3d`

For More Help
-------------
See the output of `merge-ts --help`.
"""

import os
from pathlib import Path
from typing import Annotated

import typer
import tqdm
import xarray as xr

from qcore import cli, xyts, coordinates
from workflow.scripts import merge_ts_loop

app = typer.Typer()

def merge_ts_xyts(
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
):
    """Merge XYTS files.

    Parameters
    ----------
    component_xyts_directory : Path
        The input xyts directory containing files to merge.
    output : Path
        The output xyts file.
    glob_pattern : str, optional
        Set a custom glob pattern for merging the xyts files, by default "*xyts-*.e3d".
    hdf5 : bool, optional
        If set, save output as a highly compressed HDF5 file.
    """
    component_xyts_files = sorted(
        [
            xyts.XYTSFile(
                xyts_file_path, proc_local_file=True, meta_only=True, round_dt=False
            )
            for xyts_file_path in component_xyts_directory.glob(glob_pattern)
        ],
        key=lambda xyts_file: (xyts_file.y0, xyts_file.x0),
    )
    top_left = component_xyts_files[0]
    merged_ny = top_left.ny
    merged_nt = top_left.nt

    xyts_proc_header_size = 72

    xyts_file_descriptors: list[int] = []
    for xyts_file in component_xyts_files:
        xyts_file_descriptor = os.open(xyts_file.xyts_path, os.O_RDONLY)
        # Skip the header for each file descriptor
        head_skip = os.lseek(xyts_file_descriptor, xyts_proc_header_size, os.SEEK_SET)
        if head_skip != xyts_proc_header_size:
            raise ValueError(
                f"Failed to skip header for {xyts_file.xyts_path} at {head_skip}"
            )
        xyts_file_descriptors.append(xyts_file_descriptor)

    # If output doesn't exist when we os.open it, we'll get an error.
    output.touch()
    merged_fd = os.open(output, os.O_WRONLY)

    xyts_header: bytes = (
        top_left.x0.tobytes()
        + top_left.y0.tobytes()
        + top_left.z0.tobytes()
        + top_left.t0.tobytes()
        + top_left.nx.tobytes()
        + top_left.ny.tobytes()
        + top_left.nz.tobytes()
        + top_left.nt.tobytes()
        + top_left.dx.tobytes()
        + top_left.dy.tobytes()
        + top_left.hh.tobytes()
        + top_left.dt.tobytes()
        + top_left.mrot.tobytes()
        + top_left.mlat.tobytes()
        + top_left.mlon.tobytes()
    )

    written = os.write(merged_fd, xyts_header)
    if written != len(xyts_header):
        raise ValueError(
            f"Failed to write header for {output} at {written} bytes written"
        )

    merge_ts_loop.merge_fds(
        merged_fd,
        xyts_file_descriptors,
        merged_nt,
        merged_ny,
        [f.local_nx for f in component_xyts_files],
        [f.local_ny for f in component_xyts_files],
        [f.y0 for f in component_xyts_files],
    )

    for xyts_file_descriptor in xyts_file_descriptors:
        os.close(xyts_file_descriptor)

    os.close(merged_fd)


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
    complevel: int = 4
):
    """Merge XYTS files.

    Parameters
    ----------
    component_xyts_directory : Path
        The input xyts directory containing files to merge.
    output : Path
        The output xyts file.
    glob_pattern : str, optional
        Set a custom glob pattern for merging the xyts files, by default "*xyts-*.e3d".
    complevel : int, optional
        Set the compression level for the output HDF5 file. Range
        between 1-9 (9 being the highest level of compression).
        Defaults to 4.
    """
    component_xyts_files = sorted(
        [
            xyts.XYTSFile(
                xyts_file_path, proc_local_file=True, meta_only=True, round_dt=False
            )
            for xyts_file_path in component_xyts_directory.glob(glob_pattern)
        ],
        key=lambda xyts_file: (xyts_file.y0, xyts_file.x0),
    )
    top_left = component_xyts_files[0]
    nt = top_left.nt
    nx = top_left.nx
    ny = top_left.ny
    components = 3

    xyts_proc_header_size = 72

    waveform_data = np.zeros((nt, ny, nx), dtype=np.uint16)
    for xyts_file in tqdm.tqdm(component_xyts_files, units='files'):
        x0 = xyts_file.x0
        y0 = xyts_file.y0
        x1 = x0 + xyts_file.local_nx
        y1 = y0 + xyts_file.local_ny
        data = np.fromfile(xyts_file.xyts_path, dtype=np.float32, offset=xyts_proc_header_size).reshape(
            (nt, components, xyts_file.local_ny, xyts_file.local_nx)
        )
        magnitude = np.linalg.norm(
            data,
            axis=1
        ) / 0.1
        np.rint(
            magnitude,
            out=waveform_data[:, y0:y1, x0:x1],
            dtype=np.uint16
        )

    proj = coordinates.SphericalProjection(mlon=top_left.mlon, mlat=top_left.mlat, mrot=top_left.mrot)
    dx = top_left.hh
    dt = top_left.dt
    y, x = np.meshgrid(np.arange(ny), np.arange(nx))
    lat, lon = proj.inverse(x.flatten(), y.flatten()).T
    time = np.arange(nt) * dt
    dset = xr.Dataset(
        {
            'waveform': (('time', 'y', 'x'), waveform_data),
        },
        coords={
            'time': ('time', time),
            'y': ('y', np.arange(ny)),
            'x': ('x', np.arange(nx)),
            'latitude': (('y', 'x'), lat),
            'longitude': (('y', 'x'), lon),
        },
        attrs = {
            'dx': dx,
            'dy': dx,
            'dt': dt,
            'mlon': top_left.mlon,
            'mlat': top_left.mlat,
            'mrot': top_left.mrot
        },
    )

    dset['waveform'].attrs.update({
        'scale_factor': 0.1,
        'add_offset': 0.0,
        'units': 'cm/s',
        '_FillValue': -9999,
    })

    dset.to_netcdf(
        output,
        engine='h5netcdf',
        encoding={'waveform': {
            'dtype': 'int16',
            'compression': 'zlib',
            'complevel': complevel,
            'shuffle': True,
            '_FillValue': -9999,
        }},
    )

@cli.from_docstring(app)
def merge_ts(
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
    hdf5: bool = True,
    complevel: int = 4
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
    hdf5 : bool, optional
        If set, save output as a highly compressed HDF5 file. Defaults
        to True.
    complevel : int, optional
        Set the compression level for the output HDF5 file. Range
        between 1-9 (9 being the highest level of compression).
        Defaults to 4.
    """
    if hdf5:
        merge_ts_hdf5(component_xyts_directory, output, glob_pattern, complevel)
    else:
        merge_ts_xyts(component_xyts_directory, output, glob_pattern)
