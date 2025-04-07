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

from qcore import cli, xyts
from workflow.scripts import merge_ts_loop

app = typer.Typer()


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
    glob_pattern: Annotated[str, typer.Option()] = "*xyts-*.e3d",
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
