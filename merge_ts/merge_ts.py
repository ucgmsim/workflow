#!/usr/bin/env python3
"""
Module for merging XYTS files.

This module provides functionality for merging XYTS files. It takes multiple
input XYTS files, representing small patches of a larger simulation domain, and
merges them into one large output.

$ merge_ts XYTS_DIRECTORY XYTS_DIRECTORY/output.e3d

Note:
    This module assumes the input XYTS files have the same temporal dimensions
    (i.e. nt is constant).
"""
import os
from pathlib import Path

import typer
from qcore import xyts
from typing_extensions import Annotated

from merge_ts import merge_ts_loop


def merge_ts(
    component_xyts_directory: Annotated[
        Path,
        typer.Argument(
            help="The input xyts directory containing files to merge",
            dir_okay=True,
            file_okay=False,
            exists=True,
            readable=True,
        ),
    ],
    output: Annotated[
        Path,
        typer.Argument(help="The output xyts file", dir_okay=False, writable=True),
    ],
    glob_pattern: Annotated[
        str, typer.Option(help="Set a custom glob pattern for merging the xyts files")
    ] = "*xyts-*.e3d",
):
    """Merge XYTS files."""

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

    xyts_file_descriptors = []
    for xyts_file in component_xyts_files:
        xyts_file_descriptor = os.open(xyts_file.xyts_path, os.O_RDONLY)
        # Skip the header for each file descriptor
        os.lseek(xyts_file_descriptor, xyts_proc_header_size, os.SEEK_SET)
        xyts_file_descriptors.append(xyts_file_descriptor)

    # If output doesn't exist when we os.open it, we'll get an error.
    output.touch()
    merged_fd = os.open(output, os.O_WRONLY)

    xyts_header = (
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
    os.write(merged_fd, xyts_header)
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


# The following function is here to define an entrypoint for the setup.py file.
def main():
    """Main script entrypoint."""
    typer.run(merge_ts)


if __name__ == "__main__":
    main()
