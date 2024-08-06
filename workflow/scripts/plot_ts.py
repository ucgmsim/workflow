import multiprocessing
import subprocess
from pathlib import Path
from typing import Annotated

import typer

from merge_ts import merge_ts


def plot_ts(
    srf_ffp: Annotated[
        Path, typer.Argument(help="Path to SRF file", exists=True, dir_okay=False)
    ],
    xyts_input_directory: Annotated[
        Path,
        typer.Argument(
            help="Path to xyts files to plot.", exists=True, file_okay=False
        ),
    ],
    output_ffp: Annotated[
        Path, typer.Argument(help="Path to save output animation", writable=True)
    ],
    work_directory: Annotated[
        Path, typer.Option(help="Intermediate output directory")
    ] = "/out",
):
    merged_xyts_ffp = work_directory / "timeslices-xyts.e3d"
    merge_ts.merge_ts(xyts_input_directory, merged_xyts_ffp)
    subprocess.check_call(
        [
            "plot_ts.py",
            "-n",
            str(multiprocessing.cpu_count()),
            str(merged_xyts_ffp),
            "--output",
            str(work_directory),
        ]
    )


def main():
    typer.run(plot_ts)


if __name__ == "__main__":
    main()
