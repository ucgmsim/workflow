"""Create Simulation Video.

Description
-----------
Create a simulation video from the low frequency simulation output.

Inputs
------
1. A merged timeslice file.

Outputs
-------
1. An animation of the low frequency simulation output. See [youtube](https://www.youtube.com/watch?v=Crdk3k0Prew) for an example of these videos.

Environment
-----------
Can be run in the cybershake container. Can also be run from your own computer using the `plot-ts` command which is installed after running `pip install workflow@git+https://github.com/ucgmsim/workflow`. If running on your own computer, you need to install [gmt](https://www.generic-mapping-tools.org/) and [ffmpeg](https://www.ffmpeg.org/). This stage does not run well on Windows, and is very dependent on the gmt version installed. Hypocentre is already setup to run `plot_ts.py` without installing anything.

Usage
-----
`plot-ts [OPTIONS] SRF_FFP XYTS_INPUT_DIRECTORY OUTPUT_FFP`

For More Help
-------------
See the output of `plot-ts --help`.
"""

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
    ] = Path("/out"),
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
