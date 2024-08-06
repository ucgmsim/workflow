import functools
import multiprocessing
from pathlib import Path
from typing import Annotated

import numpy as np
import typer
from pygmt_helper import plotting
from qcore.xyts import XYTSFile
from source_modelling import srf

from merge_ts import merge_ts


def plot_timeslice(
    srf_file: srf.SrfFile, xyts_file: XYTSFile, work_directory: Path, timeslice: int
) -> None:
    corners = np.array(xyts_file.corners())
    region = (
        corners[:, 0].min() - 0.5,
        corners[:, 0].max() + 0.5,
        corners[:, 1].min() - 0.25,
        corners[:, 1].max() + 0.25,
    )
    slip_quantile = srf_file.points["slip"].quantile(0.98)
    slip_cb_max = max(int(np.round(slip_quantile, -1)), 10)
    cmap_limits = (0, slip_cb_max, slip_cb_max / 10)

    fig = plotting.gen_region_fig(region=region, map_data=None)
    i = 0
    for _, segment in srf_file.header.iterrows():
        nstk = int(segment["nstk"])
        ndip = int(segment["ndip"])
        point_count = nstk * ndip
        segment_points = srf_file.points.iloc[i : i + point_count]
        cur_grid = plotting.create_grid(
            segment_points,
            "slip",
            grid_spacing="5e/5e",
            region=(
                segment_points["lon"].min(),
                segment_points["lon"].max(),
                segment_points["lat"].min(),
                segment_points["lat"].max(),
            ),
            set_water_to_nan=False,
        )
        plotting.plot_grid(
            fig,
            cur_grid,
            "hot",
            cmap_limits,
            ("white", "black"),
            transparency=0,
            reverse_cmap=True,
            plot_contours=False,
            cb_label="slip",
            continuous_cmap=True,
        )
        corners = segment_points.iloc[[0, nstk - 1, -1, (ndip - 1) * nstk]]
        fig.plot(
            x=corners["lon"].iloc[list(range(len(corners))) + [0]].to_list(),
            y=corners["lat"].iloc[list(range(len(corners))) + [0]].to_list(),
            pen="0.5p,black,-",
        )

        i += point_count


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
    cpt: Annotated[str, typer.Option(help="XYTS overlay CPT.")] = "hot",
    dpi: Annotated[int, typer.Option(help="Output video DPI.")] = 120,
    title: Annotated[
        str, typer.Option(help="Video title")
    ] = "Automatically Generated Event",
    legend: Annotated[
        str, typer.Option(help="Colour scale legend text.")
    ] = "sim2 - sim1 ground motion [cm/s]",
    border: Annotated[bool, typer.Option("Opaque map margins")] = True,
    scale: Annotated[
        float, typer.Option("Speed of animation (multiple of real time).")
    ] = 1.0,
):
    merged_xyts_ffp = work_directory / "timeslices-xyts.e3d"
    merge_ts.merge_ts(xyts_input_directory, merged_xyts_ffp)
    xyts_file = XYTSFile(merged_xyts_ffp)
    srf_file = srf.read_srf(srf_ffp)

    srf_file.points["slip"] = np.sqrt(
        srf_file.points["slip1"] ** 2
        + srf_file.points["slip2"] ** 2
        + srf_file.points["slip3"] ** 2
    )
    with multiprocessing.Pool() as pool:
        pool.map(
            functools.partial(
                plot_timeslice,
                srf_file,
                xyts_file,
                work_directory,
            ),
            range(xyts_file.nt),
        )
