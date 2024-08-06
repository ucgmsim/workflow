import subprocess

from merge_ts import merge_ts


def plot_timeslice(
    srf_file: srf.SrfFile,
    xyts_file: XYTSFile,
    work_directory: Path,
    pgv_limits: tuple[float, float, float],
    timeslice: int,
) -> None:
    import pygmt

    reload(pygmt)
    corners = np.array(xyts_file.corners())
    region = (
        corners[:, 0].min() - 0.5,
        corners[:, 0].max() + 0.5,
        corners[:, 1].min() - 0.25,
        corners[:, 1].max() + 0.25,
    )

    fig = plotting.gen_region_fig(region=region, map_data=None)
    i = 0
    for _, segment in srf_file.header.iterrows():
        nstk = int(segment["nstk"])
        ndip = int(segment["ndip"])
        point_count = nstk * ndip
        segment_points = srf_file.points.iloc[i : i + point_count]
        corners = segment_points.iloc[[0, nstk - 1, -1, (ndip - 1) * nstk]]
        fig.plot(
            x=corners["lon"].iloc[list(range(len(corners))) + [0]].to_list(),
            y=corners["lat"].iloc[list(range(len(corners))) + [0]].to_list(),
            pen="0.5p,black,-",
        )

        i += point_count

    tslice = pd.DataFrame(
        data=xyts_file.tslice_get(timeslice), columns=["lon", "lat", "motion"]
    )
    _, pgv_max, _ = pgv_limits
    cur_grid = plotting.create_grid(
        tslice[tslice["motion"] > 0.02 * pgv_max],
        "motion",
        grid_spacing="100e/100e",
        region=(
            tslice["lon"].min(),
            tslice["lon"].max(),
            tslice["lat"].min(),
            tslice["lat"].max(),
        ),
        set_water_to_nan=False,
    )

    plotting.plot_grid(
        fig,
        cur_grid,
        "hot",
        pgv_limits,
        ("white", "black"),
        transparency=50,
        reverse_cmap=True,
        cb_label="ground motion [cm/s]",
        continuous_cmap=True,
    )

    fig.grdcontour(
        annotation="-",
        interval=1,
        grid=cur_grid,
        pen="0.1p",
    )
    fig.savefig(work_directory / f"ts{timeslice:05d}.png", dpi=120, anti_alias=True)


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
    border: Annotated[bool, typer.Option(help="Opaque map margins")] = True,
    scale: Annotated[
        float, typer.Option(help="Speed of animation (multiple of real time).")
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
    pgv = pd.DataFrame(xyts_file.pgv(), columns=["lon", "lat", "pgv"])

    pgv_quantile = pgv["pgv"].quantile(0.995)
    pgv_cb_max = max(int(np.round(pgv_quantile, -1)), 10)
    cmap_limits = (0, pgv_cb_max, pgv_cb_max / 10)
    with multiprocessing.Pool() as pool:
        pool.map(
            functools.partial(
                plot_timeslice, srf_file, xyts_file, work_directory, cmap_limits
            ),
            range(xyts_file.nt),
        )


def main():
    typer.run(plot_ts)


if __name__ == "__main__":
    main()
