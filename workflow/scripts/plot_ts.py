import os
import subprocess
from argparse import ArgumentParser
from multiprocessing import Pool
from pathlib import Path
from shutil import rmtree
from tempfile import mkdtemp
from time import time
from typing import Annotated

import numpy as np
import qcore.gmt as gmt
import typer
from qcore.xyts import XYTSFile

from merge_ts import merge_ts


def plot_ts(
    srf_file: Annotated[
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

    # size of plotting area
    PAGE_WIDTH = 16
    PAGE_HEIGHT = 9
    # space around map for titles, tick labels and scales etc
    MARGIN_TOP = 1.0
    MARGIN_BOTTOM = 0.4
    MARGIN_LEFT = 1.0
    MARGIN_RIGHT = 1.7

    # prepare temp locations
    png_dir = work_directory / "TS_PNG"
    png_dir.mkdir(exist_ok=True)
    cpt_file = work_directory / "motion.cpt"
    pgv_file = work_directory / "PGV.bin"
    # load xyts
    merged_xyts = XYTSFile(merged_xyts_ffp)
    merged_xyts.pgv(pgvout=pgv_file)
    cpt_min = 0
    cpt_inc, cpt_max = gmt.xyv_cpt_range(pgv_file)[1:3]
    lowcut = cpt_max * 0.02
    highcut = None
    convergence_limit = cpt_inc * 0.2
    corners, cnr_str = merged_xyts.corners(gmt_format=True)
    region = merged_xyts.region()
    region_code = gmt.get_region(region[0], region[2])
    ll_region = merged_xyts.region(corners=corners)
    grd_dxy = "%sk" % (merged_xyts.dx / 2.0)
    # determine map sizing
    map_width = PAGE_WIDTH - MARGIN_LEFT - MARGIN_RIGHT
    map_height = PAGE_HEIGHT - MARGIN_TOP - MARGIN_BOTTOM
    # extend region to fill view window
    map_width, map_height, ll_region = gmt.fill_space(
        map_width, map_height, ll_region, proj="M", dpi=dpi, wd=work_directory
    )
    # extend map to cover margins
    if not border:
        map_width_a, map_height_a, borderless_region = gmt.fill_margins(
            ll_region,
            map_width,
            dpi,
            left=MARGIN_LEFT,
            right=MARGIN_RIGHT,
            top=MARGIN_TOP,
            bottom=MARGIN_BOTTOM,
        )
    # colour scale
    gmt.makecpt(
        cpt, cpt_file, cpt_min, cpt_max, inc=cpt_inc, invert=True, bg=None, fg=None
    )

    def bottom_template():
        t0 = time()
        bwd = work_directory / "bottom"
        bwd.mkdir(exist_ok=True)
        b = gmt.GMTPlot(bwd / "bottom.ps")
        gmt.gmt_defaults(wd=bwd, ps_media=f"Custom_{PAGE_WIDTH}ix{PAGE_HEIGHT}i")
        if border:
            b.background(PAGE_WIDTH, PAGE_HEIGHT, colour="white")
        else:
            b.spacial("M", borderless_region, sizing=map_width_a)
            # topo, water, overlay cpt scale
            b.basemap(land="lightgray", topo_cpt="grey1", resource_region=region_code)
            # map margins are semi-transparent
            b.background(
                map_width_a,
                map_height_a,
                colour="white@25",
                spacial=True,
                window=(MARGIN_LEFT, MARGIN_RIGHT, MARGIN_TOP, MARGIN_BOTTOM),
            )
        # leave space for left tickmarks and bottom colour scale
        b.spacial(
            "M", ll_region, sizing=map_width, x_shift=MARGIN_LEFT, y_shift=MARGIN_BOTTOM
        )
        if border:
            # topo, water, overlay cpt scale
            b.basemap(land="lightgray", topo_cpt="grey1", resource_region=region_code)
        # title, fault model and velocity model subtitles
        b.text(sum(ll_region[:2]) / 2.0, ll_region[3], title, size=20, dy=0.6)

        # cpt scale
        b.cpt_scale(
            "R",
            "B",
            cpt_file,
            cpt_inc,
            cpt_inc,
            label=legend,
            length=map_height,
            horiz=False,
            pos="rel_out",
            align="LB",
            thickness=0.3,
            dx=0.3,
            arrow_f=cpt_max > 0,
            arrow_b=cpt_min < 0,
        )
        # render
        b.finalise()
        b.png(dpi=dpi, clip=False, out_dir=work_directory)
        rmtree(bwd)
        print("bottom template completed in %.2fs" % (time() - t0))

    def top_template():
        t0 = time()
        twd = work_directory / "top"
        os.makedirs(twd)
        t = gmt.GMTPlot("%s/top.ps" % (twd))
        gmt.gmt_defaults(wd=twd, ps_media="Custom_%six%si" % (PAGE_WIDTH, PAGE_HEIGHT))
        t.spacial(
            "M", ll_region, sizing=map_width, x_shift=MARGIN_LEFT, y_shift=MARGIN_BOTTOM
        )
        # locations in NZ
        if ll_region[1] - ll_region[0] > 3:
            t.sites(gmt.sites_major)
        else:
            t.sites(list(gmt.sites.keys()))
        t.coastlines()
        # simulation domain
        t.path(
            cnr_str, is_file=False, split="-", close=True, width="0.4p", colour="black"
        )
        # fault path
        if args.srf is not None:
            t.fault(args.srf, is_srf=True, plane_width=0.5, top_width=1, hyp_width=0.5)
        # ticks on top otherwise parts of map border may be drawn over
        major, minor = gmt.auto_tick(ll_region[0], ll_region[1], map_width)
        t.ticks(major=major, minor=minor, sides="ws")
        # render
        t.finalise()
        t.png(dpi=args.dpi, clip=False, out_dir=gmt_temp)
        rmtree(twd)
        print("top template completed in %.2fs" % (time() - t0))

    def render_slice(n):
        t0 = time()

        # process working directory
        swd = "%s/ts%.4d" % (gmt_temp, n)
        os.makedirs(swd)

        s = gmt.GMTPlot("%s/ts%.4d.ps" % (swd, n), reset=False)
        gmt.gmt_defaults(wd=swd, ps_media="Custom_%six%si" % (PAGE_WIDTH, PAGE_HEIGHT))
        s.spacial(
            "M", ll_region, sizing=map_width, x_shift=MARGIN_LEFT, y_shift=MARGIN_BOTTOM
        )

        # timestamp text
        s.text(
            ll_region[1],
            ll_region[3],
            "t=%.2fs" % (n * merged_xyts.dt),
            align="RB",
            size="14p",
            dy=0.1,
        )
        # overlay
        if args.xyts2 is None:
            merged_xyts.tslice_get(n, comp=-1, outfile="%s/ts.bin" % (swd))
        else:
            x1 = merged_xyts.tslice_get(n, comp=-1)
            x2 = xyts2.tslice_get(n, comp=-1)
            x2[:, 2] -= x1[:, 2]
            x2.astype(np.float32).tofile(os.path.join(swd, "ts.bin"))
        s.clip(cnr_str, is_file=False)
        if args.land_crop:
            s.clip(gmt.regional_resource("NZ", resource="coastline"), is_file=True)
        gmt.table2grd(
            os.path.join(swd, "ts.bin"),
            os.path.join(swd, "ts.grd"),
            dx=grd_dxy,
            wd=swd,
            climit=convergence_limit,
        )
        s.overlay(
            os.path.join(swd, "ts.grd"),
            cpt_file,
            dx=grd_dxy,
            dy=grd_dxy,
            min_v=lowcut,
            max_v=highcut,
            contours=cpt_inc,
        )
        s.clip()

        # add seismograms if wanted
        if args.seis is not None:
            # TODO: if used again, look into storing params inside seismo file
            s.seismo(
                os.path.abspath(args.seis), n, fmt="time", colour="red", width="1p"
            )

        # create PNG
        s.finalise()
        s.png(dpi=args.dpi, clip=False, out_dir=png_dir)
        # cleanup
        rmtree(swd)
        print("timeslice %.4d completed in %.2fs" % (n, time() - t0))

    def combine_slice(n):
        """
        Sandwitch midde layer (time dependent) between basemap and top (labels etc).
        """
        png = "%s/ts%.4d.png" % (png_dir, n)
        mid = "%s/bm%.4d.png" % (png_dir, n)
        gmt.overlay("%s/bottom.png" % (gmt_temp), png, mid)
        gmt.overlay(mid, "%s/top.png" % (gmt_temp), png)

    ###
    ### start rendering
    ###
    with Pool() as pool:
        # shared bottom and top layers
        b_template = pool.apply_async(bottom_template, ())
        t_template = pool.apply_async(top_template, ())
        # middle layers
        pool.map(
            render_slice,
            range(merged_xyts.t0, merged_xyts.nt - merged_xyts.t0),
        )
        # wait for bottom and top layers
        print("waiting for templates to finish...")
        t_template.get()
        b_template.get()
        print("templates finished, combining layers...")
        # combine layers
        pool.map(
            combine_slice,
            range(merged_xyts.t0, merged_xyts.nt - merged_xyts.t0),
        )
        print("layers combined, creating animation...")
        # images -> animation
        gmt.make_movie(
            "%s/ts%%04d.png" % (png_dir),
            args.output,
            fps=int(1.0 / merged_xyts.dt * args.scale),
            codec="libx264",
        )
        print("finished.")
        # cleanup
        rmtree(gmt_temp)
