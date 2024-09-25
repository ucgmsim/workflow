"""Create Simulation Video.

Description
------------
Create a simulation video from the low frequency simulation output.

Inputs
-------
1. A merged timeslice file.

Outputs
--------
1. An animation of the low frequency simulation output. See [youtube](https://www.youtube.com/watch?v=Crdk3k0Prew) for an example of these videos.

Environment
------------
Can be run in the cybershake container. Can also be run from your own computer using the `plot-ts` command which is installed after running `pip install workflow@git+https://github.com/ucgmsim/workflow`. If running on your own computer, you need to install [gmt](https://www.generic-mapping-tools.org/) and [ffmpeg](https://www.ffmpeg.org/). This stage does not run well on Windows, and is very dependent on the gmt version installed. Hypocentre is already setup to run `plot_ts.py` without installing anything.

Usage
------
`plot-ts [OPTIONS] XYTS_INPUT_FILE OUTPUT_FFP`

For More Help
--------------
See the output of `plot-ts --help`.
"""

import tempfile
from pathlib import Path
from typing import Annotated

import numpy as np
import pygmt
import pyvista as pv
import tqdm
import typer
from velocity_modelling.bounding_box import BoundingBox

from pygmt_helper import plotting
from qcore import coordinates
from qcore.xyts import XYTSFile

app = typer.Typer()


def plot_towns(box: BoundingBox, fig: pygmt.Figure) -> None:
    """Plot towns on a GMT figure.

    Parameters
    ----------
    box : BoundingBox
        The region figure.
    fig : pygmt.Figure
        The figure to plot onto.
    """
    towns = {
        "Akaroa": (172.9683333, -43.80361111, "RB"),
        "Blenheim": (173.9569444, -41.5138888, "LM"),
        "Christchurch": (172.6347222, -43.5313888, "LM"),
        "Darfield": (172.1116667, -43.48972222, "CB"),
        "Dunedin": (170.3794444, -45.8644444, "LM"),
        "Greymouth": (171.2063889, -42.4502777, "RM"),
        "Haast": (169.0405556, -43.8808333, "LM"),
        "Kaikoura": (173.6802778, -42.4038888, "LM"),
        "Lyttleton": (172.7194444, -43.60305556, "LM"),
        "Masterton": (175.658333, -40.952778, "LM"),
        "Napier": (176.916667, -39.483333, "LM"),
        "New Plymouth": (174.083333, -39.066667, "RM"),
        "Nelson": (173.2838889, -41.2761111, "CB"),
        "Oxford": (172.1938889, -43.29555556, "LB"),
        "Palmerston North": (175.611667, -40.355000, "RM"),
        "Queenstown": (168.6680556, -45.0300000, "LM"),
        "Rakaia": (172.0230556, -43.75611111, "RT"),
        "Rolleston": (172.3791667, -43.59083333, "RB"),
        "Rotorua": (176.251389, -38.137778, "LM"),
        "Taupo": (176.069400, -38.6875, "LM"),
        "Tekapo": (170.4794444, -44.0069444, "LM"),
        "Timaru": (171.2430556, -44.3958333, "LM"),
        "Wellington": (174.777222, -41.288889, "RM"),
        "Westport": (171.5997222, -41.7575000, "RM"),
    }
    for town_name, (lon, lat, justify) in towns.items():
        if box.contains((lat, lon)):
            fig.plot(
                x=lon,
                y=lat,
                style="c0.3c",
                fill="white",
                pen="black",
            )
            fig.text(text=town_name, x=lon, y=lat, justify=justify, offset="j0.35")


@app.command(help="Plot a low-frequency simulation output as an MP4.")
def animate_low_frequency(
    xyts_ffp: Annotated[
        Path,
        typer.Argument(
            help="Path to a merged XYTS file (see merge-ts --help)",
            exists=True,
            dir_okay=False,
        ),
    ],
    output_mp4: Annotated[
        Path, typer.Argument(help="Path to output mp4", writable=True, dir_okay=False)
    ],
) -> None:
    """Render low-frequency output as a video.

    Parameters
    ----------
    xyts_ffp : Path
        Path to the (merged) low-frequency output file. See `merge_ts.merge_ts`.
    output_mp4 : Path
        Path to output mp4 file.
    """
    xyts_file = XYTSFile(xyts_ffp)

    # Convert geographical coordinates into model coordinates for
    # rendering. The input coordinates are (lat, lon, depth) but these
    # coordinates are not Cartesian. OpenGL (the renderer behind
    # pyvista) expects cartesian coordinates, so we convert (lat, lon)
    # to NZTM. Hence, the model's coordinate space is NZTM coordinates.

    corners = np.array(xyts_file.corners())[:, ::-1]
    nztm_corners = coordinates.wgs_depth_to_nztm(corners)

    # Next create the background map texture to display underneath the
    # low-frequency waves. This is done by taking the axis-aligned
    # bounding box (the region in pygmt), and using pygmt to create a
    # map of the region with towns plotted on top. We then save to a
    # temporary file and load it as an pyvista texture.

    region = (
        corners[:, 1].min() - 0.1,
        corners[:, 1].max() + 0.1,
        corners[:, 0].min() - 0.1,
        corners[:, 0].max() + 0.1,
    )

    aa_bounding_box = BoundingBox(
        coordinates.wgs_depth_to_nztm(
            np.array(
                [
                    # min lat, lon
                    [region[2], region[0]],
                    # min lat, max lon
                    [region[2], region[1]],
                    # max lat, lon
                    [region[-1], region[1]],
                    # max lat, min lon
                    [region[-1], region[0]],
                ]
            )
        )
    )
    with tempfile.NamedTemporaryFile(
        suffix=".png",
    ) as plot_background_ffp:
        fig = plotting.gen_region_fig(None, region)
        plot_towns(aa_bounding_box, fig)

        fig.savefig(plot_background_ffp.name, dpi=1200)
        background_texture = pv.read_texture(plot_background_ffp.name)

    # The seismic waves are rendered as a surface mesh. What is a
    # surface mesh? A surface mesh is a geometric representation of a
    # surface in 3D space (see
    # https://docs.pyvista.org/user-guide/what-is-a-mesh.html). Our
    # surface is the seismic waves. The surface at time t will have
    # coordinates (X, Y, Z(t)) where X, and Y are drawn from a set of
    # equally spaced points across the simulation domain and Z(t) is
    # the ground motion at (X, Y, 0) for time t multiplied by 10,000
    # (so that it is visible in the visualisation).
    meshgrid = np.meshgrid(
        np.linspace(0, 1, xyts_file.nx), np.linspace(0, 1, xyts_file.ny)
    )

    # The above x and y are evenly spaced points from [0..1] x [0..1],
    # but we need to translate these points so that the actually span
    # the simulation domain in NZTM coordinates. The following linear
    # algebra performs the linear transformation required to do this.
    x_trans = nztm_corners[1] - nztm_corners[0]
    y_trans = nztm_corners[-1] - nztm_corners[0]

    # C[i, j, k] = ∑_l A[l, i] B[l, j, k],
    # C is output, A is transformation matrix, M is meshgrid.
    xr, yr = (
        np.einsum("li,ljk->ijk", np.array([x_trans, y_trans]), meshgrid)
        + nztm_corners[0][:, np.newaxis, np.newaxis]
    )

    grid = pv.StructuredGrid(xr, yr, np.zeros_like(yr))
    grid["Ground Motion (cm/s)"] = np.zeros_like(yr).ravel()

    # The background plane holds the picture of the map, and is just a
    # flat plane at the centre of the bounding region.
    background_plane = pv.Plane(
        center=np.append(np.mean(aa_bounding_box.bounds, axis=0), 0),
        i_size=aa_bounding_box.extent_x * 1000,
        j_size=aa_bounding_box.extent_y * 1000,
    )

    # We now setup the pyvista plotter which actually renders the 3D scene.
    plotter = pv.Plotter(off_screen=True, notebook=False)
    plotter.enable_anti_aliasing()
    plotter.ren_win.SetSize((1920, 1088))

    # We now add the seismic wave mesh...
    plotter.add_mesh(
        grid,
        # Where the colour and transparency of the mesh is determined by ground motion and,
        scalars="Ground Motion (cm/s)",
        lighting=False,
        # the opacity of the mesh is 0 for ground motion up to ~1cm/s and then 1 thereafter and,
        opacity=pv.opacity_transfer_function(
            np.array([0] * 10 + [1] * 90), n_colors=100
        ),
        show_edges=True,
        # the colour map is hot and,
        cmap="hot",
        # we display a colour bar of the ground motion values vertically and,
        scalar_bar_args={"vertical": True},
        # the bar runs from 0 - 10 cm/s.
        clim=[0, 10],
    )

    # The background plane is simply added to the plot with the pygmt
    # background texture texture-mapped onto it.
    plotter.add_mesh(background_plane, show_edges=True, texture=background_texture)

    # We now reset the camera so it points down at the background
    # plane and zooms to fill.
    plotter.reset_camera(bounds=background_plane.bounds)
    plotter.view_xy()
    # TODO: Dynamically calculate an appropriate camera zoom.
    plotter.camera.zoom(1.4)

    plotter.open_movie(output_mp4, framerate=10, quality=10)

    # For each timestep, update the ground motion to be the ground
    # motion for the current timestep, and update the time counter in
    # the top right roughly every second.
    its_per_second = round(1 / xyts_file.dt)
    for i in tqdm.trange(xyts_file.nt):
        ground_motion = np.linalg.norm(xyts_file.data[i, :, :, ::-1], axis=0).T
        grid.points[:, -1] = ground_motion.ravel() * 1e4
        grid["Ground Motion (cm/s)"] = ground_motion.ravel()

        if i % its_per_second == 0:
            plotter.add_text(
                f"{i * xyts_file.dt:.2f}s",
                position="upper_right",
                name="Second Counter",
            )

        plotter.write_frame()
    plotter.close()


if __name__ == "__main__":
    app()
