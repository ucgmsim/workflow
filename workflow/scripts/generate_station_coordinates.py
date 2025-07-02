"""Station Selection.

Description
-----------
Filter a station list for in-domain stations to simulate high frequency and broadband output for.

Inputs
------
1. A station list and,
2. A realisation file containing domain parameters.

Outputs
-------
1. A station list containing only stations in-domain and with unique discretised coordinate positions in two formats:
   - Stations in the format "longitude latitude name" format in "stations.ll",
   - Stations in the format "x y name" format in "stations.statcord". The x and y are the discretised positions of each station in the domain.

Environment
-----------
Can be run in the cybershake container. Can also be run from your own computer using the `generate-station-coordinates` command which is installed after running `pip install workflow@git+https://github.com/ucgmsim/workflow`. If you do run this on your own computer, you need a version of `ll2gp` installed.

Usage
-----
`generate-station-coordinates [OPTIONS] REALISATIONS_FFP STAT_FILE OUTPUT_PATH`

For More Help
-------------
See the output of `generate-station-coordinates --help`.
"""

from pathlib import Path
from typing import Annotated

import pandas as pd
import typer

from qcore import cli, coordinates
from workflow import log_utils, realisations
from workflow.realisations import DomainParameters

app = typer.Typer()


@cli.from_docstring(app)
@log_utils.log_call()
def generate_fd_files(
    realisation_ffp: Annotated[Path, typer.Argument(readable=True, dir_okay=False)],
    stat_file: Annotated[Path, typer.Argument(readable=True, dir_okay=False)],
    output_path: Annotated[Path, typer.Argument(file_okay=False, writable=True)],
) -> None:
    """Generate station coordinate files.

    Parameters
    ----------
    realisation_ffp : Path
        Path to realisation json file.
    stat_file : Path
        The location of the station files.
    output_path : Path
        Output path for station files.
    """
    output_path.mkdir(exist_ok=True)
    domain_parameters = DomainParameters.read_from_realisation(realisation_ffp)
    domain = domain_parameters.domain

    nx = domain_parameters.nx
    ny = domain_parameters.ny
    mlat, mlon = domain.origin
    mrot = domain.bearing
    proj = coordinates.SphericalProjection(mlat=mlat, mlon=mlon, mrot=mrot)

    # where to save gridpoint and longlat station files
    gp_out = output_path / "stations.statcords"
    ll_out = output_path / "stations.ll"

    # retrieve in station names, latitudes and longitudes
    stations = pd.read_csv(
        stat_file, delimiter=r"\s+", comment="#", names=["lon", "lat", "name"]
    )

    x, y = proj(lat=stations["lat"].values, lon=stations["lon"].values)

    cx = nx // 2 * domain_parameters.resolution
    cy = ny // 2 * domain_parameters.resolution

    # translate coordinates so that top-left corner of the domain is at (0, 0)
    x += cx
    y += cy

    # C-compatible rounding of the continuous coordinates into grid point coordinates
    x = (x / domain_parameters.resolution + 0.5).astype(int)
    y = (y / domain_parameters.resolution + 0.5).astype(int)

    in_domain_mask = (
        (x >= 0) & (x < domain_parameters.nx) & (y >= 0) & (y < domain_parameters.ny)
    )
    # filter out stations outside the domain
    stations = stations.loc[in_domain_mask]

    if len(stations) == 0:
        raise ValueError("No stations in domain.")

    x = x[in_domain_mask]
    y = y[in_domain_mask]
    stations["x"] = x
    stations["y"] = y

    gp_x = x * domain_parameters.resolution - cx
    gp_y = y * domain_parameters.resolution - cy
    gp_lat, gp_lon = proj.inverse(gp_x, gp_y)
    stations["grid_lat"] = gp_lat
    stations["grid_lon"] = gp_lon

    # create grid point file
    with open(gp_out, "w", encoding="utf-8") as gpf:
        # file starts with number of entries
        gpf.write(f"{len(stations)}\n")
        # x, y, z, name
        stations.apply(
            lambda station: gpf.write(
                f"{station['x']:5d} {station['y']:5d} {1:5d} {station['name']}\n"
            ),
            axis=1,
        )

    # create ll file
    with open(ll_out, "w", encoding="utf-8") as llf:
        stations.apply(
            lambda station: llf.write(
                f"{station['grid_lon']:11.5f} {station['grid_lat']:11.5f} {station['name']}\n"
            ),
            axis=1,
        )

    realisations.append_log_entry(realisation_ffp)
