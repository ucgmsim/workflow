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
`generate-station-coordinates [OPTIONS] REALISATIONS_FFP OUTPUT_PATH`

For More Help
-------------
See the output of `generate-station-coordinates --help`.
"""

from pathlib import Path
from typing import Annotated

import numpy as np
import pandas as pd
import typer

from workflow import log_utils
from workflow.realisations import DomainParameters

app = typer.Typer()


@app.command(help="Generate station gridpoint coordinates from a list of stations.")
@log_utils.log_call
def generate_fd_files(
    realisations_ffp: Annotated[
        Path, typer.Argument(help="Path to realisation json file.", readable=True)
    ],
    output_path: Annotated[
        Path,
        typer.Argument(
            help="Output path for station files", file_okay=False, writable=True
        ),
    ],
    keep_dup_station: Annotated[
        bool,
        typer.Option(help="Keep stations whose gridpoint coordinates are identical"),
    ] = True,
    stat_file: Annotated[
        Path,
        typer.Option(
            help="The location of the station files.", readable=True, exists=True
        ),
    ] = Path("/input/stations.ll"),
) -> None:
    """Generate station coordinate files.

    Parameters
    ----------
    realisations_ffp : Path
        Path to realisation file.
    output_path : Path
        Output directory for station coordinate and latitude longitude files.
    keep_dup_station : bool
        If True, keep stations whose gridpoint coordinates are identical.
    stat_file : Path
        If True, keep stations whose gridpoint coordinates are identical.
    """
    output_path.mkdir(exist_ok=True)
    domain_parameters = DomainParameters.read_from_realisation(realisations_ffp)
    print(domain_parameters.domain)

    # where to save gridpoint and longlat station files
    gp_out = output_path / "stations.statcords"
    ll_out = output_path / "stations.ll"

    # retrieve in station names, latitudes and longitudes
    stations = pd.read_csv(
        stat_file, delimiter=r"\s+", comment="#", names=["lon", "lat", "name"]
    )

    in_domain_mask = domain_parameters.domain.contains(
        stations[["lat", "lon"]].to_numpy()
    )
    stations = stations.loc[in_domain_mask]
    # convert ll to grid points
    xy = domain_parameters.domain.wgs_depth_coordinates_to_local_coordinates(
        stations[["lat", "lon"]].to_numpy()
    )

    stations["x"] = np.round(
        domain_parameters.domain.extent_x * xy[:, 0] / domain_parameters.resolution
    ).astype(int)
    # the bounding box local coordinates start from the left bottom, so we flip that so that it starts from the top left
    stations["y"] = np.round(
        domain_parameters.domain.extent_y
        * (1 - xy[:, 1])
        / domain_parameters.resolution
    ).astype(int)
    # store gridpoints and names if unique position

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

    # convert unique grid points back to ll
    # warning: modifies sxy
    stations["y"] = (domain_parameters.ny - 1) - stations["y"]

    ll = domain_parameters.domain.local_coordinates_to_wgs_depth(
        stations[["x", "y"]].to_numpy()
        * domain_parameters.resolution
        / np.array(
            [domain_parameters.domain.extent_x, domain_parameters.domain.extent_y]
        )
    )
    stations["grid_lat"] = ll[:, 0]
    stations["grid_lon"] = ll[:, 1]

    # create ll file
    with open(ll_out, "w", encoding="utf-8") as llf:
        stations.apply(
            lambda station: llf.write(
                f"{station['grid_lon']:11.5f} {station['grid_lat']:11.5f} {station['name']}\n"
            ),
            axis=1,
        )
