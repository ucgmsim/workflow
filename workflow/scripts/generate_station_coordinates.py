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

from enum import StrEnum, auto
from pathlib import Path
from typing import Annotated

import h5py
import numpy as np
import pandas as pd
import typer

from qcore import cli, coordinates
from workflow import log_utils, realisations
from workflow.realisations import DomainParameters, Resolution

app = typer.Typer()


class Format(StrEnum):
    EMOD3D = auto()
    SW4 = auto()


def write_ascii_station_locations(
    stations: pd.DataFrame,
    ll_out: Path,
    lon_col: str = "lon",
    lat_col: str = "lat",
    name_col: str = "name",
) -> None:
    with open(ll_out, "w", encoding="utf-8") as llf:
        stations.apply(
            lambda station: llf.write(
                f"{station[lon_col]:11.5f} {station[lat_col]:11.5f} {station[name_col]}\n"
            ),
            axis=1,
        )


def write_emod3d_station_format(
    domain_parameters: DomainParameters,
    resolution_parameters: Resolution,
    stations: pd.DataFrame,
    output_path: Path,
) -> None:
    """Write station coordinates in EMOD3D format to two output files.

    Parameters
    ----------
    domain_parameters : DomainParameters
        Object containing domain definition and methods to compute grid dimensions.
    resolution_parameters : Resolution
        Object containing the grid resolution.
    stations : pd.DataFrame
        DataFrame with columns `lat`, `lon`, and `name`. Latitude and longitude
        are in degrees.
    output_path : Path
        Directory path where the output files will be written.
    """
    domain = domain_parameters.domain
    nx = domain_parameters.nx(resolution_parameters.resolution)
    ny = domain_parameters.ny(resolution_parameters.resolution)
    mlat, mlon = domain.origin
    mrot = domain.great_circle_bearing
    proj = coordinates.SphericalProjection(mlat=mlat, mlon=mlon, mrot=float(mrot))

    # where to save gridpoint and longlat station files
    gp_out = output_path / "stations.statcords"
    ll_out = output_path / "stations.ll"

    x, y = proj(
        lat=stations["lat"].to_numpy(float), lon=stations["lon"].to_numpy(float)
    ).T

    cx = nx // 2 * resolution_parameters.resolution
    cy = ny // 2 * resolution_parameters.resolution

    # translate coordinates so that top-left corner of the domain is at (0, 0)
    x += cx
    y += cy

    # C-compatible rounding of the continuous coordinates into grid point coordinates
    x = (x / resolution_parameters.resolution + 0.5).astype(int)
    y = (y / resolution_parameters.resolution + 0.5).astype(int)

    in_domain_mask = (x >= 0) & (x < nx) & (y >= 0) & (y < ny)
    # filter out stations outside the domain
    stations = stations.loc[in_domain_mask]

    if len(stations) == 0:
        raise ValueError("No stations in domain.")

    x = x[in_domain_mask]
    y = y[in_domain_mask]
    stations["x"] = x
    stations["y"] = y

    gp_x = x * resolution_parameters.resolution - cx
    gp_y = y * resolution_parameters.resolution - cy
    gp_lat, gp_lon = proj.inverse(gp_x, gp_y).T
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

    write_ascii_station_locations(
        stations, ll_out, lon_col="grid_lon", lat_col="grid_lat"
    )


def write_sw4_station_format(stations: pd.DataFrame, output_path: Path) -> None:
    with h5py.File(output_path / "stations.h5", "w") as f:
        for station_name, position in stations.set_index("name").iterrows():
            station_dset = f.create_group(station_name)
            location = station_dset.create_dataset(
                "STLA,STLO,STDP", (3), dtype=np.float64
            )
            location[0] = position["lat"]
            location[1] = position["lon"]
            location[2] = 0.0

    write_ascii_station_locations(stations, output_path / "stations.ll")


@cli.from_docstring(app)
@log_utils.log_call()
def generate_fd_files(
    realisation_ffp: Annotated[Path, typer.Argument(readable=True, dir_okay=False)],
    stat_file: Annotated[Path, typer.Argument(readable=True, dir_okay=False)],
    output_path: Annotated[Path, typer.Argument(file_okay=False, writable=True)],
    format: Format = Format.EMOD3D,
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
    resolution_parameters = Resolution.read_from_realisation(realisation_ffp)

    # retrieve in station names, latitudes and longitudes
    stations = pd.read_csv(
        stat_file,
        delimiter=r"\s+",
        comment="#",
        names=["lon", "lat", "name"],
    )

    match format:
        case Format.EMOD3D:
            write_emod3d_station_format(
                domain_parameters, resolution_parameters, stations, output_path
            )
        case Format.SW4:
            write_sw4_station_format(stations, output_path)

    realisations.append_log_entry(realisation_ffp)
