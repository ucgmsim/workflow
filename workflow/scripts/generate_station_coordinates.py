"""Station Selection.

Description
-----------
Filter a site table for in-domain stations to simulate high frequency and broadband output for.

Inputs
------
1. A site table (parquet, see `workflow.sites`) and,
2. A realisation file containing domain parameters.

Outputs
-------
1. The site table restricted to in-domain stations, in "stations.parquet". For
   EMOD3D, each station's location is that of the grid point it is snapped to.
2. The solver's station file:
   - EMOD3D: stations in the format "x y name" format in "stations.statcords". The x and y are the discretised positions of each station in the domain.
   - SW4: an HDF5 station file, "stations.h5".

Environment
-----------
Can be run in the cybershake container. Can also be run from your own computer using the `generate-station-coordinates` command which is installed after running `pip install ucgmsim-workflow@git+https://github.com/ucgmsim/workflow`. If you do run this on your own computer, you need a version of `ll2gp` installed.

Usage
-----
`generate-station-coordinates [OPTIONS] REALISATIONS_FFP SITES_FFP OUTPUT_PATH`

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
import shapely
import typer

from qcore import cli, coordinates
from workflow import log_utils, realisations, sites
from workflow.realisations import DomainParameters, Resolution

app = typer.Typer()


class Format(StrEnum):
    """The solver the station files are written for."""

    EMOD3D = auto()
    """EMOD3D `.statcords` grid-point files."""
    SW4 = auto()
    """SW4 HDF5 station file."""


def write_emod3d_station_format(
    domain_parameters: DomainParameters,
    resolution_parameters: Resolution,
    stations: pd.DataFrame,
    output_path: Path,
) -> None:
    """Write station coordinates in EMOD3D format.

    Writes the grid point of every in-domain station to `stations.statcords`,
    and the site table of those stations, located at their grid points, to
    `stations.parquet`.

    Parameters
    ----------
    domain_parameters : DomainParameters
        Object containing domain definition and methods to compute grid dimensions.
    resolution_parameters : Resolution
        Object containing the grid resolution.
    stations : pd.DataFrame
        The site table (see `workflow.sites`).
    output_path : Path
        Directory path where the output files will be written.
    """
    domain = domain_parameters.domain
    nx = domain_parameters.nx(resolution_parameters.resolution)
    ny = domain_parameters.ny(resolution_parameters.resolution)
    mlat, mlon = domain.origin
    mrot = domain.great_circle_bearing
    proj = coordinates.SphericalProjection(mlat=mlat, mlon=mlon, mrot=float(mrot))

    gp_out = output_path / "stations.statcords"

    x, y = proj(
        lat=stations["latitude"].to_numpy(float),
        lon=stations["longitude"].to_numpy(float),
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
    stations = stations.loc[in_domain_mask].copy()

    if len(stations) == 0:
        raise ValueError("No stations in domain.")

    x = x[in_domain_mask]
    y = y[in_domain_mask]

    # create grid point file
    with open(gp_out, "w", encoding="utf-8") as gpf:
        # file starts with number of entries
        gpf.write(f"{len(stations)}\n")
        # x, y, z, name
        gpf.writelines(
            f"{station_x:5d} {station_y:5d} {1:5d} {name}\n"
            for station_x, station_y, name in zip(x, y, stations.index)
        )

    # The simulations downstream see each station where EMOD3D records it.
    gp_x = x * resolution_parameters.resolution - cx
    gp_y = y * resolution_parameters.resolution - cy
    stations["latitude"], stations["longitude"] = proj.inverse(gp_x, gp_y).T
    sites.write_sites(stations, output_path / "stations.parquet")


def write_sw4_station_format(
    domain_parameters: DomainParameters, stations: pd.DataFrame, output_path: Path
) -> None:
    """Write station coordinates in SW4 format.

    Stations outside the domain are dropped rather than clamped to its
    edge: unlike EMOD3D's grid-point files there is no nearest gridpoint
    to snap to, and a station SW4 cannot place is not a recording.

    Parameters
    ----------
    domain_parameters : DomainParameters
        Domain definition, used to test which stations fall inside it.
    stations : pd.DataFrame
        The site table (see `workflow.sites`).
    output_path : Path
        Directory to write `stations.h5` and `stations.parquet` into.
    """
    lat_lon = stations[["latitude", "longitude"]].to_numpy()
    nzvm_coordinates = coordinates.wgs_depth_to_nztm(lat_lon)
    poly = domain_parameters.domain.polygon
    mask = shapely.contains_xy(poly, nzvm_coordinates[:, 0], nzvm_coordinates[:, 1])
    stations = stations.loc[mask]

    if len(stations) == 0:
        raise ValueError("No stations in domain.")

    with h5py.File(output_path / "stations.h5", "w") as f:
        for station_name, position in stations.iterrows():
            station_dset = f.create_group(station_name)
            location = station_dset.create_dataset(
                "STLA,STLO,STDP", (3,), dtype=np.float64
            )
            location[0] = position["latitude"]
            location[1] = position["longitude"]
            location[2] = 0.0

    sites.write_sites(stations, output_path / "stations.parquet")


@cli.from_docstring(app)
@log_utils.log_call()
def generate_fd_files(
    realisation_ffp: Annotated[Path, typer.Argument(readable=True, dir_okay=False)],
    sites_ffp: Annotated[Path, typer.Argument(readable=True, dir_okay=False)],
    output_path: Annotated[Path, typer.Argument(file_okay=False, writable=True)],
    format: Format = Format.EMOD3D,
) -> None:
    """Generate station coordinate files.

    Parameters
    ----------
    realisation_ffp : Path
        Path to realisation json file.
    sites_ffp : Path
        The site table (parquet, see `workflow.sites`) to pick stations from.
    output_path : Path
        Output path for station files.
    format : Format, optional
        The solver to write station files for. EMOD3D writes
        `stations.statcords`; SW4 writes `stations.h5`. Both write the
        in-domain site table to `stations.parquet`. Defaults to EMOD3D.
    """
    output_path.mkdir(exist_ok=True)
    domain_parameters = DomainParameters.read_from_realisation(realisation_ffp)

    # Only the stations in the domain are checked for complete site
    # parameters, when the filtered table is written.
    stations = pd.read_parquet(sites_ffp)

    match format:
        case Format.EMOD3D:
            resolution_parameters = Resolution.read_from_realisation(realisation_ffp)
            write_emod3d_station_format(
                domain_parameters, resolution_parameters, stations, output_path
            )
        case Format.SW4:
            write_sw4_station_format(domain_parameters, stations, output_path)

    realisations.append_log_entry(realisation_ffp)
