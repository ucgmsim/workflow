"""Generate station coordinates for EMOD3D."""

from pathlib import Path
from typing import Annotated

import numpy as np
import pandas as pd
import typer
from qcore import geo

from workflow import realisations
from workflow.realisations import DomainParameters


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
    ] = "/input/stations.ll",
) -> None:
    """Generate station gridpoint coordinates for a station list."""
    output_path.mkdir(exist_ok=True)
    domain_parameters: DomainParameters = realisations.read_config_from_realisation(
        DomainParameters, realisations_ffp
    )
    model_origin = domain_parameters.domain.origin
    hh = domain_parameters.resolution
    nx = domain_parameters.nx
    ny = domain_parameters.ny

    # where to save gridpoint and longlat station files
    gp_out = output_path / "stations.statcords"
    ll_out = output_path / "stations.ll"

    # retrieve in station names, latitudes and longitudes
    stations = pd.read_csv(
        stat_file, delimiter=" ", comment="#", names=["lon", "lat", "name"]
    )

    # convert ll to grid points
    xy = geo.ll2gp_multi(
        stations[["lon", "lat"]].to_numpy(),
        model_origin[1],
        model_origin[0],
        domain_parameters.domain.bearing,
        nx,
        ny,
        hh,
        keep_outside=True,
    )

    # store gridpoints and names if unique position
    sxy = set()
    suname = []
    for i in range(len(xy)):
        station_name = stations.iloc[i]["name"]
        if xy[i] is None or xy[i][0] == nx - 1 or xy[i][1] == ny - 1:
            print(f"Station outside domain: {station_name}")
        elif tuple(xy[i]) not in sxy:
            sxy.add(tuple(xy[i]))
            suname.append(station_name)
        elif keep_dup_station:
            # still adds in the station but raise a warning
            sxy.add(tuple(xy[i]))
            suname.append(station_name)
            print(
                f"Duplicate Station added: {station_name} at {xy[i]}",
            )
        else:
            print(f"Duplicate Station Ignored: {station_name}")

    # create grid point file
    with open(gp_out, "w", encoding="utf-8") as gpf:
        # file starts with number of entries
        gpf.write(f"{len(sxy)}\n")
        # x, y, z, name
        for xy, station_name in zip(sxy, suname):
            gpf.write(f"{xy[0]:5d} {xy[1]:5d} {1:5d} {station_name}\n")

    # convert unique grid points back to ll
    # warning: modifies sxy
    ll = geo.gp2ll_multi(
        [list(xy) for xy in sxy],
        model_origin[0],
        model_origin[1],
        domain_parameters.domain.bearing,
        nx,
        ny,
        hh,
    )

    # create ll file
    with open(ll_out, "w", encoding="utf-8") as llf:
        for pos, station_name in zip(ll, suname):
            llf.write(f"{pos[0]:11.5f} {pos[1]:11.5f} {station_name}\n")


def main():
    typer.run(generate_fd_files)


if __name__ == "__main__":
    main()
