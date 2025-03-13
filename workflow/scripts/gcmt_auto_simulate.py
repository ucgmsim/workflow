"""Automatically Simulate GCMT Solutions.

Description
-----------
This script fetches the latest GCMT solutions and identifies new, large earthquakes within 30 km of New Zealand. It then sets up and runs simulations for these earthquakes using the Cylc workflow engine.

Inputs
------
1. GCMT Solutions URL,
2. Path to old GCMT Solutions.

Outputs
-------
1. Updated GCMT Solutions file,
2. Cylc workflow directory with input data for new simulations.

Environment
-----------
This script is designed to be run in conjunction with a cron job on Hypocentre. It is not intended for researcher use.

Usage
-----
`python gcmt_auto_simulate.py GCMT_SOLUTIONS_URL OLD_GCMT_SOLUTIONS_PATH`

For More Help
-------------
See the output of `python gcmt_auto_simulate.py --help` for more details on the command-line arguments.
"""

import datetime
import json
import subprocess
from pathlib import Path
from typing import Annotated

import geopandas as gpd
import numpy as np
import requests
import shapely
import typer
from shapely import Polygon

from qcore import cli, coordinates, gmt

app = typer.Typer()


def get_nz_outline_polygon() -> Polygon:
    """Get the outline polygon of New Zealand.

    Returns
    -------
    Polygon
        The outline polygon of New Zealand.
    """
    coastline_path = gmt.GMT_DATA.fetch("data/Paths/coastline/NZ.gmt")

    gpd_df = gpd.read_file(coastline_path)
    island_polygons = [
        Polygon(
            coordinates.wgs_depth_to_nztm(
                np.array(shapely.geometry.mapping(island)["coordinates"])[:, ::-1]
            )
        )
        for island in gpd_df.geometry
    ]
    south_island, north_island = sorted(
        island_polygons, key=lambda island: island.area, reverse=True
    )[:2]
    south_island = south_island.simplify(100)
    north_island = north_island.simplify(100)
    return shapely.union(south_island, north_island)


@cli.from_docstring(app)
def gcmt_auto_simulate(
    gcmt_solutions_url: Annotated[str, typer.Argument()],
    old_gcmt_solutions_path: Annotated[Path, typer.Argument()],
):
    """Automatically simulate GCMT solutions that are new, large, and within 30 km of New Zealand.

    Parameters
    ----------
    gcmt_solutions_url : str
        GCMT Solutions URL.
    old_gcmt_solutions_path : Path
        Path to old GCMT Solutions.

    Raises
    ------
    typer.Exit
        If there are no new solutions to simulate.
    """
    updated_gcmt_solutions = requests.get(gcmt_solutions_url).json()
    if old_gcmt_solutions_path.exists():
        with open(old_gcmt_solutions_path) as old_gcmt_solutions_handle:
            old_gcmt_solutions = json.load(old_gcmt_solutions_handle)
    else:
        old_gcmt_solutions = dict()
    nz_polygon = get_nz_outline_polygon()
    solutions_to_simulate = [
        gcmt_id
        for gcmt_id, solution in updated_gcmt_solutions.items()
        if gcmt_id not in old_gcmt_solutions
        and solution["magnitude"] >= 4
        and solution["location"]["depth"] <= 60
        and shapely.distance(
            nz_polygon,
            shapely.Point(
                coordinates.wgs_depth_to_nztm(
                    np.array(
                        [
                            solution["location"]["latitude"],
                            solution["location"]["longitude"],
                        ]
                    )
                )
            ),
        )
        < 30 * 1000
    ]
    if not solutions_to_simulate:
        raise typer.Exit(code=0)
    now = datetime.datetime.now()
    workflow_id = f'gcmt_{now.strftime("%Y%m%d_%H%M%S")}'
    cylc_directory = Path.home() / "cylc-src" / workflow_id
    cylc_directory.mkdir(exist_ok=True, parents=True)

    for solution in solutions_to_simulate:
        (cylc_directory / "input" / solution).mkdir(exist_ok=True, parents=True)

    subprocess.check_call(
        ["plan-workflow"]
        + solutions_to_simulate
        + [
            str(cylc_directory / "flow.cylc"),
            "--goal",
            "im_calc",
            "--source",
            "gcmt",
            "--defaults-version",
            "24.2.2.2",
            "--target-host",
            "hypocentre",
        ]
    )

    subprocess.check_call(["cylc", "vip", workflow_id])
    with open(old_gcmt_solutions_path, "w") as old_gcmt_solutions_handle:
        json.dump(updated_gcmt_solutions, old_gcmt_solutions_handle)
