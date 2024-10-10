from pathlib import Path
from typing import Annotated

import numpy as np
import pandas as pd
import typer

from qcore import coordinates
from qcore.uncertainties import mag_scaling
from source_modelling import sources
from workflow.defaults import DefaultsVersion
from workflow.realisations import (
    RealisationMetadata,
    RupturePropagationConfig,
    SourceConfig,
)

MOMENT_TENSOR_SOLUTION_URL = "https://raw.githubusercontent.com/GeoNet/data/main/moment-tensor/GeoNet_CMT_solutions.csv"
NAN_PUBLIC_ID = "9999999"
app = typer.Typer()


@app.command(help="Generate a realisation from a GCMT event.")
def gcmt_to_realisation(
    gcmt_event_id: Annotated[
        str, typer.Argument(help="The GCMT event id to source the simulation for.")
    ],
    defaults_version: Annotated[
        DefaultsVersion,
        typer.Argument(
            help="Scientific defaults to use (determines simulation resolution among many other things)."
        ),
    ],
    realisation_ffp: Annotated[
        Path,
        typer.Argument(
            help="Path to output realisation.", writable=True, dir_okay=False
        ),
    ],
    alternate_solution: Annotated[
        bool,
        typer.Option(
            help="If indicated, choose the second possible strike, dip and rake combination instead of the first."
        ),
    ] = False,
):
    """Generate a realisation from a GCMT solution.

    Parameters
    ----------
    gcmt_event_id : str
        The GCMT event ID.
    defaults_version : DefaultsVersion
        The defaults version to use.
    realisation_ffp : Path
        The realisation filepath to output to.
    alternate_solution : bool
        If True, use the alternate solution for the GCMT realisation.
    """
    gcmt_solutions = pd.read_csv(MOMENT_TENSOR_SOLUTION_URL)

    gcmt_solutions = gcmt_solutions[
        gcmt_solutions["PublicID"] != NAN_PUBLIC_ID
    ].set_index("PublicID")

    solution = gcmt_solutions.loc[gcmt_event_id]

    magnitude = solution["Mw"]
    centroid = np.array([solution["Latitude"], solution["Longitude"]])
    rake = solution["rake2"] if alternate_solution else solution["rake1"]
    length = mag_scaling.mw_to_l_leonard(magnitude, rake)
    width = mag_scaling.mw_to_w_leonard(magnitude, rake)
    strike = coordinates.great_circle_bearing_to_nztm_bearing(
        centroid,
        length / 2,
        solution["strike2"] if alternate_solution else solution["strike1"],
    )
    dip = solution["dip2"] if alternate_solution else solution["dip1"]

    dtop = 0
    dbottom = width * np.sin(np.radians(dip))
    projected_width = width * np.cos(np.radians(dip))
    plane = sources.Plane.from_centroid_strike_dip(
        centroid, strike, None, dtop, dbottom, length, projected_width
    )
    plane.bounds += np.array([0, 0, solution["CD"] * 1000])

    source_config = SourceConfig(
        source_geometries={gcmt_event_id: sources.Fault([plane])}
    )
    expected_hypocentre = np.array([1 / 2, 1 / 2])

    rupture_config = RupturePropagationConfig(
        rupture_causality_tree={gcmt_event_id: None},
        jump_points={},
        rakes={gcmt_event_id: float(rake)},
        magnitudes={gcmt_event_id: float(magnitude)},
        hypocentre=expected_hypocentre,
    )
    metadata = RealisationMetadata(
        name=gcmt_event_id, version="1", defaults_version=defaults_version, tag="gcmt"
    )
    metadata.write_to_realisation(realisation_ffp)
    source_config.write_to_realisation(realisation_ffp)
    rupture_config.write_to_realisation(realisation_ffp)
