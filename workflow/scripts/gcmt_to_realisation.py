from pathlib import Path
from typing import Annotated

import numpy as np
import pandas as pd
import typer

from source_modelling import ccldpy, sources
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
    """
    gcmt_solutions = pd.read_csv(MOMENT_TENSOR_SOLUTION_URL)

    gcmt_solutions = gcmt_solutions[
        gcmt_solutions["PublicID"] != NAN_PUBLIC_ID
    ].set_index("PublicID")

    solution = gcmt_solutions.loc[gcmt_event_id]
    # Get a likely rupture using CCLDpy
    _, ccld_selected_rupture = ccldpy.simulate_rupture_surface(
        1,
        "crustal",
        "other",
        solution["Latitude"],
        solution["Longitude"],
        solution["CD"],  # Centroid depth
        solution["Mw"],
        "C",
        [334, 333, 333, 111, 111, 111, 0],
        strike=solution["strike1"],
        dip=solution["dip1"],
        rake=solution["rake1"],
        strike2=solution["strike2"],
        dip2=solution["dip2"],
        rake2=solution["rake2"],
    )
    ccld_selected_rupture = ccld_selected_rupture.iloc[0]
    corners = np.array(
        [
            [
                ccld_selected_rupture["ULC Latitude"],
                ccld_selected_rupture["ULC Longitude"],
                ccld_selected_rupture["ULC Depth (km)"] * 1000,
            ],
            [
                ccld_selected_rupture["URC Latitude"],
                ccld_selected_rupture["URC Longitude"],
                ccld_selected_rupture["URC Depth (km)"] * 1000,
            ],
            [
                ccld_selected_rupture["LRC Latitude"],
                ccld_selected_rupture["LRC Longitude"],
                ccld_selected_rupture["LRC Depth (km)"] * 1000,
            ],
            [
                ccld_selected_rupture["LLC Latitude"],
                ccld_selected_rupture["LLC Longitude"],
                ccld_selected_rupture["LLC Depth (km)"] * 1000,
            ],
        ]
    )
    plane = sources.Plane.from_corners(corners)
    rake = ccld_selected_rupture["Rake"]
    magnitude = ccld_selected_rupture["Magnitude"]
    hypocentre = plane.wgs_depth_coordinates_to_fault_coordinates(
        np.array(
            [
                ccld_selected_rupture["Hypocenter Latitude"],
                ccld_selected_rupture["Hypocenter Longitude"],
            ]
        )
    )

    source_config = SourceConfig(
        source_geometries={gcmt_event_id: sources.Fault([plane])}
    )

    rupture_config = RupturePropagationConfig(
        rupture_causality_tree={gcmt_event_id: None},
        jump_points={},
        rakes={gcmt_event_id: float(rake)},
        magnitudes={gcmt_event_id: float(magnitude)},
        hypocentre=hypocentre,
    )
    metadata = RealisationMetadata(
        name=gcmt_event_id, version="1", defaults_version=defaults_version, tag="gcmt"
    )
    metadata.write_to_realisation(realisation_ffp)
    source_config.write_to_realisation(realisation_ffp)
    rupture_config.write_to_realisation(realisation_ffp)
