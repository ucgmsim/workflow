"""
Generate a realisation from a GCMT solution.

Description
-----------
This script generates a realisation from a GCMT solution by fetching the necessary data from the GeoNet and automated GCMT solutions, selecting the most likely nodal plane, and generating a rupture geometry using CCLDpy.

Inputs
------
1. gcmt_event_id: The GCMT event ID to source the simulation for.
2. defaults_version: Scientific defaults to use (determines simulation resolution among many other things).
3. realisation_ffp: Path to output realisation.

Outputs
-------
A realisation file containing source configurations, rupture propagation configurations, and metadata.

Environment
-----------
Can be run in the cybershake container. Can also be run from your own computer using the `gcmt-to-realisation` command which is installed after running `pip install workflow@git+https://github.com/ucgmsim/workflow`.

Usage
-----
`gcmt-to-realisation [OPTIONS] GCMT_EVENT_ID DEFAULTS_VERSION REALISATION_FFP`

For More Help
-------------
See the output of `gcmt-to-realisation --help`.
"""

import warnings
from collections.abc import Callable
from enum import StrEnum, auto
from pathlib import Path
from typing import Annotated, Optional

import numpy as np
import pandas as pd
import requests
import typer

from qcore import cli
from qcore.uncertainties import distributions
from source_modelling import community_fault_model, magnitude_scaling, sources
from source_modelling.community_fault_model import NodalPlane
from workflow import realisations
from workflow.defaults import DefaultsVersion
from workflow.realisations import (
    RealisationMetadata,
    RupturePropagationConfig,
    SourceConfig,
)

MOMENT_TENSOR_SOLUTION_URL = "https://raw.githubusercontent.com/GeoNet/data/main/moment-tensor/GeoNet_CMT_solutions.csv"
AUTOMATED_TENSOR_URL = "https://gcmt-realtime-database-default-rtdb.asia-southeast1.firebasedatabase.app/c059adc9c34b9b1c77a3bfc04e4059ea/earthquakes.json"
NAN_PUBLIC_ID = "9999999"
app = typer.Typer()


class NodalPlaneChoice(StrEnum):
    """Nodal plane choice for GCMT solutions."""

    PLANE_1 = auto()
    """First nodal plane."""
    PLANE_2 = auto()
    """Second nodal plane."""
    MOST_LIKELY = auto()
    """The most likely nodal plane estimated from the community fault model."""


class SamplingStrategy(StrEnum):
    """Sampling strategy for hypocentre location."""

    AVERAGE = auto()
    """Average of the distribution."""
    RANDOM = auto()
    """Random sample from the distribution."""
    CENTROID = auto()
    """Use the solution centroid."""


@cli.from_docstring(app)
def gcmt_to_realisation(
    gcmt_event_id: Annotated[str, typer.Argument()],
    defaults_version: Annotated[DefaultsVersion, typer.Argument()],
    realisation_ffp: Annotated[Path, typer.Argument(writable=True, dir_okay=False)],
    hypocentre_strategy: Annotated[
        SamplingStrategy, typer.Option()
    ] = SamplingStrategy.CENTROID,
    shypo: Annotated[Optional[float], typer.Option(min=0, max=1)] = None,
    dhypo: Annotated[Optional[float], typer.Option(min=0, max=1)] = None,
    lat_hypo: Annotated[
        Optional[float],
        typer.Option(min=-90, max=90),
    ] = None,
    lon_hypo: Annotated[
        Optional[float],
        typer.Option(min=-180, max=180),
    ] = None,
    scaling_relation: Annotated[
        magnitude_scaling.ScalingRelation, typer.Option(case_sensitive=False)
    ] = magnitude_scaling.ScalingRelation.LEONARD2014,
    nodal_plane: Annotated[
        NodalPlaneChoice, typer.Option()
    ] = NodalPlaneChoice.MOST_LIKELY,
) -> None:
    """Generate a realisation from a GCMT solution.

    Parameters
    ----------
    gcmt_event_id : str
        The GCMT event ID to source the simulation for.
    defaults_version : DefaultsVersion
        Scientific defaults to use (determines simulation resolution among many other things).
    realisation_ffp : Path
        Path to output realisation.
    hypocentre_strategy : SamplingStrategy
        Sampling strategy for the hypocentre strike coordinate.
    shypo : float, optional
        The initial hypocentre strike coordinate (0 - 1). Distribution is truncated normal with mean 0.5 and standard deviation 0.25.
    dhypo : float, optional
        The initial hypocentre strike coordinate (0 - 1). Distribution is truncated Weibull.
    lat_hypo : float, optional
        The latitude coordinate of the hypocentre. Conflicts with shypo and dhypo.
    lon_hypo : float, optional
        The latitude coordinate of the hypocentre. Conflicts with shypo and dhypo.
    scaling_relation : magnitude_scaling.ScalingRelation or callable, optional
        Either the name of the magnitude scaling relation from source
        modelling to use, or a callable function that takes a
        magnitude and returns a tuple `(length, width)`. Used for custom
        scaling relations.
    nodal_plane : NodalPlaneChoice
        The nodal plane to use. Most likely will use the community fault model to
        choose a nodal plane that agrees with the tectonic fabric.
        Defaults to `MOST_LIKELY`.
    """
    if (shypo is not None or dhypo is not None) and (
        lat_hypo is not None or lon_hypo is not None
    ):
        raise typer.BadParameter(
            "The options shypo and dhypo are mutually exclusive with lat_hypo and lon_hypo."
        )

    gcmt_solutions = pd.read_csv(MOMENT_TENSOR_SOLUTION_URL)
    automated_gcmt_solutions = requests.get(AUTOMATED_TENSOR_URL).json()

    gcmt_solutions = gcmt_solutions[
        gcmt_solutions["PublicID"] != NAN_PUBLIC_ID
    ].set_index("PublicID")

    if gcmt_event_id in gcmt_solutions.index:
        solution = gcmt_solutions.loc[gcmt_event_id]
        latitude = solution["Latitude"]
        longitude = solution["Longitude"]
        centroid_depth = solution["CD"]
        magnitude = solution["Mw"]
        nodal_plane_1 = NodalPlane(
            solution["strike1"], solution["dip1"], solution["rake1"]
        )
        nodal_plane_2 = NodalPlane(
            solution["strike2"], solution["dip2"], solution["rake2"]
        )
    elif gcmt_event_id in automated_gcmt_solutions:
        solution = automated_gcmt_solutions[gcmt_event_id]
        latitude = solution["location"]["latitude"]
        longitude = solution["location"]["longitude"]
        centroid_depth = solution["location"]["depth"]
        magnitude = solution["magnitude"]
        nodal_plane_1 = NodalPlane(**solution["nodalPlanes"][0])
        nodal_plane_2 = NodalPlane(**solution["nodalPlanes"][1])
    else:
        raise typer.BadParameter(
            f"GCMT event ID {gcmt_event_id} not found in either the published GCMT solutions or automated solutions.",
            param_hint="GCMT_EVENT_ID",
        )

    model = community_fault_model.get_community_fault_model()

    match nodal_plane:
        case NodalPlaneChoice.PLANE_1:
            selected_nodal_plane = nodal_plane_1
        case NodalPlaneChoice.PLANE_2:
            selected_nodal_plane = nodal_plane_2
        case NodalPlaneChoice.MOST_LIKELY:
            selected_nodal_plane = community_fault_model.most_likely_nodal_plane(
                model, np.array([latitude, longitude]), nodal_plane_1, nodal_plane_2
            )

    rake = selected_nodal_plane.rake

    if isinstance(scaling_relation, str | magnitude_scaling.ScalingRelation):
        length, width = magnitude_scaling.magnitude_to_length_width(
            scaling_relation, magnitude, rake
        )
    elif isinstance(scaling_relation, Callable):
        length, width = scaling_relation(magnitude)

    centroid = np.array([latitude, longitude, centroid_depth])
    plane = sources.Plane.from_centroid_strike_dip(
        centroid,
        selected_nodal_plane.dip,
        length,
        width,
        strike=selected_nodal_plane.strike,
    )

    if plane.bounds[:, 2].min() < 0:
        warnings.warn(
            f"Scaling relationship produced a plane with negative depth ({plane.bounds[:, 2].min()/1000:.2f}km)."
            " Shifting the plane down to correct."
        )
        plane.bounds[:, 2] -= plane.bounds[:, 2].min()

    if lat_hypo is not None and lon_hypo is not None:
        hypocentre = plane.wgs_depth_coordinates_to_fault_coordinates(
            np.array([lat_hypo, lon_hypo])
        )
    elif shypo is not None and dhypo is not None:
        hypocentre = np.array([shypo, dhypo])
    elif hypocentre_strategy == SamplingStrategy.AVERAGE:
        hypocentre = np.array(
            [
                1 / 2,
                distributions.truncated_weibull_expected_value(1),
            ]
        )
    elif hypocentre_strategy == SamplingStrategy.RANDOM:
        hypocentre = np.array(
            [
                distributions.truncated_normal(1 / 2, 1 / 4),
                distributions.truncated_weibull(1),
            ]
        )
    else:
        hypocentre = np.array([1 / 2, 1 / 2])

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
    realisation_ffp.parent.mkdir(parents=True, exist_ok=True)
    metadata.write_to_realisation(realisation_ffp)
    source_config.write_to_realisation(realisation_ffp)
    rupture_config.write_to_realisation(realisation_ffp)
    realisations.append_log_entry(realisation_ffp)
