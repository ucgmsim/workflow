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
from enum import StrEnum, auto
from pathlib import Path
from typing import Annotated, Optional

import numpy as np
import pandas as pd
import requests
import typer

from qcore import cli
from qcore.uncertainties import distributions
from source_modelling import community_fault_model, magnitude_scaling, moment, sources
from source_modelling.community_fault_model import NodalPlane
from workflow import realisations
from workflow.defaults import DefaultsVersion
from workflow.realisations import (
    Magnitudes,
    Rakes,
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


class SourceType(StrEnum):
    """Source type for GCMT solutions."""

    FINITE_FAULT = "finite-fault"
    """Use a finite fault plane."""
    POINT_SOURCE = "point-source"
    """Use a point source approximation."""


# dyne-cm to newton metres conversion constant
NEWTON_METRES = 1e-7

    
@cli.from_docstring(app)
def gcmt_to_realisation(
    gcmt_event_id: Annotated[str, typer.Argument()],
    defaults_version: Annotated[DefaultsVersion, typer.Argument()],
    realisation_ffp: Annotated[Path, typer.Argument(writable=True, dir_okay=False)],
    source_type: Annotated[SourceType, typer.Argument()],
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
    source_type : SourceType
        The type of source to generate. FINITE_FAULT creates a finite fault plane,
        POINT_SOURCE creates a point source approximation.
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
    scaling_relation : magnitude_scaling.ScalingRelation, optional
        The magnitude scaling relation from source modelling to use.
        Defaults to Leonard2014.
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
        row = gcmt_solutions.loc[gcmt_event_id]
        latitude = float(row["Latitude"])  # type: ignore[invalid-argument-type]
        longitude = float(gcmt_solutions.at[gcmt_event_id, "Longitude"])  # type: ignore[invalid-argument-type]
        centroid_depth = float(gcmt_solutions.at[gcmt_event_id, "CD"])  # type: ignore[invalid-argument-type]
        solution_moment = float(gcmt_solutions.at[gcmt_event_id, "Mo"])  # type: ignore[invalid-argument-type]

        strike1 = float(gcmt_solutions.at[gcmt_event_id, "strike1"])  # type: ignore[invalid-argument-type]
        dip1 = float(gcmt_solutions.at[gcmt_event_id, "dip1"])  # type: ignore[invalid-argument-type]
        rake1 = float(gcmt_solutions.at[gcmt_event_id, "rake1"])  # type: ignore[invalid-argument-type]

        strike2 = float(gcmt_solutions.at[gcmt_event_id, "strike2"])  # type: ignore[invalid-argument-type]
        dip2 = float(gcmt_solutions.at[gcmt_event_id, "dip2"])  # type: ignore[invalid-argument-type]
        rake2 = float(gcmt_solutions.at[gcmt_event_id, "rake2"])  # type: ignore[invalid-argument-type]

        nodal_plane_1 = NodalPlane(strike1, dip1, rake1)
        nodal_plane_2 = NodalPlane(strike2, dip2, rake2)
    elif gcmt_event_id in automated_gcmt_solutions:
        solution = automated_gcmt_solutions[gcmt_event_id]
        latitude = solution["location"]["latitude"]
        longitude = solution["location"]["longitude"]
        centroid_depth = solution["location"]["depth"]
        solution_moment = float(solution["moment"])
        nodal_plane_1 = NodalPlane(**solution["nodalPlanes"][0])
        nodal_plane_2 = NodalPlane(**solution["nodalPlanes"][1])

    else:
        raise typer.BadParameter(
            f"GCMT event ID {gcmt_event_id} not found in either the published GCMT solutions or automated solutions.",
            param_hint="GCMT_EVENT_ID",
        )

    magnitude = moment.moment_to_magnitude(solution_moment * NEWTON_METRES)

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

    # Calculate dip direction from strike (strike + 90 degrees for right-hand rule)
    dip_direction = (selected_nodal_plane.strike + 90) % 360

    length, width = magnitude_scaling.magnitude_to_length_width(
        scaling_relation, magnitude, rake
    )

    centroid = np.array([latitude, longitude, centroid_depth])

    # Create source based on source_type parameter
    if source_type == SourceType.POINT_SOURCE:
        length_km, width_km = magnitude_scaling.magnitude_to_length_width(
            scaling_relation, magnitude, rake
        )
        length_m = length_km * 1000  # Convert km to meters
        width_m = width_km * 1000  # Convert km to meters

        source_geometry = sources.Point.from_lat_lon_depth(
            point_coordinates=np.array(
                [latitude, longitude, centroid_depth * 1000]
            ),  # Convert km to meters
            length_m=length_m,  # Use calculated length from area
            width_m=width_m,  # Use calculated width from area
            strike=selected_nodal_plane.strike,
            dip=selected_nodal_plane.dip,
            dip_dir=dip_direction,
        )

    else:
        # Create plane source (default behavior)
        plane = sources.Plane.from_centroid_strike_dip(
            centroid,
            selected_nodal_plane.dip,
            length,
            width,
            strike=selected_nodal_plane.strike,
        )

        if plane.bounds[:, 2].min() < 0:
            warnings.warn(
                f"Scaling relationship produced a plane with negative depth ({plane.bounds[:, 2].min() / 1000:.2f}km)."
                " Shifting the plane down to correct."
            )
            plane.bounds[:, 2] -= plane.bounds[:, 2].min()

        source_geometry = sources.Fault([plane])

    if lat_hypo is not None and lon_hypo is not None:
        if source_type == SourceType.POINT_SOURCE:
            # For point sources, hypocentre is always at the center (0.5, 0.5)
            hypocentre = np.array([0.5, 0.5])
        else:
            # For plane sources, convert lat/lon to fault coordinates
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

    source_config = SourceConfig(source_geometries={gcmt_event_id: source_geometry})
    magnitudes = Magnitudes(magnitudes={gcmt_event_id: float(magnitude)})
    rakes = Rakes(rakes={gcmt_event_id: float(rake)})
    rupture_config = RupturePropagationConfig(
        rupture_causality_tree={gcmt_event_id: None},
        jump_points={},
        hypocentre=hypocentre,
    )
    metadata = RealisationMetadata(
        name=gcmt_event_id, version="1", defaults_version=defaults_version, tag="gcmt"
    )

    realisation_ffp.parent.mkdir(parents=True, exist_ok=True)

    for config in [metadata, source_config, rupture_config, magnitudes, rakes]:
        config.write_to_realisation(realisation_ffp)

    realisations.append_log_entry(realisation_ffp)
