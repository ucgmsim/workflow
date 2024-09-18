#!/usr/bin/env python3
"""Domain Generation.

Description
-----------
Find a suitable simulation domain, estimating a rupture radius that captures significant ground motion, and the time the simulation should run for to capture this ground motion.

Inputs
------
A realisation file containing a metadata configuration, source definitions and rupture propagation information.

Outputs
-------
A realisation file containing velocity model and domain extent parameters.

Environment
-----------
Can be run in the cybershake container. Can also be run from your own computer using the `generate-velocity-model-parameters` command which is installed after running `pip install workflow@git+https://github.com/ucgmsim/workflow`.

Usage
-----
`generate-velocity-model-parameters [OPTIONS] REALISATION_FFP`

For More Help
-------------
See the output of `generate-velocity-model-parameters --help` or `workflow.scripts.generate_velocity_model_parameters`.
"""

from importlib import resources
from pathlib import Path
from typing import Annotated

import geopandas as gpd
import numpy as np
import numpy.typing as npt
import pandas as pd
import scipy as sp
import shapely
import typer
from shapely import Polygon
from velocity_modelling import bounding_box
from velocity_modelling.bounding_box import BoundingBox

from empirical.util import openquake_wrapper_vectorized as openquake
from empirical.util import z_model_calculations
from empirical.util.classdef import GMM, TectType
from qcore import coordinates, data
from qcore.uncertainties import mag_scaling
from source_modelling import sources
from workflow import log_utils
from workflow.realisations import (
    DomainParameters,
    RealisationMetadata,
    RupturePropagationConfig,
    SourceConfig,
    VelocityModelParameters,
)

app = typer.Typer()


def get_nz_outline_polygon() -> Polygon:
    """Get the outline polygon of New Zealand.

    Returns
    -------
    Polygon
        The outline polygon of New Zealand.
    """
    coastline_path = resources.files(data) / "Paths" / "coastline" / "NZ.gmt"

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


def pgv_estimate_from_magnitude(
    magnitude: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Return PGV for a given magnitude based on default scaling relationship.

    Parameters
    ----------
    magnitude : np.ndarray
        The magnitude(s) of the rupture(s).

    Returns
    -------
    np.ndarray
        An estimate of PGV for the rupture(s).

    References
    ----------
    See the 'Custom Models Used in VM Params' wiki page for an explanation of this function.
    """
    return np.interp(
        magnitude,
        [3.5, 4.1, 4.7, 5.2, 5.5, 5.8, 6.2, 6.5, 6.8, 7.0, 7.4, 7.7, 8.0],
        [0.015, 0.0375, 0.075, 0.15, 0.25, 0.4, 0.7, 1.0, 1.35, 1.65, 2.1, 2.5, 3.0],
    )


def rrup_buffer_polygon(
    faults: dict[str, sources.IsSource], rrups: dict[str, float]
) -> shapely.Polygon:
    """
    Create a buffer polygon around fault corners based on rupture radius.

    The function creates buffer zones around the corners of each fault using the provided rupture radius.
    These buffer zones are then combined into a single polygon that encompasses all buffered areas.

    Parameters
    ----------
    faults : dict
        A dictionary where keys are fault names and values are fault objects.
    rrups : dict
        A dictionary where keys are fault names and values are rupture radii (in kilometres).

    Returns
    -------
    shapely.Polygon
        A polygon representing the union of buffered areas around fault corners.

    Example
    -------
    >>> faults = {'fault1': source1, 'fault2': source2}
    >>> rrups = {'fault1': 10.0, 'fault2': 15.0}
    >>> buffer_polygon = rrup_buffer_polygon(faults, rrups)
    >>> print(buffer_polygon)
    """
    rrup_polygons = [
        shapely.buffer(shapely.Point(corner), rrups[fault_name] * 1000)
        for fault_name, fault in faults.items()
        for corner in fault.bounds[:, :2]
    ]

    return shapely.union_all(rrup_polygons)


def find_rrup(magnitude: float, avg_dip: float, avg_rake: float) -> float:
    """Find rrup at which pgv estimated from magnitude is close to target.

    Estimates rrup by calculating the rrup value that produces an
    estimated PGV value when measured from a site rrup distance
    away. The pgv -> rrup estimation comes from Chiou and Young (2014) [0].

    Parameters
    ----------
    magnitude : float
        The magnitude of the rupture.
    avg_dip : float
        The average dip of the faults involved.
    avg_rake : float
        The average rake of the faults involved.

    Returns
    -------
    float
        The inverse calculated rrup.

    References
    ----------
    [0]: Chiou BS-J, Youngs RR. Update of the Chiou and Youngs NGA Model for the Average Horizontal Component of Peak Ground Motion and Response Spectra. Earthquake Spectra. 2014;30(3):1117-1153.
    """
    pgv_target = pgv_estimate_from_magnitude(magnitude)

    def pgv_delta_from_rrup(rrup: float):
        vs30 = 500
        oq_dataframe = pd.DataFrame.from_dict(
            {
                "vs30": [vs30],
                "vs30measured": [False],
                "z1pt0": [z_model_calculations.chiou_young_08_calc_z1p0(vs30) * 1000],
                "dip": [avg_dip],
                "rake": [avg_rake],
                "mag": [magnitude],
                "ztor": [0],
                "rrup": [rrup],
                "rx": [rrup],
                "rjb": [rrup],
            }
        )
        # NOTE: I am assuming here that openquake returns PGV in
        # log-space. This is based on the fact that it does this for
        # PGA in this model (check the CY14_Italy_MEAN.csv test data
        # and compare the expected PGA with the PGA you calculate from
        # oq_run without exponentiation and you'll see this is true).
        pgv = np.exp(
            openquake.oq_run(
                GMM.CY_14,
                TectType.ACTIVE_SHALLOW,
                oq_dataframe,
                "PGV",
            )["PGV_mean"].iloc[0]
        )
        return np.abs(pgv - pgv_target)

    rrup_optimise_result = sp.optimize.minimize_scalar(
        pgv_delta_from_rrup, bounds=(0, 1e4)
    )
    rrup = rrup_optimise_result.x
    pgv_delta = rrup_optimise_result.fun
    if pgv_delta > 1e-4:
        raise ValueError("Failed to converge on rrup optimisation.")
    return rrup


def estimate_simulation_duration(
    bounding_box: BoundingBox,
    magnitude: float,
    faults: list[sources.IsSource],
    rakes: npt.NDArray[np.float64],
    ds_multiplier: float,
    vs30: float,
    s_wave_velocity: float,
) -> float:
    """Estimate the simulation duration required for a realisation in a given domain.

    The simulation distance is the length of time it
    takes the S-waves to reach and pass the edge of the domain from
    the centre of the fault(s).

    Parameters
    ----------
    bounding_box : BoundingBox
        The bounding box representing the simulation domain.
    magnitude : float
        The magnitude of the earthquake rupture.
    faults : list of sources.IsSource
        A list of fault objects defining the fault geometries.
    rakes : np.ndarray
        An array of rake angles for the faults.
    ds_multiplier : float
        Multiplier for the wavelength of the s-wave to adjust simulation duration.
    vs30 : float
        Average shear-wave velocity in the top 30 meters of soil (in m/s).
    s_wave_velocity : float
        Shear-wave velocity (in m/s) used to compute the travel time.

    Returns
    -------
    float
        The estimated simulation duration time (in seconds).
    """
    fault_corners = coordinates.wgs_depth_to_nztm(
        np.vstack([fault.corners for fault in faults])
    )
    fault_centroid = np.mean(fault_corners, axis=0)
    box_corners = np.append(
        bounding_box.corners,
        np.zeros((4, 1)),
        axis=1,
    )
    maximum_distance = np.max(np.linalg.norm(box_corners - fault_centroid, axis=1))
    s_wave_arrival_time = maximum_distance / s_wave_velocity

    # compute the pairwise distance between the domain corners and the fault corners
    pairwise_distance = np.linalg.norm(
        box_corners[np.newaxis, :, :] - fault_corners[:, np.newaxis, :], axis=2
    )
    largest_corner_distance = np.max(np.min(pairwise_distance, axis=1)) / 1000
    avg_rake = np.mean(rakes)
    oq_dataframe = pd.DataFrame.from_dict(
        {
            "vs30": [vs30],
            "z1pt0": [z_model_calculations.chiou_young_08_calc_z1p0(vs30)],
            "rrup": [largest_corner_distance],
            "mag": [magnitude],
            "rake": [avg_rake],
        }
    )

    ds = np.exp(
        openquake.oq_run(GMM.AS_16, TectType.ACTIVE_SHALLOW, oq_dataframe, "Ds595")[
            "Ds595_mean"
        ].iloc[0]
    )

    return s_wave_arrival_time + ds_multiplier * ds


def get_max_depth(magnitude: float, hypocentre_depth: float) -> int:
    """Estimate the maximum depth to simulate for a rupture.

    Parameters
    ----------
    magnitude : float
        The magnitude of the rupture.
    hypocentre_depth : float
        hypocentre depth (in km).

    Returns
    -------
    float
        The maximum simulation depth.

    References
    ----------
    See the "Custom Models Used in VM Params" wiki page for an explanation of this function.
    """
    return round(
        10
        + hypocentre_depth
        + (
            10
            * np.power(
                (0.5 * np.power(10, (0.55 * magnitude - 1.2)) / hypocentre_depth), 0.3
            )
        ),
        0,
    )


def total_magnitude(magnitudes: npt.NDArray[np.float64]) -> float:
    """
    Compute the total magnitude from an array of individual magnitudes.

    Parameters
    ----------
    magnitudes : np.ndarray
        An array of magnitudes.

    Returns
    -------
    float
        The total magnitude, computed from the summed moment of the input magnitudes.
    """
    return mag_scaling.mom2mag(np.sum(mag_scaling.mag2mom(magnitudes)))


@app.command(help="Generate velocity model parameters for a given realisation file")
@log_utils.log_call()
def generate_velocity_model_parameters(
    realisation_ffp: Annotated[
        Path,
        typer.Argument(
            help="The path to the realisation to generate VM parameters for."
        ),
    ],
):
    """Generate velocity model parameters for a given realisation file.

    This function reads the source and rupture propagation information and computes:

    1. The size of the simulation domain,
    2. The simulation duration.

    Both of these values are written to the realisation using `VelocityModelParameters`.

    Parameters
    ----------
    realisation_ffp : Path
        The path to the realisation file from which to read configurations and to which
        the generated velocity model parameters will be written.

    Returns
    -------
    None
        The function does not return any value. It writes the computed parameters to
        the specified realisation file.
    """
    metadata = RealisationMetadata.read_from_realisation(realisation_ffp)
    source_config = SourceConfig.read_from_realisation(realisation_ffp)
    velocity_model_parameters = (
        VelocityModelParameters.read_from_realisation_or_defaults(
            realisation_ffp, metadata.defaults_version
        )
    )

    rupture_propagation = RupturePropagationConfig.read_from_realisation(
        realisation_ffp
    )
    magnitudes = rupture_propagation.magnitudes

    rupture_magnitude = total_magnitude(np.array(list(magnitudes.values())))

    rakes = rupture_propagation.rakes

    rrups = {
        fault_name: find_rrup(
            magnitudes[fault_name],
            rupture_propagation.magnitudes[fault_name],
            rakes[fault_name],
        )
        for fault_name, fault in source_config.source_geometries.items()
    }
    log_utils.log("computed rrups", rrups=rrups)

    initial_fault = source_config.source_geometries[rupture_propagation.initial_fault]
    max_depth = get_max_depth(
        rupture_magnitude,
        initial_fault.fault_coordinates_to_wgs_depth_coordinates(
            rupture_propagation.hypocentre
        )[2]
        / 1000,
    )

    # Get bounding box

    # This polygon includes all the faults corners (which must be in the simulation domain).
    fault_bounding_box = bounding_box.minimum_area_bounding_box(
        np.vstack(
            [fault.bounds[:, :2] for fault in source_config.source_geometries.values()]
        )
    )

    # This polygon includes all areas within rrup distance of any
    # corner in the source geometries.
    # These may be in the domain where they are over land.
    rrup_bounding_polygon = rrup_buffer_polygon(source_config.source_geometries, rrups)

    # The domain is the minimum area bounding box containing all of
    # the fault corners, and all points on land within rrup distance
    # of a fault corner.
    model_domain = bounding_box.minimum_area_bounding_box_for_polygons_masked(
        must_include=[fault_bounding_box.polygon],
        may_include=[rrup_bounding_polygon],
        mask=get_nz_outline_polygon(),
    )

    sim_duration = estimate_simulation_duration(
        model_domain,
        rupture_magnitude,
        list(source_config.source_geometries.values()),
        np.fromiter(rupture_propagation.rakes.values(), float),
        velocity_model_parameters.ds_multiplier,
        velocity_model_parameters.vs30,
        velocity_model_parameters.s_wave_velocity,
    )

    domain_parameters = DomainParameters(
        resolution=velocity_model_parameters.resolution,
        domain=model_domain,
        depth=max_depth,
        duration=sim_duration,
        dt=velocity_model_parameters.dt,
    )
    domain_parameters.write_to_realisation(realisation_ffp)
