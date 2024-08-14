#!/usr/bin/env python3
"""
VM Parameters Generation.

This script generates the velocity model parameters used to generate the velocity model.

Usage
-----
To generate VM parameters for a Type-5 realisation:

```
$ python vm_params_generation.py path/to/realisation.yaml output/vm_params.yaml
```
"""

from pathlib import Path
from typing import Annotated, Optional

import geopandas as gpd
import numpy as np
import pandas as pd
import scipy as sp
import shapely
import typer
from empirical.util import openquake_wrapper_vectorized as openquake
from empirical.util import z_model_calculations
from empirical.util.classdef import GMM, TectType
from qcore import bounding_box, coordinates, gmt
from qcore.bounding_box import BoundingBox
from qcore.uncertainties import mag_scaling
from shapely import Polygon

from source_modelling import sources
from workflow import realisations
from workflow.realisations import (
    DomainParameters,
    RupturePropagationConfig,
    SourceConfig,
    VelocityModelParameters,
)


def get_nz_outline_polygon() -> Polygon:
    """
    Get the outline polygon of New Zealand.

    Returns
    -------
        Polygon: The outline polygon of New Zealand.
    """
    coastline_path = gmt.regional_resource("NZ", "coastline")
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


def pgv_estimate_from_magnitude(magnitude: np.ndarray) -> np.ndarray:
    """
    Return PGV for a given magnitude based on default scaling relationship.

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
    See the "Custom Models Used in VM Params" wiki page for an explanation of this function.
    """
    return np.interp(
        magnitude,
        [3.5, 4.1, 4.7, 5.2, 5.5, 5.8, 6.2, 6.5, 6.8, 7.0, 7.4, 7.7, 8.0],
        [0.015, 0.0375, 0.075, 0.15, 0.25, 0.4, 0.7, 1.0, 1.35, 1.65, 2.1, 2.5, 3.0],
    )


def rrup_buffer_polygon(
    faults: dict[str, sources.IsSource], rrups: dict[str, float]
) -> shapely.Polygon:
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
    [0]: Chiou BS-J, Youngs RR. Update of the
    Chiou and Youngs NGA Model for the Average Horizontal Component of
    Peak Ground Motion and Response Spectra. Earthquake
    Spectra. 2014;30(3):1117-1153.
    """
    pgv_target = pgv_estimate_from_magnitude(magnitude)

    def pgv_delta_from_rrup(rrup: float):
        vs30 = 500
        oq_dataframe = pd.DataFrame.from_dict(
            {
                "vs30": [vs30],
                "vs30measured": [False],
                "z1pt0": [z_model_calculations.chiou_young_08_calc_z1p0(vs30)],
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
    rakes: np.ndarray,
    ds_multiplier: float,
) -> float:
    """Estimate the simulation duration for a realisation simulated in a given domain.

    The simulation duration is calculated as the time for the s-waves of
    a rupture to propogate from the centre of the domain to the edge of
    the domain.

    Parameters
    ----------
    bounding_box : BoundingBox
        The simulation domain.
    type5_realisation : realisation.Realisation
        The realisation.
    ds_multiplier : float
        A multiplier for the wavelength of the s-wave.

    Returns
    -------
    float
        An estimated simulation duration time.
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
    s_wave_m_per_s = 3500
    s_wave_arrival_time = maximum_distance / s_wave_m_per_s

    # compute the pairwise distance between the domain corners and the fault corners
    pairwise_distance = np.linalg.norm(
        box_corners[np.newaxis, :, :] - fault_corners[:, np.newaxis, :], axis=2
    )
    largest_corner_distance = np.max(np.min(pairwise_distance, axis=1)) / 1000
    vs30 = 500
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


def total_magnitude(magnitudes: np.ndarray) -> float:
    return mag_scaling.mom2mag(np.sum(mag_scaling.mag2mom(magnitudes)))


def generate_velocity_model_parameters(
    realisation_filepath: Annotated[
        Path,
        typer.Argument(
            help="The path to the realisation to generate VM parameters for."
        ),
    ],
    resolution: Annotated[
        float, typer.Option(help="The resolution of the simulation in kilometres.")
    ] = 0.1,
    min_vs: Annotated[
        float,
        typer.Option(
            help="The minimum velocity (km/s) produced in the velocity model."
        ),
    ] = 0.5,
    ds_multiplier: Annotated[float, typer.Option(help="Ds multiplier")] = 1.2,
    dt: Annotated[
        Optional[float],
        typer.Option(
            help="The resolution of time (in seconds). If not specified, use resolution / 20."
        ),
    ] = None,
    vm_version: Annotated[str, typer.Option(help="Velocity model version.")] = "2.06",
    vm_topo_type: Annotated[
        str, typer.Option(help="VM topology type")
    ] = "SQUASHED_TAPERED",
):
    """Generate velocity model parameters for a realisation."""
    source_config = SourceConfig.read_from_realisation(realisation_filepath)

    rupture_propagation = RupturePropagationConfig.read_from_realisation(
        realisation_filepath
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
        np.fromiter(rupture_propagation.rakes.values()),
        ds_multiplier,
    )

    domain_parameters = DomainParameters(
        resolution=resolution,
        domain=model_domain,
        depth=max_depth,
        duration=sim_duration,
        dt=dt or resolution / 20,
    )
    velocity_model_parameters = VelocityModelParameters(
        min_vs=min_vs, version=vm_version, topo_type=vm_topo_type
    )

    domain_parameters.write_to_realisation(realisation_filepath)
    velocity_model_parameters.write_to_realisation(realisation_filepath)


def main():
    typer.run(generate_velocity_model_parameters)


if __name__ == "__main__":
    main()
