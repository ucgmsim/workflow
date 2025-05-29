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

from pygmt_helper import plotting
from qcore import cli, coordinates
from qcore.uncertainties import mag_scaling
from source_modelling import sources
from velocity_modelling import bounding_box
from velocity_modelling.bounding_box import BoundingBox
from workflow import log_utils, realisations
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
    coastline_path = plotting.GMT_DATA.fetch("data/Paths/coastline/NZ.gmt")

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

    # compute the largest distance between the fault geometry and the domain boundary, that is
    #
    # max_{x \in F_g} (min_{y \in D} d(x, y))
    #
    # where F_g is the fault geometry, and D is the domain boundary.
    largest_distance = (
        shapely.hausdorff_distance(
            shapely.union_all([fault.geometry for fault in faults]),
            bounding_box.polygon,
        )
        / 1000
    )

    s_wave_arrival_time = (largest_distance * 1000) / s_wave_velocity

    # import here rather than at the module level because openquake is slow to import
    import oq_wrapper as oqw

    avg_rake = np.mean(rakes)
    oq_dataframe = pd.DataFrame.from_dict(
        {
            "vs30": [vs30],
            "z1pt0": [oqw.estimations.chiou_young_08_calc_z1p0(vs30)],
            "rrup": [largest_distance],
            "mag": [magnitude],
            "rake": [avg_rake],
        }
    )

    ds = np.exp(
        oqw.run_gmm(oqw.constants.GMM.AS_16, oqw.constants.TectType.ACTIVE_SHALLOW, oq_dataframe, "Ds595")[
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


def pgv_from_rrup(magnitude: float, rake: float, dip: float, rrup: float) -> float:
    """
    Compute the peak ground velocity (PGV) at a given distance from a rupture.

    Parameters
    ----------
    magnitude : float
            The magnitude of the rupture.
    rake : float
            The rake angle of the rupture.
    dip : float
            The dip angle of the rupture.
    rrup : float
            The distance from the rupture (in km).

    Returns
    -------
    float
            The peak ground velocity (cm/s) at the given distance from the rupture.
    """
    # import here rather than at the module level because openquake is slow to import
    import oq_wrapper as oqw

    vs30 = 500  # default Vs30 value
    return np.exp(
        oqw.run_gmm(
            oqw.constants.GMM.CY_14,
            oqw.constants.TectType.ACTIVE_SHALLOW,
            pd.DataFrame(
                {
                    "mag": [magnitude],
                    "rake": [rake],
                    "vs30": [vs30],
                    "vs30measured": [False],
                    "dip": [dip],
                    "z1pt0": [oqw.estimations.chiou_young_08_calc_z1p0(vs30)],
                    "ztor": [0],
                    "rrup": [rrup],
                    "rjb": [rrup],
                    "rx": [rrup],
                }
            ),
            "PGV",
        )["PGV_mean"].iloc[0]
    )


@log_utils.log_call()
def estimate_rrup(
    magnitude: float, rake: float, dip: float, pgv_target: float
) -> float:
    """
    Estimate the rupture radius such that stations at this radius will
    experience the target PGV.

    Parameters
    ----------
    magnitude : float
        The magnitude of the rupture.
    rake : float
        The rake angle of the rupture.
    dip : float
        The dip angle of the rupture.
    pgv_target : float
        The target PGV value (cm/s).

    Returns
    -------
    float
            The estimated rupture radius (in km).

    Examples
    --------
    >>> # Estimate the rupture radius for a 7.5 magnitude earthquake
    >>> # with a rake of 90 degrees, a dip of 45 degrees, and a target
    >>> # PGV of 10 cm/s.
    >>> estimate_rrup(7.5, 90, 45, 10)
    60.86630588572306
    """
    return sp.optimize.minimize_scalar(
        lambda rrup: np.abs(pgv_from_rrup(magnitude, rake, dip, rrup) - pgv_target),
        bounds=(0, 1000),
        method="bounded",
    ).x


def find_rrup_bounding_polygon(
    fault: sources.IsSource,
    magnitude: float,
    rake: float,
    pgv_target: float,
) -> Polygon:
    """Find the bounding polygon for the rrup distance of a fault.

    The bounding polygon is computed by estimating rrup from the PGV
    target, and then applying an rrup-width buffer to the fault
    geometries.

    Parameters
    ----------
    fault : sources.IsSource
        The fault geometry.
    magnitude : float
        The magnitude of the rupture.
    rake : float
        The rake angle of the rupture.
    pgv_target : float
        The target PGV value (cm/s).

    Returns
    -------
    Polygon
        The bounding polygon over the rrup distance of the fault in the realisation.
    """

    rrup = estimate_rrup(
        magnitude,
        rake,
        np.mean([plane.dip for plane in fault.planes]),
        pgv_target,
    )
    logger = log_utils.get_logger(__name__)
    logger.debug("computed rrup", rrups=rrup)

    return shapely.buffer(fault.geometry, rrup * 1000)


def dict_zip(*dicts: list[dict], strict: bool = True) -> dict:
    """
    Takes the product of one or more dictionaries.

    Parameters
    ----------
    *dicts : list of dict
        Variable number of dictionaries.
    strict : bool, default False
        If True, raise an error if the keys in `dicts` are not all the same.

    Returns
    -------
    dict
        A dictionary where each value is a tuple of the corresponding values from the input dictionaries.

    Raises
    ------
    ValueError
        If strict is True and the keys in the dictionaries are not all the same.
    """
    if not dicts:
        return {}

    keys = set(dicts[0].keys())
    for dict in dicts[1:]:
        keys = keys.intersection(dict.keys())

    if strict and len(keys) != len(dicts[0]):
        raise ValueError("Keys in dictionaries are not all the same.")
    result = {key: tuple(d[key] for d in dicts) for key in list(keys)}

    return result


def pgv_target(
    rupture_propagation_config: RupturePropagationConfig,
    velocity_model_parameters: VelocityModelParameters,
) -> float:
    """Compute the PGV target for the realisation.

    Parameters
    ----------
    rupture_propagation_config : RupturePropagationConfig
        The rupture propagation configuration containing magnitudes.
    velocity_model_parameters : VelocityModelParameters
        The velocity model parameters containing PGV interpolants.

    Returns
    -------
    float
        The PGV target for the realisation.
    """
    total_magnitude = mag_scaling.mom2mag(
        sum(
            mag_scaling.mag2mom(magnitude)
            for magnitude in rupture_propagation_config.magnitudes.values()
        )
    )
    return np.interp(
        total_magnitude,
        velocity_model_parameters.pgv_interpolants[:, 0],
        velocity_model_parameters.pgv_interpolants[:, 1],
    )


@cli.from_docstring(app)
@log_utils.log_call()
def generate_velocity_model_parameters(
    realisation_ffp: Annotated[Path, typer.Argument()],
) -> None:
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
    realisation_pgv_target = pgv_target(rupture_propagation, velocity_model_parameters)

    initial_fault = source_config.source_geometries[rupture_propagation.initial_fault]
    max_depth = get_max_depth(
        rupture_magnitude,
        initial_fault.planes[0].bottom_m / 1000,
    )

    # This polygon includes all the faults corners + a 2km buffer (which must be in the simulation domain).
    fault_buffer_polygons = [
        shapely.buffer(fault.geometry, 2000)
        for fault in source_config.source_geometries.values()
    ]
    # This polygon includes all areas within rrup distance of any
    # corner in the source geometries.
    # These may be in the domain where they are over land.
    rrup_bounding_polygons = [
        find_rrup_bounding_polygon(*args, pgv_target=realisation_pgv_target)
        for args in dict_zip(
            source_config.source_geometries,
            magnitudes,
            rupture_propagation.rakes,
        ).values()
    ]

    # The domain is the minimum area bounding box containing all of
    # the fault corners, and all points on land within rrup distance
    # of a fault corner.
    model_domain = bounding_box.minimum_area_bounding_box_for_polygons_masked(
        must_include=fault_buffer_polygons,
        may_include=rrup_bounding_polygons,
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
    realisations.append_log_entry(realisation_ffp)
