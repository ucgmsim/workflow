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
Can be run in the cybershake container. Can also be run from your own computer using the `generate-domain` command which is installed after running `pip install workflow@git+https://github.com/ucgmsim/workflow`.

Usage
-----
`generate-domain [OPTIONS] REALISATION_FFP`

For More Help
-------------
See the output of `generate-domain --help` or `workflow.scripts.generate_domain`.
"""

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

import numpy as np
import numpy.typing as npt
import pandas as pd
import shapely
import typer

from qcore import cli, geo
from source_modelling import magnitude_scaling, moment, sources
from velocity_modelling import bounding_box
from velocity_modelling.bounding_box import BoundingBox
from workflow import log_utils, realisations, utils
from workflow.realisations import (
    DomainParameters,
    Magnitudes,
    Rakes,
    RealisationMetadata,
    SourceConfig,
    VelocityModelParameters,
)

app = typer.Typer()


def get_significant_duration(
    magnitude: float, distance: float, vs30: float, rake: float, z1pt0: float
) -> float:
    """Estimate significant duration using Afshari-Stewart (2016).

    Parameters
    ----------
    magnitude : float
        The magnitude of the rupture.
    distance : float
        The rupture distance (rrup) in kilometers to estimate Ds595 for.
    vs30 : float
        The Vs30 value at the site.
    rake : float
        The rake parameter.
    z1pt0 : float
        The Z1.0 estimated at the site.

    Returns
    -------
    float
        The estimated Ds595 of the rupture (in seconds).
    """

    import oq_wrapper as oqw

    # Create the input context for the GMM
    ctx = pd.DataFrame(
        {
            "vs30": [vs30],
            "z1pt0": [z1pt0],
            "rrup": [distance],
            "mag": [magnitude],
            "rake": [rake],
        }
    )

    # Execute and transform from log space
    results = oqw.run_gmm(
        oqw.constants.GMM.AS_16,
        oqw.constants.TectType.ACTIVE_SHALLOW,
        ctx,
        "Ds595",
    )

    return np.exp(results["Ds595_mean"].iloc[0])


def boundary_distance(
    domain: BoundingBox, sources: Iterable[sources.IsSource]
) -> float:
    r"""Compute the distance from source to boundary.

    The largest distance between the fault geometry and the domain
    boundary is the Hausdorff distance, that is max_{x \in F_g}
    (min_{y \in D} d(x, y)) where F_g is the fault geometry, and D is
    the domain boundary.

    Parameters
    ----------
    domain : BoundingBox
        The domain bounding box.
    sources : Iterable[sources.IsSource]
        The sources to compute the boundary distance for.

    Returns
    -------
    float
        The Hausdorff distance between the sources and the boundary (in meters).
    """
    source_geometry = shapely.union_all([source.geometry for source in sources])
    bounding_box_geometry = domain.polygon
    return float(shapely.hausdorff_distance(source_geometry, bounding_box_geometry))


def total_magnitude(magnitudes: Iterable[magnitude_scaling.BoldM]) -> float:
    """
    Compute the total magnitude from an array of individual magnitudes.

    Parameters
    ----------
    magnitudes : Iterable[magnitude_scaling.BoldM]
        An array of magnitudes.

    Returns
    -------
    float
        The total magnitude, computed from the summed moment of the input magnitudes.
    """
    total_moment = sum(
        moment.magnitude_to_moment(magnitude, bold_m=True) for magnitude in magnitudes
    )
    return moment.moment_to_magnitude(total_moment)


@dataclass
class RuptureContext:
    """Rupture context container class."""

    magnitude: float
    """Rupture magnitude."""
    rake: float
    """Rupture rake."""
    vs30: float
    """Rupture vs30."""
    z1pt0: float
    """Rupture Z1.0."""
    s_wave_velocity: float
    """Rupture s-wave velocity."""
    ds_multiplier: float
    """Rupture Ds multiplier."""


def rupture_context_from(
    magnitudes: Magnitudes,
    rakes: Rakes,
    velocity_model_parameters: VelocityModelParameters,
) -> RuptureContext:
    """Create a rupture context from a rupture.

    Parameters
    ----------
    magnitudes : Magnitudes
        The magnitudes of the rupture faults.
    rakes : Rakes
        The rakes in the rupture faults.
    velocity_model_parameters : VelocityModelParameters
        The velocity model parameters. Only use s_wave_velocity and
        ds_multiplier.

    Returns
    -------
    RuptureContext
        The rupture context computed from the source information and
        velocity model parameters.
    """
    import oq_wrapper as oqw

    magnitude = total_magnitude(magnitudes.magnitudes.values())
    rake = average_rake(rakes, magnitudes)

    z1pt0 = float(
        oqw.estimations.chiou_young_08_calc_z1p0(velocity_model_parameters.vs30)
    )

    s_wave_velocity = velocity_model_parameters.s_wave_velocity
    ds_multiplier = velocity_model_parameters.ds_multiplier

    return RuptureContext(
        magnitude=magnitude,
        rake=rake,
        vs30=velocity_model_parameters.vs30,
        z1pt0=z1pt0,
        s_wave_velocity=s_wave_velocity,
        ds_multiplier=ds_multiplier,
    )


def average_rake(rakes: Rakes, magnitudes: Magnitudes) -> float:
    """Find moment-weighted average rupture rake.

    Parameters
    ----------
    rakes : Rakes
        The rakes to average.
    magnitudes : Magnitudes
        The magnitudes to weight rakes by.

    Returns
    -------
    float
        The moment-weighted average rake.
    """
    moments = {
        k: moment.magnitude_to_moment(magnitude, bold_m=True)
        for k, magnitude in magnitudes.magnitudes.items()
    }
    max_moment = max(moments.values())
    # re-normalise for debugging purposes and to maybe improve
    # floating-point accuracy in the avg wbearing function.
    moments = {k: moment / max_moment for k, moment in moments.items()}

    weighted_rakes = list(utils.dict_zip(rakes.rakes, moments).values())
    return geo.avg_wbearing(weighted_rakes)  # type: ignore[invalid-argument-type]


def estimate_simulation_duration(
    rupture_context: RuptureContext,
    bounding_box: BoundingBox,
    faults: Iterable[sources.IsSource],
) -> float:
    """Estimate the simulation duration required for a realisation in a given domain.

    The simulation duration is the length of time it
    takes the S-waves to reach and pass the edge of the domain from
    the centre of the fault(s).

    Parameters
    ----------
    rupture_context : RuptureContext
        The rupture context.
    bounding_box : BoundingBox
        The domain to estimate boundary distances for.
    faults : Iterable[sources.IsSource]
        The faults to estimate boundary distances for.

    Returns
    -------
    float
        The estimated simulation duration time (in seconds).
    """
    largest_distance = boundary_distance(bounding_box, faults)

    significant_duration = get_significant_duration(
        distance=largest_distance / 1000.0,
        magnitude=rupture_context.magnitude,
        vs30=rupture_context.vs30,
        rake=rupture_context.rake,
        z1pt0=rupture_context.z1pt0,
    )

    s_wave_arrival_time = largest_distance / rupture_context.s_wave_velocity
    total_duration = s_wave_arrival_time + (
        significant_duration * rupture_context.ds_multiplier
    )

    return total_duration


def simulation_max_depth(magnitude: float, bottom_depth: float) -> float:
    """Estimate the maximum depth to simulate for a rupture.

    This function estimates the max depth of the domain by ensuring
    the z-extent is deep enough so the bottom does not interfere with
    waves initially radiating downwards from the earthquake before
    returning to the surface. Its design is,

    1. The bottom depth of the rupture plus
    2. a flat 10km buffer and,
    3. a magnitude dependent power term that accounts for the fact
    that waves refract deeper from larger magnitude earthquakes, but
    4. accounting for the fact that deeper earthquakes for the same
    magnitude result in weaker shaking at the surface.

    Parameters
    ----------
    magnitude : float
        The magnitude of the rupture.
    bottom_depth : float
        Bottom depth (in km).

    Returns
    -------
    float
        The maximum simulation depth.

    References
    ----------
    Robin's slack message that describes the max depth estimation:
    https://uceqeng.slack.com/archives/C06L1MRUQF8/p1718865771998809?thread_ts=1718846081.570789&cid=C06L1MRUQF8.
    """
    return round(
        10
        + bottom_depth
        + (
            10
            * np.power(
                (0.5 * np.power(10, (0.55 * magnitude - 1.2)) / bottom_depth), 0.3
            )
        ),
        ndigits=0,  # like default rounding behaviour but returns a float.
    )


def estimate_r_surface(
    rrup_interpolants: npt.NDArray[np.floating], magnitude: float, ztor: float
) -> float:
    """Estimate the horizontal surface distance from a rupture.

    Parameters
    ----------
    rrup_interpolants : array of floats
        A 2D array where the first row contains magnitude values and the second
        row contains the corresponding rrup values (in km).
    magnitude : float
        The magnitude of the rupture.
    ztor : float
        The depth to the top of the rupture plane (in km).

    Returns
    -------
    float
        The estimated horizontal surface distance (in metres).
    """
    rrup = float(np.interp(magnitude, rrup_interpolants[0], rrup_interpolants[1]))

    if rrup < ztor:
        raise ValueError(
            f"Top-of-rupture depth {ztor=} is greater than estimated rrup {rrup=}."
        )

    r_surface = np.sqrt(rrup**2 - ztor**2)
    return r_surface * 1000


def fault_top(fault: sources.IsSource) -> float:
    """Return the top of rupture distance (ztor).

    Parameters
    ----------
    fault : sources.IsSource
        The fault to compute rupture top for.

    Returns
    -------
    float
        The top-depth of the rupture in km.
    """
    if isinstance(fault, sources.Point):
        # TODO: backport this into source modelling
        ztor = fault.centroid[2] - fault.width_m / 2 * np.sin(np.radians(fault.dip))
    else:
        ztor = fault.top_m
    ztor /= 1000.0
    return ztor


def find_r_surfaces(
    source_config: SourceConfig,
    magnitudes: Magnitudes,
    rrup_interpolants: npt.NDArray[np.floating],
) -> dict[str, float]:
    """Find r_surfaces for all sources.

    Parameters
    ----------
    source_config : SourceConfig
        The sources to find r_surfaces for.
    magnitudes : Magnitudes
        The magnitudes of the rupture on each source.
    rrup_interpolants : array of floats
        The rrup function of magnitude.

    Returns
    -------
    dict[str, float]
        A key-value mapping of source name to r_surface.
    """
    return {
        fault_name: estimate_r_surface(rrup_interpolants, magnitude, fault_top(fault))
        for fault_name, (fault, magnitude) in utils.dict_zip(
            source_config.source_geometries,
            magnitudes.magnitudes,
        ).items()
    }


def source_max_depth(faults: Iterable[sources.IsSource]) -> float:
    """Find the max depth of the rupture sources.

    Parameters
    ----------
    faults : Iterable[sources.IsSource]
        The faults to find depths for.

    Returns
    -------
    float
        The maximum source depth of the rupture, in meters.
    """
    depths: list[float] = []

    for fault in faults:
        if isinstance(fault, sources.Point):
            # TODO: backport this into source modelling
            bottom_m = fault.centroid[2] + fault.width_m / 2 * np.sin(
                np.radians(fault.dip)
            )
        else:
            bottom_m = fault.bottom_m
        depths.append(bottom_m)

    return max(depths)


def estimate_domain(
    source_config: SourceConfig,
    rrups: dict[str, float],
    nz_outline: shapely.Geometry,
    fault_buffer: float,
) -> BoundingBox:
    """Estimate a domain for a rupture.

    Parameters
    ----------
    source_config : SourceConfig
        The sources in the rupture.
    rrups : dict[str, float]
        The rrups for each source.
    nz_outline : Geometry
        The NZ outline polygon.
    fault_buffer : float
        Buffer width (km) around sources in rupture. Domain edge is
        guaranteed not be within this distance from any source.

    Returns
    -------
    BoundingBox
        The smallest domain containing the sources and areas within
        the estimated surface distance for the given magnitudes.
    """
    # This polygon includes all the faults corners + a variable
    # (default 2km) fault buffer. This polygon must be included in the
    # domain.
    fault_buffer_polygons = [
        shapely.buffer(fault.geometry, fault_buffer * 1000.0)
        for fault in source_config.source_geometries.values()
    ]

    rrup_bounding_polygons = [
        shapely.buffer(fault.geometry, rrup)
        for fault, rrup in utils.dict_zip(
            source_config.source_geometries, rrups
        ).values()
    ]

    # The domain is the minimum area bounding box containing all of
    # the fault corners, and all points on land within rrup distance
    # of a fault corner.
    model_domain = bounding_box.minimum_area_bounding_box_for_polygons_masked(
        must_include=fault_buffer_polygons,
        may_include=rrup_bounding_polygons,
        mask=nz_outline,  # type: ignore[invalid-argument-type]
    )
    return model_domain


def domain_max_depth(
    source_config: SourceConfig,
    magnitudes: Magnitudes,
) -> float:
    """Estimate the maximum reasonable simulation depth.

    Parameters
    ----------
    source_config : SourceConfig
        The faults in the rupture.
    magnitudes : Magnitudes
        The magnitudes of each rupture.

    Returns
    -------
    float
        The estimated maximum reasonable simulation depth, in kilometers.
    """
    max_depth_km = source_max_depth(source_config.source_geometries.values()) / 1000
    magnitude = total_magnitude(magnitudes.magnitudes.values())

    return simulation_max_depth(magnitude, max_depth_km)


def generate_domain(
    source_config: SourceConfig,
    magnitudes: Magnitudes,
    rakes: Rakes,
    velocity_model_parameters: VelocityModelParameters,
) -> DomainParameters:
    """
    Computes simulation domain spatial extent and temporal duration.

    Parameters
    ----------
    source_config : SourceConfig
        Configuration containing the geometries for all faults involved in
        the realisation.
    magnitudes : Magnitudes
        The magnitudes associated with each source in the realisation.
    rakes : Rakes
        The rake angles for the source geometries.
    velocity_model_parameters : VelocityModelParameters
        Parameters defining the velocity model, including Vs30, S-wave
        velocity, and the duration scaling multiplier.

    Returns
    -------
    DomainParameters
        An object containing the computed model domain (bounding box),
        the maximum simulation depth, and the estimated simulation duration.
    """

    rupture_context = rupture_context_from(magnitudes, rakes, velocity_model_parameters)

    rrups = find_r_surfaces(
        source_config, magnitudes, velocity_model_parameters.rrup_interpolants
    )
    nz_outline = utils.get_nz_outline_polygon()
    model_domain = estimate_domain(
        source_config, rrups, nz_outline, velocity_model_parameters.fault_buffer
    )
    sim_duration = estimate_simulation_duration(
        rupture_context, model_domain, source_config.source_geometries.values()
    )
    depth = domain_max_depth(source_config, magnitudes)

    domain_parameters = DomainParameters(
        domain=model_domain,
        depth=depth,
        duration=sim_duration,
    )
    return domain_parameters


@cli.from_docstring(app)
@log_utils.log_call()
def generate_domain_from_realisation(
    realisation_ffp: Annotated[Path, typer.Argument()],
) -> None:
    """Generate domain parameters for a given realisation file.

    This function reads the source, rupture propagation, and velocity model
    information and computes:

    1. The size of the simulation domain,
    2. The simulation duration.

    Both of these values are written to the realisation using `DomainParameters`.

    Parameters
    ----------
    realisation_ffp : Path
        The path to the realisation file from which to read configurations and to which
        the generated domain parameters will be written.

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

    magnitudes = Magnitudes.read_from_realisation(realisation_ffp)
    rakes = Rakes.read_from_realisation(realisation_ffp)
    domain_parameters = generate_domain(
        source_config, magnitudes, rakes, velocity_model_parameters
    )
    domain_parameters.write_to_realisation(realisation_ffp)
    realisations.append_log_entry(realisation_ffp)
