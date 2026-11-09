#!/usr/bin/env python3
"""Rupture generation.

Description
-----------
Draw a realisation's rupture with the rupture generator, in the realisation's NZCVM
velocity model.

    generate-rupture realisation.json realisation.srf

The realisation's `rupture_generation` section mirrors the generator's own tables, and is
passed through unchanged. Everything else the generator needs the realisation already
knows, and this is the only place the two vocabularies meet: the fault geometry becomes a
GeoJSON fault system, and the hypocentre, the 1D velocity model, the per-fault magnitudes
and rakes, the causality tree, the seed and the rupture-speed profile become the tables
that describe them. A point source is drawn on the patch it carries, and the fault the
rupture starts on has its deep rise-time and rupture-speed tapers moved below the
hypocentre, as genslip moved them. The realisation's jump points are not used.

With `rupture_generation.propagation` set to `sampled` or `likeliest`, the generator
chooses the tree from its jump model instead, rooted at the realisation's hypocentre, and
the realisation's causality tree is not used for the SRF.

The draw reads shear speed and density from the realisation's `nzcvm` layers less Backus
averaging, at every subfault and along every crossing between faults, rather than from
the rupture file's 1D `[medium]`.

This is the successor to `realisation-to-srf`, which drives genslip from the `srf`
section, and runs beside it until that path is retired.

Inputs
------
1. A realisation with source geometries (faults, planes or points), magnitudes, rakes, a
   rupture propagation tree, a 1D velocity model, seeds, and `rupture_generation`,
   `rupture_velocity` and `nzcvm` sections.

Outputs
-------
1. The SRF, its header comments carrying the run record.
2. Beside it, the fault system (`.geojson`) and the rupture file (`.toml`) it was drawn
   from.

Environment
-----------
Can be run in the cybershake container, which holds NZCVM's data root.

Usage
-----
`generate-rupture [OPTIONS] REALISATION_FFP SRF_FFP`

For More Help
-------------
See the output of `generate-rupture --help`.
"""

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pyproj
import shapely
import tomli_w
import typer
from nzcvm.config.layers.backus import BackusAveragedLayerConfig

from qcore import cli, coordinates
from rupture_generator import Medium, generate, write_rupture
from rupture_generator import config as rupture_config
from rupture_generator.record import run_record
from source_modelling import sources
from workflow import log_utils, realisations, velocity_model
from workflow.realisations import (
    Magnitudes,
    NZCVMSettings,
    Rakes,
    RuptureGenerationConfig,
    RupturePropagationConfig,
    RuptureVelocity,
    Seeds,
    SourceConfig,
    VelocityModel1D,
)
from workflow.velocity_model import NZTM_EPSG

app = typer.Typer()

MAX_SHEAR_SPEED_KM_S = 7.0
"""A conservative bound on any shear speed NZCVM returns.

The fastest model nodes are the EP2020 tomography's mantle, 6.2 km/s at 750 km, and
EP2010, EP2025 and the Chow 2020 mix share that reference. Basins stay below 3.2 km/s
and the default offshore 1D model below 4.7 km/s, the query only interpolates and
blends between nodes, and the clamps only raise Vs to their floors. So 7.0 km/s leaves
margin for a future model while staying finite.

The jump search prunes crossings by it, and a crossing that reads a faster one fails
rather than drawing a wrong rupture.
"""

VERTICAL_DIP_TOLERANCE_DEG = 1e-6
"""How close to 90 degrees counts as vertical.

`sources.Plane.dip_dir` is 0.0 for a vertical plane rather than a bearing, because there
is no side to dip towards. The generator still wants a dip direction that picks a side
of the trace, so a vertical fault is given the one square to its strike.
"""

_GEOD = pyproj.Geod(ellps="WGS84")


def _azimuth_deg(start: np.ndarray, end: np.ndarray) -> float:
    """The true azimuth from one NZTM position to another, in degrees.

    Parameters
    ----------
    start : np.ndarray
        The first position, NZTM (northing, easting).
    end : np.ndarray
        The second position, NZTM (northing, easting).

    Returns
    -------
    float
        The azimuth at `start`, clockwise from north.
    """
    (lat0, lon0), (lat1, lon1) = coordinates.nztm_to_wgs_depth(np.vstack([start, end]))[
        :, :2
    ]
    azimuth, _, _ = _GEOD.inv(lon0, lat0, lon1, lat1)
    return float(azimuth) % 360.0


def fault_section(name: str, geometry: sources.Plane | sources.Fault) -> dict[str, Any]:
    """Describe one fault as a GeoJSON section.

    A section is a trace hung on one dip, one depth range and one dip direction: its
    bottom edge is the trace moved by a single vector. That is how `sources.Fault` builds
    its planes too (it refuses planes whose dip directions differ), so the generator
    rebuilds the realisation's corners to within metres.

    The generator orders a trace so the fault dips to its right, and `sources.Plane`
    already orders its corners that way, so the trace is never reversed and the
    hypocentre's position along strike means the same thing to both.

    Parameters
    ----------
    name : str
        The fault's name.
    geometry : sources.Plane or sources.Fault
        The fault.

    Returns
    -------
    dict
        The GeoJSON `Feature`.
    """
    planes = [geometry] if isinstance(geometry, sources.Plane) else geometry.planes
    # Consecutive planes share a trace corner, so the whole trace is the first plane's
    # near corner followed by every plane's far one. `bounds` are NZTM (y, x, depth).
    trace = np.vstack([planes[0].bounds[0]] + [plane.bounds[1] for plane in planes])[
        :, :2
    ]
    dip_deg = float(np.mean([plane.dip for plane in planes]))

    # The generator measures its dip direction 1 km out from the trace's middle point,
    # as a true azimuth, and projects it. The first plane's top-to-bottom vector is that
    # direction in NZTM, so it is read back as an azimuth from the same point. nshmdb
    # turns a section's dip direction into an NZTM bearing at each plane's own start, so
    # its planes hang from vectors a grid convergence apart, and the generator's bottom
    # corners can sit metres from theirs: far inside a subfault.
    if abs(dip_deg - 90.0) < VERTICAL_DIP_TOLERANCE_DEG:
        dip_direction_deg = (_azimuth_deg(trace[0], trace[-1]) + 90.0) % 360.0
    else:
        middle = trace[len(trace) // 2]
        heading = planes[0].bounds[-1, :2] - planes[0].bounds[0, :2]
        heading *= 1000.0 / np.linalg.norm(heading)
        dip_direction_deg = _azimuth_deg(middle, middle + heading)

    located = coordinates.nztm_to_wgs_depth(trace)
    feature = {
        "type": "Feature",
        "geometry": {
            "type": "LineString",
            "coordinates": [[float(lon), float(lat)] for lat, lon in located[:, :2]],
        },
        "properties": {
            "name": name,
            "dip_deg": dip_deg,
            "dip_direction_deg": dip_direction_deg,
            "upper_depth_km": planes[0].top_m / 1000.0,
            "lower_depth_km": planes[0].bottom_m / 1000.0,
        },
    }
    return feature


def point_plane(point: sources.Point) -> sources.Plane:
    """The plane a point source stands for.

    The generator draws finite faults alone, so a point source is drawn on the patch it
    already carries: `length_m` by `width_m` about the point, at its strike, dip and dip
    direction, which `gcmt-to-realisation` sizes by magnitude scaling. genslip's path
    gave a point source one subfault and a cosine slip-rate function; this one has a slip
    distribution and the finite-fault pulse instead.

    A patch that would break the surface is lowered until its top edge is at the
    surface, because the generator refuses a fault above ground.

    Parameters
    ----------
    point : sources.Point
        The point source.

    Returns
    -------
    sources.Plane
        Its patch.
    """
    latitude, longitude, depth_m = point.coordinates
    depth_km = depth_m / 1000.0 + max(0.0, -point.top_m / 1000.0)
    if depth_km != depth_m / 1000.0:
        log_utils.get_logger(__name__).info(
            "Lowered a point source's patch to the surface",
            lowered_m=round(depth_km * 1000.0 - depth_m),
        )
    dip_direction = (
        {}
        if abs(point.dip - 90.0) < VERTICAL_DIP_TOLERANCE_DEG
        else {"dip_dir": point.dip_dir}
    )
    return sources.Plane.from_centroid_strike_dip(
        np.array([latitude, longitude, depth_km]),
        point.dip,
        point.length,
        point.width,
        strike=point.strike,
        **dip_direction,
    )


def finite_geometries(
    source_config: SourceConfig,
) -> dict[str, sources.Plane | sources.Fault]:
    """The realisation's sources as finite faults, each point source on its patch.

    Parameters
    ----------
    source_config : SourceConfig
        The realisation's source geometries.

    Returns
    -------
    dict of str to sources.Plane or sources.Fault
        Each source's finite fault, by name.
    """
    return {
        name: point_plane(geometry) if isinstance(geometry, sources.Point) else geometry
        for name, geometry in source_config.source_geometries.items()
    }


def build_fault_system(
    geometries: dict[str, sources.Plane | sources.Fault],
) -> dict[str, Any]:
    """Describe the realisation's faults as a GeoJSON fault system.

    Parameters
    ----------
    geometries : dict of str to sources.Plane or sources.Fault
        The faults, by name.

    Returns
    -------
    dict
        The `FeatureCollection`, one section per fault.
    """
    return {
        "type": "FeatureCollection",
        "features": [
            fault_section(name, geometry) for name, geometry in geometries.items()
        ],
    }


def deepened_taper(
    profile: dict[str, Any], hypocentre_depth_km: float
) -> dict[str, Any] | None:
    """Move a depth profile's deep taper below the hypocentre, as genslip does.

    genslip starts its deep rise-time and rupture-speed tapers at the hypocentre's depth
    when that is deeper than where they would otherwise start, so a rupture never
    nucleates inside the slow, long-rise-time zone. A four-point profile is a shallow
    taper over its first two points and a deep one over its last two; the deep one keeps
    its width and moves down.

    Parameters
    ----------
    profile : dict
        A depth profile, as the generator's TOML states it.
    hypocentre_depth_km : float
        The hypocentre's depth, in kilometres.

    Returns
    -------
    dict or None
        The moved profile, or None when it needs no moving or is not a four-point
        profile.
    """
    if profile.get("type") != "profile" or len(profile["depth_km"]) != 4:
        return None
    shift_km = hypocentre_depth_km - profile["depth_km"][2]
    if shift_km <= 0.0:
        return None
    shallow, deep = profile["depth_km"][:2], profile["depth_km"][2:]
    return profile | {"depth_km": [*shallow, *(depth + shift_km for depth in deep)]}


def build_rupture(
    rupture_generation: RuptureGenerationConfig,
    geometries: dict[str, sources.Plane | sources.Fault],
    rupture_propagation: RupturePropagationConfig,
    magnitudes: Magnitudes,
    rakes: Rakes,
    velocity_model_1d: VelocityModel1D,
    rupture_velocity: RuptureVelocity,
    seeds: Seeds,
    geometry_path: Path,
) -> dict[str, Any]:
    """Assemble the rupture file from the realisation.

    Parameters
    ----------
    rupture_generation : RuptureGenerationConfig
        The generator's own tables, passed through.
    geometries : dict of str to sources.Plane or sources.Fault
        The faults, by name, which place the hypocentre at depth.
    rupture_propagation : RupturePropagationConfig
        Supplies the hypocentre and the causality tree.
    magnitudes : Magnitudes
        The per-fault magnitudes.
    rakes : Rakes
        The per-fault rakes.
    velocity_model_1d : VelocityModel1D
        The 1D velocity model the rupture propagates through.
    rupture_velocity : RuptureVelocity
        The rupture-speed depth profile.
    seeds : Seeds
        Supplies the rupture seed.
    geometry_path : Path
        The fault system, as the rupture file names it: relative to the rupture file.

    Returns
    -------
    dict
        The rupture file's contents.

    Raises
    ------
    ValueError
        If the rupture velocity varies with slip, which the generator does not model.
    """
    if rupture_velocity.rvfrac_slip_sig is not None:
        raise ValueError(
            "rupture_velocity.rvfrac_slip_sig is set, and the rupture generator has no "
            "rupture speed that varies with slip. Use realisation-to-srf."
        )

    initial_fault = rupture_propagation.initial_fault
    strike_fraction, dip_fraction = (
        float(value) for value in rupture_propagation.hypocentre
    )

    # The realisation's model is layer thicknesses; the generator indexes by the depth to
    # each layer's bottom, which is their running sum.
    model = velocity_model_1d.model
    medium = {
        "bottom_depth_km": np.cumsum(model["thickness"]).astype(float).tolist(),
        "shear_speed_km_s": model["Vs"].astype(float).tolist(),
        "density_g_cm3": model["rho"].astype(float).tolist(),
    }

    # genslip's shal_vrup and deep_vrup: the front slows to these fractions of its
    # speed across each transition, linearly, and is unmodified between them.
    transitions_km = [
        rupture_velocity.shallow_depth - rupture_velocity.shallow_transition_range,
        rupture_velocity.shallow_depth + rupture_velocity.shallow_transition_range,
        rupture_velocity.deep_depth - rupture_velocity.deep_transition_range,
        rupture_velocity.deep_depth + rupture_velocity.deep_transition_range,
    ]
    speed = {
        "velocity_fraction": rupture_velocity.rvfrac,
        "factor": {
            "type": "profile",
            "depth_km": transitions_km,
            "values": [
                rupture_velocity.rvfrac_shal,
                1.0,
                1.0,
                rupture_velocity.rvfrac_deep,
            ],
        },
    }

    rupture: dict[str, Any] = {
        "seed": seeds.genslip_seed,
        "geometry": {
            "path": str(geometry_path),
            "crs": f"EPSG:{NZTM_EPSG}",
            "spacing_km": rupture_generation.spacing_km,
        },
        "hypocentre": {
            "segment": initial_fault,
            "strike_fraction": strike_fraction,
            "dip_fraction": dip_fraction,
        },
        "medium": medium,
        # `per_fault` rather than `finite`: the hazard model already decided how the
        # moment divides between the faults, and genslip drew each fault at its own.
        "source": {
            "type": "per_fault",
            "magnitudes": {
                name: float(magnitude)
                for name, magnitude in magnitudes.magnitudes.items()
            },
            "rakes": {name: float(rake) for name, rake in rakes.rakes.items()},
        },
        "slip": rupture_generation.slip,
        "rise": rupture_generation.rise,
        "rake": rupture_generation.rake,
        "speed": speed,
        "perturbation": rupture_generation.perturbation,
        "pulse": rupture_generation.pulse,
    }

    # The fault the rupture starts on gets its deep tapers moved below the hypocentre.
    # The others are left as they are: where the front enters them is the generator's
    # to find.
    hypocentre_depth_km = (
        geometries[initial_fault].fault_coordinates_to_wgs_depth_coordinates(
            np.array([strike_fraction, dip_fraction])
        )[2]
        / 1000.0
    )
    overrides = {}
    for table, factor in (
        ("rise", rupture_generation.rise.get("factor")),
        ("speed", speed["factor"]),
    ):
        deepened = (
            None if factor is None else deepened_taper(factor, hypocentre_depth_km)
        )
        if deepened is not None:
            overrides[table] = {"factor": deepened}
    if overrides:
        rupture["segments"] = {initial_fault: overrides}

    propagation = propagation_table(
        rupture_generation, rupture_propagation, len(geometries)
    )
    if propagation is not None:
        rupture["propagation"] = propagation

    return rupture


def propagation_table(
    rupture_generation: RuptureGenerationConfig,
    rupture_propagation: RupturePropagationConfig,
    fault_count: int,
) -> dict[str, Any] | None:
    """The generator's `[propagation]` table: which fault triggers which.

    By default the tree is stated rather than drawn, because the hazard model chose which
    fault triggers which. `rupture_generation.propagation` can instead have the generator
    draw a tree from its jump model (`sampled`, with the realisation's seed) or take the
    jump model's likeliest (`likeliest`). Either way the tree is rooted at the fault the
    realisation's hypocentre is on. The realisation's jump points are never used: the
    generator finds where each fault starts from its parent's solved onsets, under its
    own jump model.

    Parameters
    ----------
    rupture_generation : RuptureGenerationConfig
        Supplies the propagation type and the jump model.
    rupture_propagation : RupturePropagationConfig
        Supplies the causality tree, for `predetermined`.
    fault_count : int
        How many faults the realisation has.

    Returns
    -------
    dict or None
        The table, or None for a single fault, which has no tree to state.
    """
    if fault_count < 2:
        return None
    propagation = rupture_generation.propagation
    if propagation == "predetermined":
        return {
            "type": "predetermined",
            "parents": {
                child: parent
                for child, parent in rupture_propagation.rupture_causality_tree.items()
                if parent is not None
            },
            "model": rupture_generation.jump,
        }
    log_utils.get_logger(__name__).warning(
        "The rupture generator chooses its own tree: the SRF will not follow "
        "rupture_propagation.rupture_causality_tree, only its hypocentre",
        propagation=propagation,
        initial_fault=rupture_propagation.initial_fault,
    )
    return {"type": propagation, "model": rupture_generation.jump}


def write_rupture_input(
    realisation_ffp: Path,
    geometry_output_path: Path,
    rupture_output_path: Path,
) -> None:
    """Write the rupture generator's input files from a realisation file.

    Parameters
    ----------
    realisation_ffp : Path
        Path to the realisation file to read.
    geometry_output_path : Path
        Path to write the fault system (GeoJSON).
    rupture_output_path : Path
        Path to write the rupture file (TOML) `rupture-generator` reads.
    """
    logger = log_utils.get_logger(__name__)
    metadata = realisations.RealisationMetadata.read_from_realisation(realisation_ffp)
    source_config = SourceConfig.read_from_realisation(realisation_ffp)
    rupture_propagation = RupturePropagationConfig.read_from_realisation(
        realisation_ffp
    )
    magnitudes = Magnitudes.read_from_realisation(realisation_ffp)
    rakes = Rakes.read_from_realisation(realisation_ffp)
    seeds = Seeds.read_from_realisation_or_random(realisation_ffp)
    rupture_generation = RuptureGenerationConfig.read_from_realisation_or_defaults(
        realisation_ffp, metadata.defaults_version
    )
    velocity_model_1d = VelocityModel1D.read_from_realisation_or_defaults(
        realisation_ffp, metadata.defaults_version
    )
    rupture_velocity = RuptureVelocity.read_from_realisation_or_defaults(
        realisation_ffp, metadata.defaults_version
    )

    geometries = finite_geometries(source_config)
    fault_system = build_fault_system(geometries)
    rupture = build_rupture(
        rupture_generation,
        geometries,
        rupture_propagation,
        magnitudes,
        rakes,
        velocity_model_1d,
        rupture_velocity,
        seeds,
        Path(
            os.path.relpath(
                geometry_output_path.resolve(), rupture_output_path.resolve().parent
            )
        ),
    )

    geometry_output_path.write_text(json.dumps(fault_system, indent=2))
    rupture_output_path.write_text(tomli_w.dumps(rupture))
    # The generator's own strict reader: an unknown key or a value out of range in the
    # realisation's tables fails here, before a job is queued to draw it.
    rupture_config.load(rupture_output_path)
    logger.info(
        "Wrote rupture generator input",
        geometry=str(geometry_output_path),
        rupture=str(rupture_output_path),
    )


def nzcvm_query(
    settings: NZCVMSettings, east_north_m: np.ndarray
) -> velocity_model.PointQuery:
    """Load the NZCVM layers around some positions, to query at points.

    Backus averaging is left out: it averages over a grid's cells, and has no meaning
    at a point. Depths are below the topography, where SW4 places the rupture, so the
    rock read is the rock SW4 simulates with (clamps and the Ely taper included).

    Parameters
    ----------
    settings : NZCVMSettings
        The realisation's NZCVM settings.
    east_north_m : np.ndarray
        NZTM positions, ``(n, 2)``, easting and northing in metres, that every later
        query lies within 1 km of.

    Returns
    -------
    PointQuery
        The velocity model's query, its z the depth below the topography in metres.
    """
    return velocity_model.point_query(
        settings,
        shapely.MultiPoint(east_north_m).convex_hull.buffer(1000.0),
        exclude=(BackusAveragedLayerConfig,),
        below_ground=True,
    )


def nzcvm_medium(query: velocity_model.PointQuery) -> Medium:
    """Wrap a velocity model query as the rupture generator's medium.

    The generator reads shear speed and density at the same positions in turn, so the
    last positions' read is kept for the second.

    Parameters
    ----------
    query : PointQuery
        The velocity model, in NZTM metres and NZCVM's units.

    Returns
    -------
    Medium
        Shear speed (km/s) and density (g/cm^3) at chart positions (NZTM km).
    """
    cache: dict[str, np.ndarray] = {}

    def read(positions_km: np.ndarray, component: str) -> np.ndarray:
        if not (cache and np.array_equal(cache["at"], positions_km)):
            east, north, depth = np.moveaxis(positions_km * 1000.0, -1, 0)
            qualities = query(east, north, depth)
            # NZCVM's m/s and kg/m^3 are a thousand km/s and g/cm^3.
            cache.update(
                at=positions_km.copy(),
                vs=qualities["vs"] / 1000.0,
                rho=qualities["rho"] / 1000.0,
            )
        return cache[component]

    return Medium(
        shear_speed_km_s=lambda positions_km: read(positions_km, "vs"),
        density_g_cm3=lambda positions_km: read(positions_km, "rho"),
        max_shear_speed_km_s=MAX_SHEAR_SPEED_KM_S,
    )


@cli.from_docstring(app)
def generate_rupture(realisation_ffp: Path, srf_ffp: Path) -> None:
    """Draw a realisation's rupture in its NZCVM velocity model.

    Parameters
    ----------
    realisation_ffp : Path
        Path to the realisation file to read.
    srf_ffp : Path
        Path to write the SRF. The fault system (`.geojson`) and rupture file
        (`.toml`) it is drawn from are written beside it.
    """
    rupture_ffp = srf_ffp.with_suffix(".toml")
    write_rupture_input(realisation_ffp, srf_ffp.with_suffix(".geojson"), rupture_ffp)
    metadata = realisations.RealisationMetadata.read_from_realisation(realisation_ffp)
    settings = NZCVMSettings.read_from_realisation_or_defaults(
        realisation_ffp, metadata.defaults_version
    )
    config = rupture_config.load(rupture_ffp)
    scenario = rupture_config.build(config)
    nodes_km = [
        chart.nodes[..., :2].reshape(-1, 2) for chart in scenario.realisation.values()
    ]
    # The `per_fault` source takes no medium, so the 1D `[medium]` the rupture file
    # states is read nowhere and the draw reads only NZCVM.
    drawn = generate(
        scenario.realisation,
        nzcvm_medium(nzcvm_query(settings, np.concatenate(nodes_km) * 1000.0)),
        scenario.sources,
        scenario.model,
        seed=scenario.seed,
        jump_propagation_model=scenario.jump_model,
    )
    write_rupture(
        str(srf_ffp),
        drawn,
        comments=[
            *run_record(rupture_ffp, config).splitlines(),
            "medium = NZCVM (nzcvm layers less backus, below topography), not [medium]",
        ],
    )
    log_utils.get_logger(__name__).info("Wrote SRF", srf=str(srf_ffp))
    realisations.append_log_entry(realisation_ffp)
