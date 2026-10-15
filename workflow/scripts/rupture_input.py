#!/usr/bin/env python3
"""Rupture generator input configuration.

Description
-----------
Write the fault system and rupture file `rupture-generator` reads, from a realisation.

    rupture-generator rupture.toml realisation.srf

The realisation's `rupture_generation` section mirrors the generator's own tables, and is
passed through unchanged. Everything else the generator needs the realisation already
knows, and this is the only place the two vocabularies meet: the fault geometry becomes a
GeoJSON fault system, and the hypocentre, the 1D velocity model, the per-fault magnitudes
and rakes, the causality tree, the seed and the rupture-speed profile become the tables
that describe them.

This is the successor to `realisation-to-srf`, which drives genslip from the `srf`
section, and runs beside it until that path is retired.

Inputs
------
1. A realisation with source geometries, magnitudes, rakes, a rupture propagation tree,
   a 1D velocity model, seeds, and `rupture_generation` and `rupture_velocity` sections.

Outputs
-------
1. The fault system (GeoJSON), one section per fault.
2. The rupture file (TOML) `rupture-generator` reads, naming the fault system.

Environment
-----------
Can be run in the cybershake container. Can also be run from your own computer using the
`create-rupture-input` command after
`pip install workflow@git+https://github.com/ucgmsim/workflow`.

Usage
-----
`create-rupture-input [OPTIONS] REALISATION_FFP GEOMETRY_OUTPUT_PATH RUPTURE_OUTPUT_PATH`

For More Help
-------------
See the output of `create-rupture-input --help`.
"""

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pyproj
import tomli_w
import typer

from qcore import cli, coordinates
from rupture_generator import config as rupture_config
from source_modelling import sources
from workflow import log_utils, realisations
from workflow.realisations import (
    Magnitudes,
    Rakes,
    RuptureGenerationConfig,
    RupturePropagationConfig,
    RuptureVelocity,
    Seeds,
    SourceConfig,
    VelocityModel1D,
)

app = typer.Typer()

NZTM_EPSG = 2193
"""The projected CRS the generator builds its charts in.

`source_modelling` holds plane bounds in this frame, so every plane comes back out of the
generator where the realisation put it.
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


def build_fault_system(source_config: SourceConfig) -> dict[str, Any]:
    """Describe the realisation's sources as a GeoJSON fault system.

    Parameters
    ----------
    source_config : SourceConfig
        The realisation's source geometries.

    Returns
    -------
    dict
        The `FeatureCollection`, one section per fault.

    Raises
    ------
    TypeError
        If a source is a point, which the generator cannot rupture.
    """
    features = []
    for name, geometry in source_config.source_geometries.items():
        if not isinstance(geometry, (sources.Plane, sources.Fault)):
            raise TypeError(
                f"{name} is a {type(geometry).__name__}; the rupture generator draws "
                "finite faults alone. Use realisation-to-srf for a point source."
            )
        features.append(fault_section(name, geometry))
    return {"type": "FeatureCollection", "features": features}


def build_rupture(
    rupture_generation: RuptureGenerationConfig,
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
            "correlation": rupture_generation.correlation,
        },
        "slip": rupture_generation.slip,
        "rise": rupture_generation.rise,
        "rake": rupture_generation.rake,
        "speed": speed,
        "perturbation": rupture_generation.perturbation,
        "pulse": rupture_generation.pulse,
    }

    parents = {
        child: parent
        for child, parent in rupture_propagation.rupture_causality_tree.items()
        if parent is not None
    }
    # Stated rather than sampled, because the hazard model chose which fault triggers
    # which. The jump points are not carried across: the generator finds where each
    # fault starts from its parent's solved onsets. A single fault has no tree to state.
    if parents:
        rupture["propagation"] = {"type": "predetermined", "parents": parents}

    return rupture


@cli.from_docstring(app)
def generate_rupture_input(
    realisation_ffp: Path,
    geometry_output_path: Path,
    rupture_output_path: Path,
) -> None:
    """Generate the rupture generator's input files from a realisation file.

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

    fault_system = build_fault_system(source_config)
    rupture = build_rupture(
        rupture_generation,
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

    realisations.append_log_entry(realisation_ffp)
