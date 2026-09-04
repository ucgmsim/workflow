#!/usr/bin/env python3
"""Rupture generator input configuration.

Description
-----------
Write the two configuration files `rupture-generator` reads, from a realisation.

The generator splits a rupture into a geometry, which is digitised once and reused, and
a source, which is what varies between realisations:

    rupture-generator mesh     geometry.toml  mesh.h5
    rupture-generator generate rupture.toml   mesh.h5  realisation.srf

so this writes both. It is the only place the realisation's vocabulary and the
generator's meet: the realisation's `srf` section mirrors
`rupture_generator.config.rupture.RuptureConfig` section for section, and everything the
realisation already knows -- the fault geometry, the hypocentre, the velocity model, the
per-fault magnitudes and rakes, the causality tree, the seed and the rupture-speed
profile -- is injected here rather than being written down twice.

Inputs
------
1. A realisation with source geometries, magnitudes, rakes, a rupture propagation tree,
   a 1D velocity model, and `srf` and `rupture_velocity` sections.

Outputs
-------
1. A geometry configuration (TOML) for `rupture-generator mesh`.
2. A rupture configuration (TOML) for `rupture-generator generate`.

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

from pathlib import Path

import numpy as np
import pyproj
import tomli_w
import typer

from qcore import cli, coordinates
from rupture_generator import moment as rupture_moment
from rupture_generator import stages
from rupture_generator import timing as rupture_timing
from rupture_generator.config.geometry import (
    Discretisation,
    FaultConfig,
    GeometryConfig,
    LonLat,
    PlaneConfig,
    PointConfig,
    SurfaceConfig,
)
from rupture_generator.config.rupture import (
    FieldConfig,
    HypocentreConfig,
    PerFaultSourceConfig,
    PointSourceConfig,
    PredeterminedPropagation,
    RandomConfig,
    RuptureConfig,
    SlipConfig,
    TimingConfig,
    VelocityModelConfig,
)
from source_modelling import sources
from workflow import log_utils, realisations
from workflow.realisations import (
    Magnitudes,
    Rakes,
    RealisationMetadata,
    RupturePropagationConfig,
    RuptureVelocity,
    Seeds,
    SourceConfig,
    SRFConfig,
    VelocityModel1D,
)

app = typer.Typer()

MINIMUM_SUPPORTED_PLANE_FRACTION = 0.5
"""How much of its own length a plane must keep at depth for its trace bend to stand.

At a bend the two planes either side share a bottom column placed along the bisector, so
each plane's bottom edge shortens by about `reach * sin(deflection / 2)` at that end,
where `reach` is how far the fault steps horizontally on its way down. A plane shorter
than twice that closes to a triangle at depth and there is no grid on it -- the generator
refuses the mesh rather than sample it. Half is the fraction the generator's own
converter settled on against this scenario.
"""

NZTM_EPSG = 2193
"""The projected CRS the mesh is built in.

The generator refuses a geographic CRS: it works in a Cartesian frame and leaves it only
to place positions on the globe. This is the same frame `source_modelling` already holds
plane bounds in, so nothing is reprojected on the way through.
"""

VERTICAL_DIP_TOLERANCE_DEG = 1e-6
"""How close to 90 degrees counts as vertical when reading the dip direction off a plane.

`sources.Plane.dip_dir` returns 0.0 for a vertical plane rather than a bearing, because
there is no side to dip towards, so the sign test below would read that 0.0 as a real
bearing and pick a side at random. A vertical fault hangs on neither side and the
generator only needs *a* value, so it gets `right`.
"""


def _lon_lat(wgs_depth_coordinate: np.ndarray) -> LonLat:
    """Convert a (latitude, longitude, depth) row into the generator's LonLat.

    Parameters
    ----------
    wgs_depth_coordinate : np.ndarray
        A coordinate in `source_modelling`'s (latitude, longitude, depth) order.

    Returns
    -------
    LonLat
        The same position, longitude first.
    """
    return LonLat(
        longitude_deg=float(wgs_depth_coordinate[1]),
        latitude_deg=float(wgs_depth_coordinate[0]),
    )


def _fault_dip_deg(planes: list[sources.Plane]) -> float:
    """The one dip a run of planes shares, in degrees.

    `sources.Fault` has already checked that its planes agree on dip, so the spread
    across them is float noise from the corner positions the dip is read back out of --
    around 1e-12 degrees. It is not harmless: the generator hangs each plane's bottom
    edge from its own dip, and fusing planes whose dips differ in the last bits leaves
    the joined grid's rows a fraction of a micron out of step, which `validate_chart`
    refuses as a mesh that "came from somewhere else". Averaging states the shared value
    once.

    Parameters
    ----------
    planes : list of sources.Plane
        The planes of one fault.

    Returns
    -------
    float
        Their common dip.
    """
    return float(np.mean([plane.dip for plane in planes]))


def _simplify_trace(trace: np.ndarray, reach_km: float) -> tuple[np.ndarray, float]:
    """Drop trace points the fault's own depth cannot support.

    A digitised trace can be finer than the surface beneath it: a short plane at a sharp
    bend loses its whole bottom edge to the bend and closes to a triangle at depth. Such
    a trace is thinned until every plane keeps at least
    `MINIMUM_SUPPORTED_PLANE_FRACTION` of its length down there, dropping at each step
    whichever point moves the trace least.

    This *moves the fault*, so the worst displacement is returned for the caller to
    report: a few hundred metres on a fault kilometres deep is well inside what the trace
    itself is known to, and a reader should be told rather than left to assume.

    Parameters
    ----------
    trace : np.ndarray
        The trace points, projected, one row per point.
    reach_km : float
        How far the fault steps horizontally between its top and bottom edges.

    Returns
    -------
    tuple of (np.ndarray, float)
        The kept trace points, and how far the trace moved, in kilometres.
    """

    def deviation_km(points: np.ndarray, index: int) -> float:
        """How far `index` sits from the line that would replace it."""
        span = points[index + 1] - points[index - 1]
        length = float(np.linalg.norm(span))
        if length < np.finfo(float).eps:
            return 0.0
        offset = points[index] - points[index - 1]
        # The 2-D cross product, written out: `np.cross` takes 3-vectors only.
        area = span[0] * offset[1] - span[1] * offset[0]
        return float(abs(area) / length) / 1000.0

    def unsupported(points: np.ndarray) -> list[int]:
        """Which interior points leave a plane too short for its own depth."""
        offenders = []
        for index in range(1, len(points) - 1):
            before = points[index] - points[index - 1]
            after = points[index + 1] - points[index]
            lengths = (
                float(np.linalg.norm(before)) / 1000.0,
                float(np.linalg.norm(after)) / 1000.0,
            )
            if min(lengths) < np.finfo(float).eps:
                offenders.append(index)
                continue
            turn = np.arccos(
                np.clip(
                    float(np.dot(before, after))
                    / (lengths[0] * lengths[1] * 1000.0**2),
                    -1.0,
                    1.0,
                )
            )
            lost_km = reach_km * np.sin(turn / 2.0)
            if min(lengths) < MINIMUM_SUPPORTED_PLANE_FRACTION**-1 * lost_km:
                offenders.append(index)
        return offenders

    kept = trace
    worst_km = 0.0
    while len(kept) > 2:
        offenders = unsupported(kept)
        if not offenders:
            break
        victim = min(offenders, key=lambda index: deviation_km(kept, index))
        worst_km = max(worst_km, deviation_km(kept, victim))
        kept = np.delete(kept, victim, axis=0)

    return kept, worst_km


def _dip_direction(plane: sources.Plane) -> str:
    """Which side of its trace a plane hangs on, as the generator names it.

    `right` means the fault dips away to the right of the walk along the trace, so the
    dip direction is the strike turned clockwise by ninety degrees.

    Parameters
    ----------
    plane : sources.Plane
        The plane to read.

    Returns
    -------
    str
        Either `"right"` or `"left"`.
    """
    if abs(plane.dip - 90.0) < VERTICAL_DIP_TOLERANCE_DEG:
        return "right"
    return "right" if (plane.dip_dir - plane.strike) % 360.0 < 180.0 else "left"


def build_geometry(
    source_config: SourceConfig,
    srf_config: SRFConfig,
    metadata: RealisationMetadata,
) -> GeometryConfig:
    """Describe the realisation's sources as a geometry the generator can mesh.

    The planes are read through `sources.Plane`'s own properties rather than
    reconstructed from corner positions: a `sources.Fault` has already checked that its
    planes share a dip and a width and run end to end along strike, which is exactly the
    invariant a run of `PlaneConfig`s hanging from one origin expresses.

    Note that the causality tree is *not* written here. It belongs to the rupture rather
    than to the fault, `GeometryConfig` has no field for it, and the generator refuses
    unknown keys.

    Parameters
    ----------
    source_config : SourceConfig
        The realisation's source geometries.
    srf_config : SRFConfig
        Supplies the subfault size the planes are discretised at.
    metadata : RealisationMetadata
        Supplies the title.

    Returns
    -------
    GeometryConfig
        The geometry configuration.

    Raises
    ------
    ValueError
        If a source is neither a point, a plane, nor a fault.
    """
    logger = log_utils.get_logger(__name__)
    discretisation = Discretisation(subfault_size_km=srf_config.resolution)

    surfaces: list[SurfaceConfig] = []
    for name, geometry in source_config.source_geometries.items():
        match geometry:
            case sources.Point():
                surfaces.append(
                    PointConfig(
                        name=name,
                        centre=_lon_lat(geometry.coordinates),
                        # `Point.coordinates` carries depth in metres.
                        depth_km=float(geometry.coordinates[2]) / 1000.0,
                        strike_deg=float(geometry.strike) % 360.0,
                        dip_deg=float(geometry.dip),
                        size_km=float(geometry.length),
                    )
                )
            case sources.Plane() | sources.Fault():
                planes = (
                    [geometry]
                    if isinstance(geometry, sources.Plane)
                    else geometry.planes
                )
                dip_deg = _fault_dip_deg(planes)
                top_depth_km = planes[0].top_m / 1000.0
                bottom_depth_km = (
                    float(np.mean([plane.bottom_m for plane in planes])) / 1000.0
                )
                dip_direction = _dip_direction(planes[0])

                # Consecutive planes share a trace corner, so the whole trace is the
                # first plane's near corner followed by every plane's far one.
                trace = np.vstack(
                    [planes[0].bounds[0]] + [plane.bounds[1] for plane in planes]
                )
                reach_km = (bottom_depth_km - top_depth_km) / np.tan(
                    np.radians(dip_deg)
                )
                kept, moved_km = _simplify_trace(trace[:, :2], reach_km)
                if len(kept) < len(trace):
                    logger.info(
                        "Thinned a trace its own depth cannot support",
                        fault=name,
                        points_before=len(trace),
                        points_after=len(kept),
                        worst_deviation_m=round(moved_km * 1000.0),
                    )

                # `bounds` are (northing, easting, depth) and the thinning worked on the
                # first two, so depth is put back before projecting.
                located = coordinates.nztm_to_wgs_depth(
                    np.column_stack([kept, np.zeros(len(kept))])
                )
                surfaces.append(
                    FaultConfig(
                        name=name,
                        origin=_lon_lat(located[0]),
                        top_depth_km=top_depth_km,
                        planes=[
                            PlaneConfig(
                                end=_lon_lat(point),
                                dip_deg=dip_deg,
                                bottom_depth_km=bottom_depth_km,
                                dip_direction=dip_direction,
                                discretisation=discretisation,
                            )
                            for point in located[1:]
                        ],
                    )
                )
            case _:
                raise ValueError(
                    f"{name} is a {type(geometry).__name__}, which is not a geometry "
                    "the rupture generator can mesh."
                )

    return GeometryConfig(
        crs=pyproj.CRS(NZTM_EPSG),
        surfaces=surfaces,
        title=metadata.name,
    )


def build_rupture(
    source_config: SourceConfig,
    srf_config: SRFConfig,
    rupture_propagation: RupturePropagationConfig,
    magnitudes: Magnitudes,
    rakes: Rakes,
    velocity_model_1d: VelocityModel1D,
    rupture_velocity: RuptureVelocity,
    seeds: Seeds,
    metadata: RealisationMetadata,
) -> RuptureConfig:
    """Translate the realisation's configuration into the generator's.

    The realisation's `srf` section mirrors `RuptureConfig` section for section, so each
    group is splatted straight in. What it does not carry is injected here: the
    hypocentre, the velocity model, the per-fault magnitudes and rakes, the seed, the
    causality tree, and the rupture-speed profile, which lives in its own section because
    `hf-sim` reads the same physical values.

    Parameters
    ----------
    source_config : SourceConfig
        Supplies the fault the hypocentre fractions are resolved against.
    srf_config : SRFConfig
        The realisation's rupture generation configuration.
    rupture_propagation : RupturePropagationConfig
        Supplies the hypocentre and the causality tree.
    magnitudes : Magnitudes
        The per-fault magnitudes.
    rakes : Rakes
        The per-fault rakes.
    velocity_model_1d : VelocityModel1D
        The 1D velocity model the rupture propagates through.
    rupture_velocity : RuptureVelocity
        The depth-dependent rupture-speed profile.
    seeds : Seeds
        Supplies the rupture seed.
    metadata : RealisationMetadata
        Supplies the title.

    Returns
    -------
    RuptureConfig
        The generator's configuration.
    """
    initial_fault = source_config.source_geometries[rupture_propagation.initial_fault]

    # The realisation gives the hypocentre as fractions along strike and down dip; the
    # generator uses in-fault arc lengths, so they are resolved against the initial
    # fault's own extent. The meshed extent can differ by less than a subfault, because
    # a plane is cut into whole cells.
    hypocentre = HypocentreConfig(
        fault=rupture_propagation.initial_fault,
        strike_km=float(rupture_propagation.hypocentre[0]) * initial_fault.length,
        dip_km=float(rupture_propagation.hypocentre[1]) * initial_fault.width,
    )

    # The realisation's model is layer thicknesses; the generator indexes by the depth to
    # each layer's bottom, which is their running sum.
    model = velocity_model_1d.model
    velocity_model = VelocityModelConfig(
        bottom_depth_km=[float(depth) for depth in np.cumsum(model["thickness"])],
        shear_speed_km_s=[float(vs) for vs in model["Vs"]],
        density_g_cm3=[float(rho) for rho in model["rho"]],
    )

    # `from_dict` rather than a `**` splat, here and for the other mirrored groups: the
    # realisation's `timing` carries nested tables (the depth ramps), and a splat would
    # hand them over as plain dicts -- unvalidated, and without the attributes the
    # generator's stages read off them.
    timing = TimingConfig.from_dict(
        srf_config.timing
        | {
            "shallow_ramp": {
                "centre_km": rupture_velocity.shallow_depth,
                "half_width_km": rupture_velocity.shallow_transition_range,
            },
            "deep_ramp": {
                "centre_km": rupture_velocity.deep_depth,
                "half_width_km": rupture_velocity.deep_transition_range,
            },
            "shallow_speed_factor": rupture_velocity.rvfrac_shal,
            "deep_speed_factor": rupture_velocity.rvfrac_deep,
        }
    )

    common = {
        "hypocentre": hypocentre,
        "velocity_model": velocity_model,
        "timing": timing,
        "field": FieldConfig.from_dict(
            srf_config.field | {"velocity_fraction": rupture_velocity.rvfrac}
        ),
        # One event seed; the generator draws every stage from its own named substream of
        # it. `realisation` selects an independent stream from the same seed, which is
        # what makes a campaign restartable, and the realisation carries one event.
        "random": RandomConfig(seed=seeds.rupture_seed, realisation=0),
        "title": metadata.name,
    }

    # A point source draws no fields, so the generator refuses a `[slip]` section beside
    # one -- a section that would be read and ignored is a different earthquake than the
    # one written down.
    if len(source_config.source_geometries) == 1 and isinstance(
        initial_fault, sources.Point
    ):
        name = rupture_propagation.initial_fault
        magnitude = float(magnitudes.magnitudes[name])
        dip_deg = float(initial_fault.dip)
        rake_deg = float(rakes.rakes[name])
        return RuptureConfig(
            source=PointSourceConfig(
                magnitude=magnitude,
                # A point source states its rise time outright, where a finite one
                # derives it from the moment. This is that same derivation, through the
                # generator's own function so the two cannot disagree: the realisation
                # carries the Graves & Pitarka coefficient, not a rise time.
                rise_time_s=stages.average_rise_time_s(
                    rupture_moment.seismic_moment_nm(magnitude),
                    srf_config.source["rise_time_coefficient"],
                    rupture_timing.alpha_t(dip_deg, rake_deg),
                ),
                average_dip_deg=dip_deg,
                average_rake_deg=rake_deg,
            ),
            **common,
        )

    parents = {
        child: parent
        for child, parent in rupture_propagation.rupture_causality_tree.items()
        if parent is not None
    }

    # Stated rather than sampled, because the hazard model chose which fault triggers
    # which and recomputing it would be generating a different earthquake. The jump
    # *points* are deliberately not carried across: the generator finds them from the
    # solved wavefront rather than by closest approach, so importing them would import an
    # answer to a question it asks itself, and asks differently.
    #
    # A single-segment rupture has no edges to state, and the generator refuses an empty
    # tree rather than accept a section that says nothing. It keeps its default.
    propagation = (
        {"propagation": PredeterminedPropagation(parents=parents)} if parents else {}
    )

    return RuptureConfig(
        # `per_fault` rather than `finite`: the hazard model already decided how the
        # moment divides between the faults, and deriving that division again from one
        # event magnitude would discard what it said.
        source=PerFaultSourceConfig.from_dict(
            srf_config.source
            | {
                "magnitudes": {
                    name: float(magnitude)
                    for name, magnitude in magnitudes.magnitudes.items()
                },
                "rakes": {name: float(rake) for name, rake in rakes.rakes.items()},
            }
        ),
        slip=SlipConfig.from_dict(srf_config.slip),
        **propagation,
        **common,
    )


@cli.from_docstring(app)
def generate_template(
    realisation_ffp: Path,
    geometry_output_path: Path,
    rupture_output_path: Path,
) -> None:
    """Generate rupture generator configuration files from a realisation file.

    Parameters
    ----------
    realisation_ffp : Path
        Path to the realisation file to read.
    geometry_output_path : Path
        Path to write the geometry configuration `rupture-generator mesh` reads.
    rupture_output_path : Path
        Path to write the rupture configuration `rupture-generator generate` reads.
    """
    metadata = RealisationMetadata.read_from_realisation(realisation_ffp)
    source_config = SourceConfig.read_from_realisation(realisation_ffp)
    rupture_propagation = RupturePropagationConfig.read_from_realisation(
        realisation_ffp
    )
    magnitudes = Magnitudes.read_from_realisation(realisation_ffp)
    rakes = Rakes.read_from_realisation(realisation_ffp)
    seeds = Seeds.read_from_realisation_or_random(realisation_ffp)
    srf_config = SRFConfig.read_from_realisation_or_defaults(
        realisation_ffp, metadata.defaults_version
    )
    velocity_model_1d = VelocityModel1D.read_from_realisation_or_defaults(
        realisation_ffp, metadata.defaults_version
    )
    rupture_velocity = RuptureVelocity.read_from_realisation_or_defaults(
        realisation_ffp, metadata.defaults_version
    )

    rupture_config = build_rupture(
        source_config,
        srf_config,
        rupture_propagation,
        magnitudes,
        rakes,
        velocity_model_1d,
        rupture_velocity,
        seeds,
        metadata,
    )
    geometry_config = build_geometry(source_config, srf_config, metadata)

    # `to_dict` rather than the `to_toml` mixin. Both configs hold discriminated unions
    # -- surfaces, the source, the propagation -- and only `to_dict` dispatches on the
    # object's own type; the TOML mixin serialises against the *declared* base and writes
    # every subtype out as an empty table.
    geometry_output_path.write_text(tomli_w.dumps(geometry_config.to_dict()))
    rupture_output_path.write_text(tomli_w.dumps(rupture_config.to_dict()))

    realisations.append_log_entry(realisation_ffp)
