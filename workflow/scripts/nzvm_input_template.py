#!/usr/bin/env python3
"""NZCVM Velocity Model Configuration.

Description
-----------
Generate an NZCVM velocity model configuration from a realisation. The
configuration describes both the *grid* the model is sampled onto and the
*layers* that are queried to fill it, so one realisation can produce several
distinct velocity models:

- ``--format sw4`` writes an :class:`~nzcvm.config.grids.sw4.SW4GridConfig`,
  a mesh-refined grid written out as an sfile.
- ``--format emod3d`` writes an :class:`~nzcvm.config.grids.emod3d.EMOD3DGrid`,
  a uniform grid written out as EMOD3D binary files.
- ``--layers full`` keeps the realisation's whole layer stack (basins,
  coastline, offshore, Ely GTL taper, ...).
- ``--layers tomography`` keeps only the background tomography query and the
  numerical clamps, which is the reference "no bells and whistles" model.

Inputs
------
1. A realisation with domain parameters, resolution, and an `nzcvm` section.

Outputs
-------
1. An NZCVM velocity model configuration (JSON), ready for `nzcvm generate`.

Environment
-----------
Can be run in the cybershake container. Can also be run from your own computer
using the `create-nzvm-input` command after
`pip install workflow@git+https://github.com/ucgmsim/workflow`.

Usage
-----
`create-nzvm-input [OPTIONS] REALISATION_FFP OUTPUT_PATH`

For More Help
-------------
See the output of `create-nzvm-input --help`.
"""

import dataclasses
import functools
import json
from enum import StrEnum, auto
from pathlib import Path
from typing import Annotated

import pyproj
import typer
from nzcvm.config import VelocityModelConfig
from nzcvm.config.grids.emod3d import EMOD3DGrid, TopographyType
from nzcvm.config.grids.model import Model
from nzcvm.config.grids.sw4 import MeshRefinement, SW4GridConfig
from nzcvm.config.layers.core import LayerConfig
from nzcvm.config.layers.query import QueryLayerConfig
from nzcvm.coordinates import Coordinate

from qcore import cli
from workflow.realisations import (
    DomainParameters,
    NZCVMSettings,
    RealisationMetadata,
    Refinements,
    Resolution,
    VelocityModelParameters,
)

app = typer.Typer()

NZTM_EPSG = 2193
"""The CRS every NZCVM grid is built in."""

SW4_DEPTH_OFFSET_KM = 10.0
"""Extra depth (km) modelled below the domain so SW4 refinement adjustment has room."""

SW4_SUPERGRID_PADDING = 50
"""Gridpoints of velocity model kept below the bottom refinement for the supergrid sponge.

Per the SW4 User Guide the supergrid sponge (30 gridpoints) at the bottom of
the domain must be contained in the bottom refinement, so the velocity model
has to have values there. `create-sw4-input` pads the *domain* by
`sw4.supergrid_padding`; this pads the *model* by more than that so the
padded domain always lands inside it.
"""

EMOD3D_FREE_SURFACE_PADDING = 1
"""Extra gridpoints in z for EMOD3D's free surface shift.

EMOD3D shifts the model down one gridpoint for the free surface and so never
reads the deepest layer (see `genmodel.c`), which means the model must carry
one row more than the domain has. `create-e3d-par` assumes the same padding
when it checks the velocity model's size, so the two must agree.
"""

TOMOGRAPHY_LAYER_TYPES = frozenset({"clamp", "query"})
"""Layer types kept by `--layers tomography`.

The tomography-only model is the background 3D tomography and nothing else:
no basins, no offshore model, no Ely near-surface taper. The clamps stay
because they are numerical guards on the simulation (minimum Vs, Vp/Vs
ratio bounds), not a feature of the geology.
"""

DEFAULT_TOMOGRAPHY_GLOB = "ep2020.zarr"
"""Model glob matching only the tomography mesh in the NZCVM model directory."""


class GridFormat(StrEnum):
    """The simulator whose grid the velocity model is sampled onto."""

    SW4 = auto()
    EMOD3D = auto()


class LayerSelection(StrEnum):
    """How much of the realisation's NZCVM layer stack to keep."""

    FULL = auto()
    """Every layer the realisation configures."""
    TOMOGRAPHY = auto()
    """Background tomography and numerical clamps only."""


def select_layers(
    layers: list[LayerConfig],
    selection: LayerSelection,
    tomography_glob: str,
) -> list[LayerConfig]:
    """Filter a realisation's layer stack down to `selection`.

    Parameters
    ----------
    layers : list[LayerConfig]
        The realisation's configured layers, in pipeline order.
    selection : LayerSelection
        Which layers to keep.
    tomography_glob : str
        Glob matching the tomography mesh, used to narrow the query layer when
        `selection` is `LayerSelection.TOMOGRAPHY`.

    Returns
    -------
    list[LayerConfig]
        The kept layers, in their original order.

    Raises
    ------
    ValueError
        If the selection leaves no query layer, since a velocity model with
        nothing to query is always empty.
    """
    if selection == LayerSelection.FULL:
        return layers

    selected = []
    for layer in layers:
        if getattr(layer, "type", None) not in TOMOGRAPHY_LAYER_TYPES:
            continue
        if isinstance(layer, QueryLayerConfig):
            layer = dataclasses.replace(layer, model_globs=[tomography_glob])
        selected.append(layer)

    if not any(isinstance(layer, QueryLayerConfig) for layer in selected):
        raise ValueError(
            "Layer selection left no query layer: the realisation's nzcvm "
            "section must configure one for a velocity model to be generated."
        )
    return selected


def sw4_grid(
    domain_parameters: DomainParameters,
    refinements: Refinements,
    nzcvm_settings: NZCVMSettings,
) -> SW4GridConfig:
    """Build the SW4 mesh-refined grid configuration.

    Parameters
    ----------
    domain_parameters : DomainParameters
        The domain to model.
    refinements : Refinements
        The theoretical mesh refinements, resolved against the domain depth.
    nzcvm_settings : NZCVMSettings
        Supplies the topographic surface and chunking.

    Returns
    -------
    SW4GridConfig
        The grid configuration.
    """
    domain = domain_parameters.domain
    domain_refinements = refinements.refinements_for_depth(
        domain_parameters.depth + SW4_DEPTH_OFFSET_KM
    )
    domain_refinements[-1].bottom += (
        SW4_SUPERGRID_PADDING * domain_refinements[-1].resolution
    )

    origin_lat, origin_lon = domain.origin
    return SW4GridConfig(
        extent_x=domain.extent_x * 1000.0,
        extent_y=domain.extent_y * 1000.0,
        orientation=Model(
            origin_lon=origin_lon,
            origin_lat=origin_lat,
            crs=pyproj.CRS(NZTM_EPSG),
            azimuth=domain.great_circle_bearing,
        ),
        surface=nzcvm_settings.surface,
        chunks=nzcvm_settings.chunks,
        refinements={
            f"layer_{refinement.resolution}m": MeshRefinement(
                resolution=refinement.resolution, bottom=refinement.bottom
            )
            for refinement in domain_refinements
        },
    )


def emod3d_grid(
    domain_parameters: DomainParameters,
    resolution: Resolution,
    velocity_model_parameters: VelocityModelParameters,
    nzcvm_settings: NZCVMSettings,
) -> EMOD3DGrid:
    """Build the EMOD3D uniform grid configuration.

    Parameters
    ----------
    domain_parameters : DomainParameters
        The domain to model.
    resolution : Resolution
        The uniform grid spacing, in kilometres.
    velocity_model_parameters : VelocityModelParameters
        Supplies the topography type.
    nzcvm_settings : NZCVMSettings
        Supplies the topographic surface and chunking.

    Returns
    -------
    EMOD3DGrid
        The grid configuration, sized to match what `create-e3d-par` expects.
    """
    domain = domain_parameters.domain
    origin_lat, origin_lon = domain.origin
    return EMOD3DGrid(
        surface=nzcvm_settings.surface,
        nx=domain_parameters.nx(resolution.resolution),
        ny=domain_parameters.ny(resolution.resolution),
        nz=domain_parameters.nz(resolution.resolution) + EMOD3D_FREE_SURFACE_PADDING,
        # NOTE: NZCVM works in metres, the realisation's resolution in kilometres.
        resolution=resolution.resolution * 1000.0,
        orientation=Model(
            origin_lon=origin_lon,
            origin_lat=origin_lat,
            crs=pyproj.CRS(NZTM_EPSG),
            azimuth=domain.great_circle_bearing,
        ),
        topo_type=TopographyType(velocity_model_parameters.topo_type.lower()),
        # The EMOD3D grid is only ever chunked horizontally: depth is a single
        # chunk by construction, so a k chunk size would be silently ignored.
        chunks={
            coordinate: size
            for coordinate, size in nzcvm_settings.chunks.items()
            if coordinate in (Coordinate.I, Coordinate.J)
        },
    )


@cli.from_docstring(app)
def generate_template(
    realisation_ffp: Path,
    output_path: Path,
    format: Annotated[GridFormat, typer.Option()] = GridFormat.SW4,
    layers: Annotated[LayerSelection, typer.Option()] = LayerSelection.FULL,
    tomography_glob: Annotated[str, typer.Option()] = DEFAULT_TOMOGRAPHY_GLOB,
) -> None:
    """Generate an NZCVM velocity model configuration from a realisation file.

    Parameters
    ----------
    realisation_ffp : Path
        Path to the realisation file containing domain parameters.
    output_path : Path
        Path where the generated configuration will be written.
    format : GridFormat
        The simulator whose grid to sample the velocity model onto.
    layers : LayerSelection
        How much of the realisation's layer stack to keep. `tomography` drops
        the basins, offshore model and near-surface taper.
    tomography_glob : str
        Glob matching the tomography mesh, used when `layers` is `tomography`.
    """
    metadata = RealisationMetadata.read_from_realisation(realisation_ffp)
    domain_parameters = DomainParameters.read_from_realisation(realisation_ffp)
    nzcvm_settings = NZCVMSettings.read_from_realisation(realisation_ffp)

    if not nzcvm_settings.layers:
        raise ValueError("NZCVM requires at least one defined layer.")

    match format:
        case GridFormat.SW4:
            grid = sw4_grid(
                domain_parameters,
                Refinements.read_from_realisation_or_defaults(
                    realisation_ffp, metadata.defaults_version
                ),
                nzcvm_settings,
            )
        case GridFormat.EMOD3D:
            grid = emod3d_grid(
                domain_parameters,
                Resolution.read_from_realisation_or_defaults(
                    realisation_ffp, metadata.defaults_version
                ),
                VelocityModelParameters.read_from_realisation_or_defaults(
                    realisation_ffp, metadata.defaults_version
                ),
                nzcvm_settings,
            )

    config = VelocityModelConfig(
        grid=grid,
        layers=select_layers(nzcvm_settings.layers, layers, tomography_glob),
    )

    output_path.write_text(
        config.to_json(encoder=functools.partial(json.dumps, indent=4))  # ty: ignore
    )
