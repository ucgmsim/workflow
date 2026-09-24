#!/usr/bin/env python3
"""NZCVM Velocity Model Configuration.

Description
-----------
Generate an NZCVM velocity model configuration from a realisation. The
configuration describes both the *grid* the model is sampled onto and the
*layers* that are queried to fill it. The layers are taken from the
realisation's `nzcvm` section, and the grid is chosen with ``--format``:

- ``--format sw4`` writes an :class:`~nzcvm.config.grids.sw4.SW4GridConfig`,
  a mesh-refined grid written out as an sfile.
- ``--format emod3d`` writes an :class:`~nzcvm.config.grids.emod3d.EMOD3DGrid`,
  a uniform grid written out as EMOD3D binary files.

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
from nzcvm.coordinates import Coordinate

from qcore import cli
from workflow import sw4
from workflow.realisations import (
    DomainParameters,
    NZCVMSettings,
    RealisationMetadata,
    Refinements,
    Resolution,
    SW4Parameters,
    VelocityModelParameters,
)

app = typer.Typer()

NZTM_EPSG = 2193

SW4_DEPTH_OFFSET_KM = 10.0
"""Extra depth (km) modelled below the domain so SW4 refinement adjustment has room."""

SW4_MODEL_SLACK_GRIDPOINTS = 4
"""Gridpoints of velocity model kept beyond the padded SW4 grid, per face."""

EMOD3D_FREE_SURFACE_PADDING = 1
"""Extra gridpoints in z for EMOD3D's free surface shift."""


class GridFormat(StrEnum):
    """The simulator whose grid the velocity model is sampled onto."""

    SW4 = auto()
    EMOD3D = auto()


def _sw4_grid(
    domain_parameters: DomainParameters,
    refinements: Refinements,
    sw4_params: SW4Parameters,
    nzcvm_settings: NZCVMSettings,
) -> SW4GridConfig:
    """Build the SW4 mesh-refined grid configuration.

    The model is deliberately larger than the domain. `create-sw4-input` pads the
    SW4 grid by one supergrid sponge width on every face, so the model has to be
    padded by at least as much or SW4 queries outside the sfile."""
    domain = domain_parameters.domain
    domain_refinements = refinements.refinements_for_depth(
        domain_parameters.depth + SW4_DEPTH_OFFSET_KM
    )

    # NOTE: The sponge width must be measured on the refinements *SW4* will use,
    # i.e. resolved against the bare domain depth. Resolving against
    # `depth + SW4_DEPTH_OFFSET_KM` can land on a coarser bottom layer than SW4
    # actually gets (a 400 m bottom layer gives a 12 km sponge where SW4's 200 m
    # one gives 6 km), which would overstate the padding needed here and, worse,
    # disagree with `create-sw4-input`.
    coarsest_resolution = sw4.coarsest_resolution(refinements, domain_parameters.depth)
    supergrid_width = sw4.supergrid_width(sw4_params, coarsest_resolution)
    model_padding = supergrid_width + SW4_MODEL_SLACK_GRIDPOINTS * coarsest_resolution

    domain_refinements[-1].bottom += model_padding

    padding_km = model_padding / 1000.0
    domain = domain.pad(pad_x=(padding_km, padding_km), pad_y=(padding_km, padding_km))

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


def _emod3d_grid(
    domain_parameters: DomainParameters,
    resolution: Resolution,
    velocity_model_parameters: VelocityModelParameters,
    nzcvm_settings: NZCVMSettings,
) -> EMOD3DGrid:
    """Build the EMOD3D uniform grid configuration."""
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
    """
    metadata = RealisationMetadata.read_from_realisation(realisation_ffp)
    domain_parameters = DomainParameters.read_from_realisation(realisation_ffp)
    nzcvm_settings = NZCVMSettings.read_from_realisation_or_defaults(
        realisation_ffp, metadata.defaults_version
    )

    if not nzcvm_settings.layers:
        raise ValueError("NZCVM requires at least one defined layer.")

    match format:
        case GridFormat.SW4:
            grid = _sw4_grid(
                domain_parameters,
                Refinements.read_from_realisation_or_defaults(
                    realisation_ffp, metadata.defaults_version
                ),
                SW4Parameters.read_from_realisation_or_defaults(
                    realisation_ffp, metadata.defaults_version
                ),
                nzcvm_settings,
            )
        case GridFormat.EMOD3D:
            grid = _emod3d_grid(
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
        layers=nzcvm_settings.layers,
    )

    output_path.write_text(
        # This should be be a string but `to_json` is not smart enough to
        # realise that. The `str` is a free no-op that convinces the type
        # checker that the input is, in fact, a string.
        str(config.to_json(encoder=functools.partial(json.dumps, indent=4)))
    )
