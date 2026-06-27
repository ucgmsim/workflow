#!/usr/bin/env python3
import functools
import json
from pathlib import Path

import pyproj
import typer
from nzcvm.config import VelocityModelConfig
from nzcvm.config.grids.model import Model
from nzcvm.config.grids.sw4 import MeshRefinement, SW4GridConfig
from nzcvm.config.metadata import ModelMetadata

from qcore import cli
from workflow import domain
from workflow.realisations import DomainParameters, VelocityModelParameters

app = typer.Typer()



@cli.from_docstring(app)
def generate_template(realisation_ffp: Path, output_path: Path) -> None:
    """Generate a template VM file from a realisation file.

    Parameters
    ----------
    realisation_ffp : Path
        Path to the realisation file containing domain parameters.
    output_path : Path
        Path where the generated template will be written.
    """
    domain_parameters = DomainParameters.read_from_realisation(realisation_ffp)
    velocity_model_parameters = VelocityModelParameters.read_from_realisation(realisation_ffp)
    offset = 10.0
    refinements = domain.domain_refinements(domain_parameters.depth + offset)
    origin = domain_parameters.domain.origin
    origin_lat = origin[0]
    origin_lon = origin[1]
    azimuth = domain_parameters.domain.great_circle_bearing

    
    buffer = 1.10
    extent_y = buffer * domain_parameters.domain.extent_y * 1000.0
    extent_x = buffer * domain_parameters.domain.extent_x * 1000.0
    if not velocity_model_parameters.surface:
        raise ValueError('NZCVM requires defined surface path.')
    if not velocity_model_parameters.layers:
        raise ValueError('NZCVM requires at least one defined layer.')

    config = VelocityModelConfig(
        grid=SW4GridConfig(
            extent_x=extent_x,
            extent_y=extent_y,
            orientation=Model(
                origin_lon=origin_lon,
                origin_lat=origin_lat,
                crs=pyproj.CRS(2193),
                azimuth=azimuth
            ),
            surface=velocity_model_parameters.surface,
            chunks=velocity_model_parameters.chunks,
            refinements=
                {
                    f'layer_{refinement.resolution}m': MeshRefinement(resolution=refinement.resolution, bottom=refinement.bottom)
                 for refinement in refinements
                }
        ),
        layers=velocity_model_parameters.layers
    )
        

    with open(output_path, 'w') as f:
        f.write(config.to_json(encoder=functools.partial(json.dumps, indent=4))) # ty: ignore
