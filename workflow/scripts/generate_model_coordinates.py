"""
Generate model coordinates for EMOD3D for a single realisation.

This script outputs the model coordinates in three places:

1. A model coordinates file, containing the discretisation of the domain.
2. A model bounds file, containing the boundary of the model.
3. A model params file, containing metadata about the discretisation.
"""

import subprocess
from pathlib import Path
from typing import Annotated

import pandas as pd
import typer
from qcore import coordinates

from workflow import realisations
from workflow.realisations import DomainParameters


def generate_model_coordinates(
    realisation_ffp: Annotated[
        Path,
        typer.Argument(
            help="Path to realisation JSON file", exists=True, readable=True
        ),
    ],
    output_ffp: Annotated[
        Path,
        typer.Argument(
            help="Path to directory to output model coordinates",
            exists=True,
            writable=True,
        ),
    ],
    gen_model_coords_ffp: Annotated[
        Path,
        typer.Argument(
            help="Path to the gen model coords binary", exists=True, readable=True
        ),
    ] = Path("/EMOD3D/tools/gen_model_cords"),
) -> None:
    """Generate model coordinates for EMOD3D."""
    output_ffp.mkdir(exist_ok=True)
    domain_parameters: DomainParameters = realisations.read_config_from_realisation(
        DomainParameters, realisation_ffp
    )
    x_len = domain_parameters.domain.extent_x
    y_len = domain_parameters.domain.extent_y
    z_len = domain_parameters.depth
    resolution = domain_parameters.resolution
    grid_file = output_ffp / "gridfile"
    model_params = output_ffp / "model_params"
    grid_file.write_text(
        "\n".join(
            [
                f"xlen={x_len:.4f}",
                f"{0:10.4f} {x_len:10.4f} {resolution:13.6e}",
                f"ylen={y_len:.4f}",
                f"{0:10.4f} {y_len:10.4f} {resolution:13.6e}",
                f"zlen={z_len:.4f}",
                f"{0:10.4f} {z_len:10.4f} {resolution:13.6e}\n",
            ]
        )
    )
    model_origin = domain_parameters.domain.origin
    model_corners = coordinates.nztm_to_wgs_depth(domain_parameters.domain.corners)
    x_shift = (domain_parameters.extent_x - domain_parameters.resolution) / 2
    y_shift = (domain_parameters.extent_y - domain_parameters.resolution) / 2
    model_params.write_text(
        "\n".join(
            [
                "Model origin coordinates:",
                f" lon= {model_origin[1]:10.5f} lat= {model_origin[0]:10.5f} rotate= {domain_parameters.bearing:7.2f}",
                "",
                "Model origin shift (cartesian vs. geographic):",
                f" xshift(km)= {x_shift:12.5f} yshift(km)= {y_shift:12.5f}" "",
                "Model corners:",
            ]
            + [
                f" c{i + 1}= {corner[1]:10.5f} {corner[0]:10.5f}"
                for i, corner in enumerate(model_corners)
            ]
            + [
                "",
                "Model Dimensions:",
                f" xlen= {x_len:10.4f} km",
                f" ylen= {y_len:10.4f} km",
                f" zlen= {z_len:10.4f} km",
                "",
            ]
        )
    )


def main():
    typer.run(generate_model_coordinates)


if __name__ == "__main__":
    main()
