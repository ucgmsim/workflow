"""Write Model Coordinates.

Description
-----------
Write out model parameters for EMOD3D.

Inputs
------
1. A realisation file containing domain parameters.

Outputs
-------
1. A model parameters file describing the location of the domain in latitude, longitude,
2. A grid parameters file describing the discretisation of the domain.

Environment
-----------
Can be run in the cybershake container. Can also be run from your own computer using the `generate-model-coordinates` command which is installed after running `pip install workflow@git+https://github.com/ucgmsim/workflow`.

Usage
-----
`generate-station-coordinates [OPTIONS] REALISATIONS_FFP OUTPUT_PATH`

For More Help
-------------
See the output of `generate-model-coordinates --help`.
"""

from pathlib import Path
from typing import Annotated

import typer

from qcore import coordinates
from workflow.realisations import DomainParameters

app = typer.Typer()


@app.command(help="Generate model coordinate files for EMOD3D from a realisation file")
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
            writable=True,
        ),
    ],
) -> None:
    """
    Generate model coordinate files for EMOD3D from a realisation JSON file.

    This function reads domain parameters from a realisation and generates
    two output files: one containing grid file specifications and the other containing
    model parameters. These files are used for EMOD3D simulations.

    Parameters
    ----------
    realisation_ffp : Path
        Path to the realisation file.
    output_ffp : Path
        Path to the directory where the output model coordinate files will be saved.
        The directory will be created if it does not exist.

    Returns
    -------
    None
        This function does not return a value. It writes two files to the specified output directoy:
        - `grid_file` containing the grid dimensions and resolution.
        - `model_params` containing the model origin coordinates, shifts, corners, and dimensions.
    """
    output_ffp.mkdir(exist_ok=True)
    domain_parameters = DomainParameters.read_from_realisation(realisation_ffp)
    x_len = domain_parameters.domain.extent_x
    y_len = domain_parameters.domain.extent_y
    z_len = domain_parameters.depth
    resolution = domain_parameters.resolution
    grid_file = output_ffp / "grid_file"
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
    x_shift = -(domain_parameters.domain.extent_x - domain_parameters.resolution) / 2
    y_shift = -(domain_parameters.domain.extent_y - domain_parameters.resolution) / 2
    model_params.write_text(
        "\n".join(
            [
                "Model origin coordinates:",
                f" lon= {model_origin[1]:10.5f} lat= {model_origin[0]:10.5f}"
                f" rotate= {domain_parameters.domain.great_circle_bearing:7.2f}",
                "",
                "Model origin shift (cartesian vs. geographic):",
                f" xshift(km)= {x_shift:12.5f} yshift(km)= {y_shift:12.5f}",
                "",
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
