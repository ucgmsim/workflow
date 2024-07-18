import subprocess
from pathlib import Path
from typing import Annotated

import pandas as pd
import typer

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
    output_ffp.mkdir(exist_ok=True)
    domain_parameters: DomainParameters = realisations.read_config_from_realisation(
        DomainParameters, realisation_ffp
    )
    x_len = domain_parameters.domain.extent_x
    y_len = domain_parameters.domain.extent_y
    z_len = domain_parameters.depth
    resolution = domain_parameters.resolution
    grid_file = output_ffp / "gridfile"
    grid_out = output_ffp / "gridout"
    model_coords = output_ffp / "model_coords"
    model_params = output_ffp / "model_params"
    model_bounds = output_ffp / "model_bounds"
    grid_file.write_text(
        "\n".join(
            [
                f"xlen={x_len}",
                f"{0:10.4f} {x_len:10.4f} {resolution:13.6e}",
                f"ylen={y_len}",
                f"{0:10.4f} {y_len:10.4f} {resolution:13.6e}",
                f"zlen={z_len}",
                f"{0:10.4f} {z_len:10.4f} {resolution:13.6e}",
            ]
        )
    )
    model_origin = domain_parameters.domain.origin
    with model_params.open(mode="w") as model_params_out:
        subprocess.check_call(
            [
                str(gen_model_coords_ffp),
                "geoproj=1",
                f"gridfile={grid_file}",
                f"gridout={grid_out}",
                "centreorigin=1",
                "docoords=1",
                "nzout=1",
                f"name='{model_coords}'",
                "gzip=0",
                "latfirst=0",
                f"modellon={model_origin[1]}",
                f"modellat={model_origin[0]}",
                f"modelrot={domain_parameters.domain.bearing}",
            ],
            stdout=model_params_out,
        )

    model_coords_frame = pd.read_csv(
        model_coords, delimiter="\s+", header=None, names=["lon", "lat", "x", "y"]
    )
    model_boundary = model_coords_frame[
        model_coords_frame["x"] in {0, domain_parameters.nx - 1}
        or model_coords["y"] in {0, domain_parameters.ny - 1}
    ]
    model_boundary.to_csv(model_bounds, delimiter=" ", float_format=".6f")


def main():
    typer.run(generate_model_coordinates)


if __name__ == "__main__":
    main()
