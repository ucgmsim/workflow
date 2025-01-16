"""Convert a legacy realisation into the modern JSON format."""

from pathlib import Path
from typing import Annotated

import numpy as np
import pandas as pd
import typer

from source_modelling.sources import Fault, Plane
from workflow.defaults import DefaultsVersion
from workflow.realisations import (
    RealisationMetadata,
    RupturePropagationConfig,
    SourceConfig,
)

app = typer.Typer()


@app.command(help="Convert a legacy realisation into the modern JSON format.")
def convert_realisation(
    old_realisation_path: Annotated[
        Path,
        typer.Argument(
            help="Path to the old realisation CSV.",
            exists=True,
            readable=True,
            dir_okay=False,
        ),
    ],
    realisation_path: Annotated[
        Path,
        typer.Argument(
            help="Path to output converted realisation.", writable=True, dir_okay=False
        ),
    ],
    defaults_version: Annotated[
        DefaultsVersion,
        typer.Argument(help="Defaults version to use for the new realisation."),
    ],
) -> None:
    """Convert a realisation from the old CSV format to the new JSON format.

    Parameters
    ----------
    old_realisation_path : Path
        Path to the realisation CSV.
    realisation_path : Path
        Output path for the new JSON fomat.
    defaults_version :
        Defaults version to use.
    """
    old_realisation = pd.read_csv(old_realisation_path).iloc[0]
    name = old_realisation["name"]
    metadata = RealisationMetadata(
        name=name, version="1", defaults_version=defaults_version
    )
    planes: list[Plane] = []
    dip = old_realisation["dip"]
    dip_dir = old_realisation["dip_dir"]
    for i in range(old_realisation["plane_count"]):
        centroid = np.array(
            [
                old_realisation[f"clat_subfault_{i}"],
                old_realisation[f"clon_subfault_{i}"],
            ]
        )
        strike = old_realisation[f"strike_subfault_{i}"]
        dtop = old_realisation[f"dtop_subfault_{i}"]
        length = old_realisation[f"length_subfault_{i}"]
        width = old_realisation[f"width_subfault_{i}"]
        planes.append(
            Plane.from_centroid_strike_dip(
                centroid, dip, length, width, dtop=dtop, strike=strike, dip_dir=dip_dir
            )
        )
    fault = Fault(planes)
    shypo = old_realisation["shypo"] / fault.length + 1 / 2
    dhypo = old_realisation["dhypo"] / fault.width

    sources = SourceConfig(source_geometries={name: fault})
    rupture_propagation_config = RupturePropagationConfig(
        rupture_causality_tree={name: None},  # Trivial rupture propagation tree
        jump_points={},  # no jump points
        rakes={name: old_realisation["rake"]},
        magnitudes={name: old_realisation["magnitude"]},
        hypocentre=np.array([shypo, dhypo]),
    )
    for config in [metadata, sources, rupture_propagation_config]:
        config.write_to_realisation(realisation_path)
