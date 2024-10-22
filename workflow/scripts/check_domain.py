import os
from pathlib import Path
from typing import Annotated

import numpy as np
import shapely
import typer

from source_modelling import srf
from workflow import log_utils
from workflow.realisations import DomainParameters

app = typer.Typer()


@app.command(help="Check an SRF's output for viability.")
@log_utils.log_call()
def check_domain(
    realisation_ffp: Annotated[
        Path, typer.Argument(help="The path to the realisation.")
    ],
    srf_ffp: Annotated[Path, typer.Argument(help="The path to the SRF file to check.")],
    velocity_model: Annotated[
        Path, typer.Argument(help="The path to the velocity model.")
    ],
):
    """Check an SRF's contents for viability.

    Parameters
    ----------
    realisation_ffp : Path
        The path to the realisation for the SRF.
    srf_ffp : Path
        The path to the SRF.

    Raises
    ------
    typer.Exit
        If any of the checks fail.
    """
    srf_file = srf.read_srf(srf_ffp)
    srf_geometry = srf_file.geometry

    domain = DomainParameters.read_from_realisation(realisation_ffp)
    logger = log_utils.get_logger(__name__)

    if not shapely.contains_properly(domain.domain.polygon, srf_geometry):
        logger.error("SRF is outside the boundary of the domain")
        raise typer.Exit(code=1)

    hausdorf_distance = shapely.hausdorff_distance(srf_geometry, domain.domain.polygon)
    if hausdorf_distance < 1000:
        logger.warning(
            log_utils.structured_log(
                "SRF geometry is close the edge of the domain. This could indicate a patch outside the domain, but may not be an error in its own right.",
                closest_distance=hausdorf_distance,
            ),
        )
        raise typer.Exit(code=1)

    velocity_model_estimated_size = 4 * domain.nx * domain.ny * domain.nz
    if any(
        (velocity_model / velocity_model_component).stat().st_size
        != velocity_model_estimated_size
        for velocity_model_component in [
            "rho3dfile.d",
            "vp3dfile.p",
            "vs3dfile.s",
            "in_basin_mask.b",
        ]
    ):
        logger.error(
            log_utils.structured_log(
                "The expected velocity model size does not match the computed velocity model size",
                expected_size=velocity_model_estimated_size,
            ),
        )
        raise typer.Exit(code=1)
