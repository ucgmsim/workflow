"""Check Domain Script.

Description
-----------
This script checks the viability of an SRF's contents against a given domain and velocity model.

Inputs
------
1. A realisation file containing domain parameters,
2. An SRF file,
3. A velocity model.

Outputs
-------
No direct outputs. Logs errors and warnings if the SRF is outside the domain boundary or if the velocity model size does not match the expected size.

Environment
-----------
Can be run in the cybershake container. Can also be run from your own computer using the `check-domain` command which is installed after running `pip install workflow@git+https://github.com/ucgmsim/workflow`.

Usage
-----
`check-domain [OPTIONS] REALISATION_FFP SRF_FILE_FFP VELOCITY_MODEL_FFP`

For More Help
-------------
See the output of `check-domain --help`.
"""
from pathlib import Path
from typing import Annotated

import shapely
import typer

from qcore import cli
from source_modelling import srf
from workflow import log_utils
from workflow.realisations import DomainParameters

app = typer.Typer()


@cli.from_docstring(app)
@log_utils.log_call()
def check_domain(
    realisation_ffp: Annotated[Path, typer.Argument()],
    srf_ffp: Annotated[Path, typer.Argument()],
    velocity_model: Annotated[Path, typer.Argument()],
):
    """Check an SRF's contents for viability.

    Parameters
    ----------
    realisation_ffp : Path
        The path to the realisation.
    srf_ffp : Path
        The path to the SRF file to check.
    velocity_model : Path
        The path to the velocity model.

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
