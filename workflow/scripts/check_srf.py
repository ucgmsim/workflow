from pathlib import Path
from typing import Annotated

import numpy as np
import typer

from qcore.uncertainties import mag_scaling
from source_modelling import moment, srf
from workflow import log_utils
from workflow.realisations import RealisationParseError, RupturePropagationConfig

app = typer.Typer()


@app.command(help="Check an SRF's output for viability.")
@log_utils.log_call()
def check_srf(
    realisation_ffp: Annotated[
        Path, typer.Argument(help="The path to the realisation for the SRF.")
    ],
    srf_ffp: Annotated[Path, typer.Argument(help="The path to the SRF file to check.")],
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
        If any of the checks fail.typer.Exit
    """
    srf_file = srf.read_srf(srf_ffp)
    srf_magnitude = moment.moment_to_magnitude(
        moment.MU
        * (srf_file.points["area"].sum() / 1e6)
        * srf_file.points["slip"].mean()
    )
    logger = log_utils.get_logger("__name__")
    try:
        rupture_prop_config = RupturePropagationConfig.read_from_realisation(
            realisation_ffp
        )
        magnitude = mag_scaling.mom2mag(
            sum(
                mag_scaling.mag2mom(mag)
                for mag in rupture_prop_config.magnitudes.values()
            )
        )
        if not np.isclose(srf_magnitude, magnitude, atol=5e-3):
            logger.error(
                log_utils.structured_log(
                    "Mismatch SRF magnitude",
                    srf_magnitude=srf_magnitude,
                    realisation_magnitude=magnitude,
                )
            )
            raise typer.Exit(code=1)
    except RealisationParseError:
        pass

    if srf_magnitude >= 11:
        logger.error(
            log_utils.structured_log(
                "Implausible SRF magnitude", srf_magnitude=magnitude
            )
        )
        raise typer.Exit(code=1)

    if (srf_file.points["dep"] < 0).any():
        logger.error(
            log_utils.structured_log(
                "Negative SRF depth detected", min_depth=srf_file.points["depth"].min()
            )
        )
        raise typer.Exit(code=1)

    if not np.isclose(srf_file.points["tinit"].min(), 0):
        logger.warning(
            log_utils.structured_log(
                "SRF does not begin at zero (this may not be an error depending on SRF setup)",
                tinit=srf_file.points["tinit"].min(),
            )
        )
        raise typer.Exit(code=1)
