"""
Check SRF File.

Description
-----------
Check an SRF's contents for viability.

Inputs
------
1. realisation_ffp : Path
    The path to the realisation for the SRF.
2. srf_ffp : Path
    The path to the SRF file to check.

Outputs
-------
None

Environment
-----------
Can be run in the cybershake container. Can also be run from your own computer using the `check-srf` command which is installed after running `pip install workflow@git+https://github.com/ucgmsim/workflow`.

Usage
-----
`check-srf [OPTIONS] REALISATION_FFP SRF_FILE_FFP`

For More Help
-------------
See the output of `check-srf --help`.
"""

from pathlib import Path
from typing import Annotated

import numpy as np
import typer

from qcore import cli
from qcore.uncertainties import mag_scaling
from source_modelling import srf
from workflow import log_utils
from workflow.realisations import (
    RealisationMetadata,
    RealisationParseError,
    RupturePropagationConfig,
    VelocityModel1D,
)

app = typer.Typer()


@cli.from_docstring(app)
@log_utils.log_call()
def check_srf(
    realisation_ffp: Annotated[Path, typer.Argument()],
    srf_ffp: Annotated[Path, typer.Argument()],
):
    """Check an SRF's contents for viability.

    Parameters
    ----------
    realisation_ffp : Path
        The path to the realisation for the SRF.
    srf_ffp : Path
        The path to the SRF file to check.

    Raises
    ------
    typer.Exit
        If any of the checks fail.
    """
    metadata = RealisationMetadata.read_from_realisation(realisation_ffp)
    velocity_model = VelocityModel1D.read_from_realisation_or_defaults(
        realisation_ffp, metadata.defaults_version
    ).model
    velocity_model["mu"] = velocity_model["Vs"] ** 2 * velocity_model["rho"] * 1e10
    velocity_model["depth"] = velocity_model["thickness"].cumsum()

    srf_file = srf.read_srf(srf_ffp)
    indices = np.minimum(
        len(velocity_model["depth"]) - 1,
        np.searchsorted(velocity_model["depth"], srf_file.points["dep"]),
    )
    mu = velocity_model["mu"].iloc[indices].values
    srf_magnitude = mag_scaling.mom2mag(
        (srf_file.points["area"].values * srf_file.points["slip"].values * mu).sum()
    )
    logger = log_utils.get_logger("__name__")

    if srf_file.points.isnull().values.any():
        logger.error("SRF contains NaN values")
        raise typer.Exit(code=1)

    if (srf_file.points["dep"] < 0).any():
        logger.error(
            "Negative SRF depth detected", min_depth=srf_file.points["depth"].min()
        )
        raise typer.Exit(code=1)

    if not np.isclose(srf_file.points["tinit"].min(), 0):
        logger.warning(
            "SRF does not begin at zero (this may not be an error depending on SRF setup)",
            tinit=srf_file.points["tinit"].min(),
        )
        raise typer.Exit(code=1)

    if (srf_file.points["lat"].abs() > 90).any():
        logger.error(
            "SRF Latitude values out of bounds",
            lat_range=(srf_file.points["lat"].min(), srf_file.points["lat"].max()),
        )
        raise typer.Exit(code=1)

    if (srf_file.points["lon"].abs() > 180).any():
        logger.error(
            "SRF Longitude values out of bounds",
            lat_range=(srf_file.points["lon"].min(), srf_file.points["lon"].max()),
        )
        raise typer.Exit(code=1)

    try:
        _ = srf_file.planes
    except ValueError:
        logger.error("SRF geometry is broken.")
        raise typer.Exit(code=1)

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
                "Mismatch SRF magnitude",
                srf_magnitude=srf_magnitude,
                realisation_magnitude=magnitude,
            )
            raise typer.Exit(code=1)
        logger.info("SRF Magnitude", expected=magnitude, actual=srf_magnitude)
    except RealisationParseError:
        pass

    if srf_magnitude >= 11:
        logger.error("Implausible SRF magnitude", srf_magnitude=magnitude)
        raise typer.Exit(code=1)
