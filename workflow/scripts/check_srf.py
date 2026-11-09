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
Can be run in the cybershake container. Can also be run from your own computer using the `check-srf` command which is installed after running `pip install ucgmsim-workflow@git+https://github.com/ucgmsim/workflow`.

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
import pandas as pd
import typer

from qcore import cli
from source_modelling import moment, srf
from workflow import log_utils, realisations
from workflow.realisations import (
    Magnitudes,
    RealisationMetadata,
    RealisationParseError,
    VelocityModel1D,
)

app = typer.Typer()


def srf_rigidity(realisation_ffp: Path, points: pd.DataFrame) -> np.ndarray:
    """Return the shear modulus at each SRF point.

    A version 2.0 SRF records the shear-wave speed (``vs``, cm/s) and density
    (``den``, g/cm^3) its moment was computed with, so the rigidity comes from
    those columns. Genslip fills them from the realisation's 1D velocity model
    and ``generate-rupture`` from the NZCVM. A version 1.0 SRF has no such
    columns, so its rigidity falls back to the realisation's 1D velocity model.

    Parameters
    ----------
    realisation_ffp : Path
        The path to the realisation for the SRF.
    points : pd.DataFrame
        The SRF points, with a ``dep`` column (km) and, for a version 2.0 SRF,
        ``vs`` and ``den`` columns.

    Returns
    -------
    np.ndarray
        The shear modulus at each point (dyne/cm^2).
    """
    if {"vs", "den"} <= set(points.columns) and (
        (points["vs"] > 0) & (points["den"] > 0)
    ).all():
        return (points["vs"].to_numpy() ** 2) * points["den"].to_numpy()

    metadata = RealisationMetadata.read_from_realisation(realisation_ffp)
    velocity_model = VelocityModel1D.read_from_realisation_or_defaults(
        realisation_ffp, metadata.defaults_version
    ).model
    # Vs (km/s) -> cm/s is 1e5, so Vs^2 rho picks up 1e10 to be dyne/cm^2.
    mu = velocity_model["Vs"].to_numpy() ** 2 * velocity_model["rho"].to_numpy() * 1e10
    depth = velocity_model["thickness"].cumsum().to_numpy()
    indices = np.minimum(len(depth) - 1, np.searchsorted(depth, points["dep"]))
    return mu[indices]


@cli.from_docstring(app)
@log_utils.log_call()
def check_srf(
    realisation_ffp: Annotated[Path, typer.Argument()],
    srf_ffp: Annotated[Path, typer.Argument()],
) -> None:
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
    srf_file = srf.read_srf(srf_ffp)
    mu = srf_rigidity(realisation_ffp, srf_file.points)
    # SRF area (cm^2) x slip (cm) x mu (dyne/cm^2) is a moment in dyne-cm, but
    # `moment_to_magnitude` takes Nm.
    srf_moment_dyne_cm = (
        np.array(srf_file.points["area"].values)
        * np.array(srf_file.points["slip"].values)
        * mu
    ).sum()
    srf_magnitude = moment.moment_to_magnitude(
        moment.dyne_cm_to_newton_metre(srf_moment_dyne_cm), bold_m=True
    )
    logger = log_utils.get_logger("__name__")

    if srf_file.points.isnull().values.any():
        logger.error("SRF contains NaN values")
        raise typer.Exit(code=1)

    if (srf_file.points["dep"] < 0).any():
        logger.error(
            "Negative SRF depth detected", min_depth=srf_file.points["dep"].min()
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
        magnitudes = Magnitudes.read_from_realisation(realisation_ffp)
        magnitude = moment.moment_to_magnitude(
            sum(
                moment.magnitude_to_moment(mag, bold_m=True)
                for mag in magnitudes.magnitudes.values()
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
        logger.error("Implausible SRF magnitude", srf_magnitude=srf_magnitude)
        raise typer.Exit(code=1)
    realisations.append_log_entry(realisation_ffp)
