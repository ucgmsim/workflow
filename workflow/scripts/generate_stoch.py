"""Stoch Generation.

Description
-----------
Generate Stoch file for HF simulation. This file is just a down-sampled version of the SRF.

Inputs
------
A realisation file containing a metadata configuration, and a generated SRF file.

Outputs
-------
A [Stoch](https://wiki.canterbury.ac.nz/display/QuakeCore/File+Formats+Used+In+Ground+Motion+Simulation#FileFormatsUsedInGroundMotionSimulation-Stochformat) file containing a down-sampled version of the SRF.

Usage
-----
`generate-stoch [OPTIONS] REALISATION_FFP SRF_FFP STOCH_FFP`

Environment
-----------
Can be run in the cybershake container. Can also be run from your own computer using the `generate-stoch` command which is installed after running `pip install workflow@git+https://github.com/ucgmsim/workflow`. If you are executing on your own computer you also need to specify the `srf2stoch` path (`--srf2stoch-path`).

For More Help
-------------
See the output of `generate-stoch --help` or `workflow.scripts.generate_stoch`.
"""

from pathlib import Path
from typing import Annotated

import typer

from qcore import cli
from source_modelling import sources, srf
from workflow import log_utils, realisations
from workflow.realisations import RealisationMetadata, SourceConfig, StochConfig

app = typer.Typer()


@cli.from_docstring(app)
@log_utils.log_call()
def generate_stoch(
    realisation_ffp: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    srf_ffp: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    stoch_ffp: Annotated[Path, typer.Argument(dir_okay=False)],
    srf2stoch_path: Annotated[Path, typer.Option(exists=True)] = Path(
        "/EMOD3D/tools/srf2stoch"
    ),
) -> None:
    """Generate a stoch file from an SRF file.

    This function uses the `srf2stoch` binary to generate a stoch file from the provided SRF file.

    Parameters
    ----------
    realisation_ffp : Path
        Path to the realisation.
    srf_ffp : Path
        Path to the SRF file which is used as input for the stoch file generation.
    stoch_ffp : Path
        Path to the output file where the generated stoch file will be saved.
    srf2stoch_path : Path, optional
        Path to the `srf2stoch` binary used for the conversion.
    """
    metadata = RealisationMetadata.read_from_realisation(realisation_ffp)
    stoch_config = StochConfig.read_from_realisation_or_defaults(
        realisation_ffp, metadata.defaults_version
    )

    source_config = SourceConfig.read_from_realisation(realisation_ffp)

    if all(
        isinstance(fault, sources.Point)
        for fault in source_config.source_geometries.values()
    ):
        srf_file = srf.read_srf(srf_ffp)
        source = srf_file.header.iloc[0]
        srf_nstk = int(source["nstk"])
        srf_len = float(source["len"])
        dx = srf_len / srf_nstk
        srf_ndip = int(source["ndip"])
        srf_wid = float(source["wid"])
        dy = srf_wid / srf_ndip
    else:
        geometries = list(source_config.source_geometries.values())
        min_length = min(fault.length for fault in geometries)
        min_width = min(fault.width for fault in geometries)
        # If the stoch dx is greater than the length (resp. dy and width), we might get an empty stoch file
        dx = min(stoch_config.stoch_dx, min_length / 2)
        dy = min(stoch_config.stoch_dy, min_width / 2)

    log_utils.log_check_call(
        [
            str(srf2stoch_path),
            f"dx={dx}",
            f"dy={dy}",
            f"infile={srf_ffp}",
            f"outfile={stoch_ffp}",
        ]
    )
    realisations.append_log_entry(realisation_ffp)
