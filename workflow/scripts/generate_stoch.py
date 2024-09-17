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

import subprocess
from pathlib import Path
from typing import Annotated

import typer

from workflow import log_utils
from workflow.realisations import HFConfig, RealisationMetadata

app = typer.Typer()


@app.command(help="Generate a stoch file from an SRF file.")
@log_utils.log_call
def generate_stoch(
    realisation_ffp: Annotated[
        Path, typer.Argument(help="Path to realisation", exists=True, dir_okay=False)
    ],
    srf_ffp: Annotated[
        Path, typer.Argument(help="Path to SRF.", exists=True, dir_okay=False)
    ],
    stoch_ffp: Annotated[
        Path, typer.Argument(help="Output Stoch filepath.", dir_okay=False)
    ],
    srf2stoch_path: Annotated[
        Path, typer.Option(exists=True, help="Path to srf2stoch binary")
    ] = Path("/EMOD3D/tools/srf2stoch"),
):
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
    hf_config = HFConfig.read_from_realisation_or_defaults(
        realisation_ffp, metadata.defaults_version
    )

    log_utils.log_check_call(
        [
            str(srf2stoch_path),
            f"dx={hf_config.stoch_dx}",
            f"dy={hf_config.stoch_dy}",
            f"infile={srf_ffp}",
            f"outfile={stoch_ffp}",
        ]
    )
