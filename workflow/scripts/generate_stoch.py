import subprocess
from pathlib import Path
from typing import Annotated

import typer

from workflow.realisations import HFConfig, RealisationMetadata


def generate_stoch(
    realisation_ffp: Annotated[
        Path, typer.Argument("Path to realisation", exists=True, dir_okay=False)
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
    """Generate stoch file from SRF."""
    metadata = RealisationMetadata.read_from_realisation(realisation_ffp)
    hf_config = HFConfig.read_from_realisation_or_defaults(
        realisation_ffp, metadata.defaults_version
    )

    subprocess.check_call(
        [
            str(srf2stoch_path),
            f"dx={hf_config.stoch_dx}",
            f"dy={hf_config.stoch_dy}",
            f"infile={srf_ffp}",
            f"outfile={stoch_ffp}",
        ]
    )


def main():
    typer.run(generate_stoch)


if __name__ == "__main__":
    main()
