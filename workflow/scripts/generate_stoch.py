import subprocess
from pathlib import Path
from typing import Annotated

import typer

from source_modelling import srf


def generate_stoch(
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
    "Generate stoch file from SRF."
    srf_data = srf.read_srf(srf_ffp)

    subprocess.check_call(
        [
            str(srf2stoch_path),
            "dx=2",
            "dy=2",
            f"infile={srf_ffp}",
            f"outfile={stoch_ffp}",
        ]
    )


def main():
    typer.run(generate_stoch)


if __name__ == "__main__":
    main()
