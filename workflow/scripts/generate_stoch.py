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
    first_segment = srf_data.header.iloc[0]
    dx = first_segment["len"] / first_segment["nstk"]
    dy = first_segment["wid"] / first_segment["ndip"]

    subprocess.check_call(
        [
            str(srf2stoch_path),
            f"dx={dx}",
            f"dy={dy}",
            f"infile={srf_ffp}",
            f"outfile={stoch_ffp}",
        ]
    )


def main():
    typer.run(generate_stoch)


if __name__ == "__main__":
    main()
