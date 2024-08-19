#!/usr/bin/env python
"""
Simulates high frequency seismograms for stations.
"""

import subprocess
from pathlib import Path
from typing import Annotated, Any, Iterable

import numpy as np
import typer

from workflow.realisations import DomainParameters, HFConfig, RealisationMetadata

HEAD_STAT = 24
FLOAT_SIZE = 4


def format_hf_input(input_lines: Iterable[Any]) -> str:
    return "\n".join(str(line) for line in input_lines)


def run_hf(
    realisation_ffp: Annotated[Path, typer.Argument(help="Path to realisation file.")],
    stoch_ffp: Annotated[
        Path,
        typer.Argument(help="Input stoch file.", exists=True),
    ],
    station_file: Annotated[
        Path, typer.Argument(help="Location of station file.", exists=True)
    ],
    out_file: Annotated[
        Path, typer.Argument(help="Filepath for HF output.", file_okay=False)
    ],
    hf_sim_path: Annotated[Path, typer.Option(help="Path to HF sim binary")] = Path(
        "/EMOD3D/tools/hf"
    ),
    velocity_model: Annotated[
        Path,
        typer.Option(
            help="Path to velocity model (1D). Ignored if --site-specific is set"
        ),
    ] = Path("/Cant1D_v2-midQ_leer.1d"),
):
    domain_parameters = DomainParameters.read_from_realisation(realisation_ffp)
    metadata = RealisationMetadata.read_from_realisation(realisation_ffp)
    hf_config = HFConfig.read_from_realisation_or_defaults(
        realisation_ffp, metadata.defaults_version
    )

    with open(station_file, "r") as station_file_handle:
        number_of_stations = len(station_file_handle.readlines())
    hf_sim_input = format_hf_input(
        [
            "",
            hf_config.sdrop,
            station_file,
            out_file,
            f"{len(hf_config.rayset)} {' '.join(str(ray) for ray in hf_config.rayset)}",
            int(not hf_config.no_siteamp),
            f"{hf_config.nbu} {hf_config.ift} {hf_config.flo} {hf_config.fhi}",
            hf_config.seed,
            1,
            f"{domain_parameters.duration} {hf_config.dt} {hf_config.fmax} {hf_config.kappa} {hf_config.qfexp}",
            f"{hf_config.rvfac} {hf_config.rvfac_shal} {hf_config.rvfac_deep} {hf_config.czero} {hf_config.calpha}",
            f"{hf_config.mom or -1} {hf_config.rupv or -1}",
            stoch_ffp,
            velocity_model,
            hf_config.vs_moho,
            f"{hf_config.nl_skip} {hf_config.vp_sig} {hf_config.vsh_sig} {hf_config.rho_sig} {hf_config.qs_sig} {int(hf_config.ic_flag)}",
            hf_config.velocity_name,
            f"{hf_config.fa_sig1} {hf_config.fa_sig2} {hf_config.rv_sig1}",
            hf_config.path_dur,
            f"{hf_config.stress_parameter_adjustment_fault_area or -1} "
            f"{hf_config.stress_parameter_adjustment_target_magnitude or -1} "
            f"{hf_config.stress_parameter_adjustment_tect_type or -1}",
            0,
            "",
        ]
    )
    with open(velocity_model, "r") as f:
        f.readline()
        vs = np.float32(float(f.readline().split()[2]) * 1000.0)

    # We run binary with subprocess.run rather than subprocess.check_call
    # because subprocess.run accepts the input argument.
    output = subprocess.run(
        str(hf_sim_path),
        input=hf_sim_input,
        check=True,
        text=True,
        stderr=subprocess.PIPE,
    )
    # for some reason, e_dist values are not part of the stdout
    e_dist = np.fromstring(output.stderr, dtype="f4", sep="\n")
    if e_dist.size != number_of_stations:
        print(f"Expected {number_of_stations} e_dist values, got {e_dist.size}")
        print(output.stderr)
        exit(1)

    # write e_dist and vs to file
    with open(out_file, "r+b") as out:
        for i in range(number_of_stations):
            out.seek(HEAD_STAT - 2 * FLOAT_SIZE, 1)
            e_dist[i].tofile(out)
            vs.tofile(out)


def main():
    typer.run(run_hf)


if __name__ == "__main__":
    main()
