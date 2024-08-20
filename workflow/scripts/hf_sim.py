#!/usr/bin/env python
"""
Simulates high frequency seismograms for stations.
"""

import functools
import multiprocessing
import shutil
import struct
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Annotated, Any, Iterable

import numpy as np
import pandas as pd
import tqdm
import typer

from workflow.realisations import DomainParameters, HFConfig, RealisationMetadata

HEAD_STAT = 24
FLOAT_SIZE = 4


def format_hf_input(input_lines: Iterable[Any]) -> str:
    return "\n".join(str(line) for line in input_lines)


def hf_simulate_station(
    hf_config: HFConfig,
    domain_parameters: DomainParameters,
    velocity_model: Path,
    stoch_ffp: Path,
    output_directory: Path,
    hf_sim_path: Path,
    longitude: float,
    latitude: float,
    name: str,
) -> None:
    raw_hf_output_ffp = output_directory / f"{name}.hf"
    with tempfile.NamedTemporaryFile(mode="w") as station_input_file:
        station_input_file.write(f"{longitude} {latitude} {name}\n")
        station_input_file.flush()
        hf_sim_input = format_hf_input(
            [
                "",
                hf_config.sdrop,
                station_input_file.name,
                raw_hf_output_ffp,
                f"{len(hf_config.rayset)} {' '.join(str(ray) for ray in hf_config.rayset)}",
                int(not hf_config.no_siteamp),
                f"{hf_config.nbu} {hf_config.ift} {hf_config.flo} {hf_config.fhi}",
                hf_config.seed,
                1,  # one station in the input
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
                0,  # maybe don't need this?
                f"{hf_config.stress_parameter_adjustment_fault_area or -1} "
                f"{hf_config.stress_parameter_adjustment_target_magnitude or -1} "
                f"{hf_config.stress_parameter_adjustment_tect_type or -1}",
                0,  # seek bytes to 0 (no binary offset for this output)
                "",
            ]
        )

        try:
            output = subprocess.run(
                str(hf_sim_path),
                input=hf_sim_input,
                check=True,
                text=True,
                stderr=subprocess.PIPE,
            )
        except subprocess.CalledProcessError as e:
            print(e.stderr)
            raise e

        epicentre_distance = np.fromstring(output.stderr, dtype="f4", sep="\n")
        if epicentre_distance.size != 1:
            raise ValueError(
                f"Expected exactly one epicentre_distance value, get {epicentre_distance.size}"
            )
        return epicentre_distance[0]


def hf_simulate_station_worker(*args):
    return hf_simulate_station(*args[:-1], *args[-1])


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
    work_directory: Annotated[
        Path,
        typer.Option(
            help="Path to work directory.", exists=True, writable=True, file_okay=False
        ),
    ] = Path("/out"),
):
    domain_parameters = DomainParameters.read_from_realisation(realisation_ffp)
    metadata = RealisationMetadata.read_from_realisation(realisation_ffp)
    hf_config = HFConfig.read_from_realisation_or_defaults(
        realisation_ffp, metadata.defaults_version
    )

    stations = pd.read_csv(
        station_file,
        delimiter=r"\s+",
        header=None,
        names=["longitude", "latitude", "name"],
    )
    mask = domain_parameters.domain.contains(
        stations[["latitude", "longitude"]].to_numpy()
    )
    stations = stations[mask]

    with multiprocessing.Pool() as pool:
        epicentre_distances = list(
            tqdm.tqdm(
                pool.imap(
                    functools.partial(
                        hf_simulate_station_worker,
                        hf_config,
                        domain_parameters,
                        velocity_model,
                        stoch_ffp,
                        work_directory,
                        hf_sim_path,
                    ),
                    stations.values.tolist(),
                ),
                total=len(stations),
            )
        )
        stations["epicentre_distance"] = epicentre_distances
    start = time.process_time()
    with open(velocity_model, "r") as f:
        f.readline()
        vs = np.float32(float(f.readline().split()[2]) * 1000.0)
        stations["vs"] = vs

    nt = int(domain_parameters.duration / hf_config.dt)
    with open(out_file, "wb") as output_file_handle:
        header_data: list[Any] = (
            [
                len(stations),
                nt,
                hf_config.seed,
                int(hf_config.site_specific),
                hf_config.path_dur,
                len(hf_config.rayset),
            ]
            + hf_config.rayset
            + [0] * (4 - len(hf_config.rayset))
            + [
                hf_config.nbu,
                hf_config.ift,
                hf_config.nl_skip,
                int(hf_config.ic_flag),
                1,  # this part of the header says that the HF was generated with individual stations in parallel
                int(hf_config.site_specific),
                domain_parameters.duration,
                hf_config.dt,
                0.0,  # start time of the timeseries (assumed to be 0)
                hf_config.sdrop,
                hf_config.kappa,
                hf_config.qfexp,
                hf_config.fmax,
                hf_config.flo,
                hf_config.fhi,
                hf_config.rvfac,
                hf_config.rvfac_shal,
                hf_config.rvfac_deep,
                hf_config.czero or -1.0,
                hf_config.calpha or -1.0,
                hf_config.mom or -1.0,
                hf_config.rupv or -1.0,
                hf_config.vs_moho,
                hf_config.vp_sig,
                hf_config.vsh_sig,
                hf_config.rho_sig,
                hf_config.qs_sig,
                hf_config.fa_sig1,
                hf_config.fa_sig2,
                hf_config.rv_sig1,
                stoch_ffp.name.encode("utf-8"),
                velocity_model.name.encode("utf-8"),
            ]
        )
        format_specifiers = {bytes: "64s", int: "i", float: "f"}
        header_format = "".join(
            [format_specifiers[type(value)] for value in header_data]
        )
        output_file_handle.write(struct.pack(header_format, *header_data))
        output_file_handle.write(b"\0" * 512)
        # Write station headers to HF output
        stations.apply(
            lambda station: output_file_handle.write(
                struct.pack(
                    "ff8sff",
                    station["longitude"],
                    station["latitude"],
                    station["name"].encode("utf-8"),
                    station["epicentre_distance"],
                    station["vs"],
                )
            ),
            axis=1,
        )
        # Copy station timeseries from each station into the master
        # file. Usually an iterrows here would be a red flag, but this
        # should be ok because the number of stations is in the
        # thousands, not the millions.
        for _, station in stations.iterrows():
            station_file_path = work_directory / f"{station['name']}.hf"
            with open(station_file_path, mode="rb") as station_file_data:
                shutil.copyfileobj(station_file_data, output_file_handle)

    print(f"Merging time took: {time.process_time() - start}")


def main():
    typer.run(run_hf)


if __name__ == "__main__":
    main()
