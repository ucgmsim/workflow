#!/usr/bin/env python
"""High Frequency Simulation.

Description
-----------
Generate stochastic high frequency ground acceleration data for a number of stations.

Inputs
------
1. A station list (in the "latitude longitude name" format),
2. A 1D velocity model,
3. A stoch file,
4. A realisation with domain parameters and metadata.

Outputs
-------
1. A combined HF simulation output containing ground acceleration data for each station.

Environment
-----------
Can be run in the cybershake container. Can also be run from your own computer using the `hf-sim` command which is installed after running `pip install workflow@git+https://github.com/ucgmsim/workflow`. If you do run this on your own computer, you need a version of `hb_high_binmod` installed.

> [!NOTE]
> The high-frequency code is very brittle. It is recommended you have both versions 6.0.3 and 5.4.5 built to run with. Sometimes it is necessary to switch between versions if one does not work.

Usage
-----
`hf-sim [OPTIONS] REALISATION_FFP STOCH_FFP STATION_FILE OUT_FILE`

For More Help
-------------
See the output of `hf-sim --help`.
"""

import functools
import multiprocessing
import os
import shutil
import struct
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Annotated, Any

import numpy as np
import pandas as pd
import tqdm
import typer

from workflow import log_utils
from workflow.realisations import DomainParameters, HFConfig, RealisationMetadata

app = typer.Typer()

HEAD_STAT = 24
FLOAT_SIZE = 4


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
    """Simulate a seismic station using the HF (High-Frequency) simulation tool.

    Parameters
    ----------
    hf_config : HFConfig
        Configuration object containing the parameters for the high-frequency simulation.
    domain_parameters : DomainParameters
        Domain parameters such as simulation duration and other domain-specific settings.
    velocity_model : Path
        Path to the velocity model file used in the simulation.
    stoch_ffp : Path
        Path to the stoch file used as input for the simulation.
    output_directory : Path
        Directory where the HF output file will be saved.
    hf_sim_path : Path
        Path to the HF simulation binary.
    longitude : float
        Longitude of the seismic station.
    latitude : float
        Latitude of the seismic station.
    name : str
        Name of the seismic station, used for naming the output file.

    Returns
    -------
    float
        The epicentre distance obtained from the simulation output.

    Raises
    ------
    ValueError
        If the output does not contain exactly one epicentre distance value.
    CalledProcessError
        If the HF binary throws an error. A note to the exception is
        added with the stderr.
    """
    raw_hf_output_ffp = output_directory / f"{name}.hf"
    # expected size is n_components (3) * float size (4) * number of timesteps
    expected_size = 12 * round(domain_parameters.duration / hf_config.dt)
    if (
        raw_hf_output_ffp.exists()
        and os.stat(raw_hf_output_ffp).st_size == expected_size
    ):
        return
    with tempfile.NamedTemporaryFile(mode="w") as station_input_file:
        station_input_file.write(f"{longitude} {latitude} {name}\n")
        station_input_file.flush()
        hf_sim_input = [
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

        try:
            hf_sim_input_str = "\n".join(str(line) for line in hf_sim_input)

            print("---\n" + hf_sim_input_str + "\n---")
            log_utils.log("running hf", station=name, input=hf_sim_input_str)
            output = subprocess.run(
                str(hf_sim_path),
                input=hf_sim_input_str,
                check=True,
                text=True,
                stderr=subprocess.PIPE,
            )
        except subprocess.CalledProcessError as e:
            log_utils.log("hf failed", station=name, stdout=e.stdout, stderr=e.stderr)
            raise
        log_utils.log("hf succeeded", station=name)
        epicentre_distance = np.fromstring(output.stderr, dtype="f4", sep="\n")
        if epicentre_distance.size != 1:
            raise ValueError(
                f"Expected exactly one epicentre_distance value, got {epicentre_distance.size}"
            )
        return epicentre_distance[0]


@app.command(
    help="Run the HF (High-Frequency) simulation and generate the HF output file."
)
@log_utils.log_call()
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
        "/EMOD3D/tools/hb_high_binmod_v6.0.3"
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
    """Run the HF (High-Frequency) simulation and generate the HF output file.

    This function performs the following steps:
    1. Reads configuration and domain parameters from the realisation file.
    2. Filters stations based on their location relative to the domain.
    3. Uses multiprocessing to simulate each station and calculate epicentre distances.
    4. Reads the velocity model and calculates the `vs` value.
    5. Writes the HF output file, including header and station-specific data.

    Parameters
    ----------
    realisation_ffp : Path
        Path to the JSON file containing realisation data.
    stoch_ffp : Path
        Path to the input stochastic file.
    station_file : Path
        Path to the file containing station locations and names.
    out_file : Path
        Filepath where the HF output will be saved.
    hf_sim_path : Path, optional
        Path to the HF simulation binary.
    velocity_model : Path, optional
        Path to the 1D velocity model. Ignored if site-specific model
        is set is set in the `HFConfig` of `realisation_ffp`.
    work_directory : Path, optional
        Directory for intermediate files. Must be writable.

    Returns
    -------
    None
        The function does not return any value. It writes the HF output directly to `out_file`.
    """
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
        stations["epicentre_distance"] = pool.starmap(
            functools.partial(
                hf_simulate_station,
                hf_config,
                domain_parameters,
                velocity_model,
                stoch_ffp,
                work_directory,
                hf_sim_path,
            ),
            stations.values.tolist(),
        )

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
