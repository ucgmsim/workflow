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

import concurrent.futures
import subprocess
import tempfile
from concurrent.futures.thread import ThreadPoolExecutor
from pathlib import Path
from typing import Annotated

import numpy as np
import pandas as pd
import typer
import xarray as xr

from qcore import cli
from workflow import log_utils, realisations, utils
from workflow.realisations import (
    DomainParameters,
    HFConfig,
    RealisationMetadata,
    Seeds,
    VelocityModel1D,
)

app = typer.Typer()


def build_hf_input(
    stoch_ffp: Path,
    velocity_model: Path,
    hf_config: HFConfig,
    seeds: Seeds,
    domain_parameters: DomainParameters,
) -> str:
    """Build a high-frequency input template string.

    Parameters
    ----------
    stoch_ffp : Path
        The path to the stoch file.
    velocity_model : Path
        The path to the velocity model.
    hf_config : HFConfig
        The high-frequency config.
    seeds : Seeds
        The seeds.
    domain_parameters : DomainParameters
        The simulation domain parameters.

    Returns
    -------
    str
        A template HF input, this template has two format placeholders
        `station_input_file` and `output_file` which can be
        substituted to yield a high-frequency input in for each
        station.
    """
    hf_sim_input = [
        "",
        hf_config.sdrop,
        "{station_input_file}",
        "{output_file}",
        f"{len(hf_config.rayset)} {' '.join(str(ray) for ray in hf_config.rayset)}",
        int(not hf_config.no_siteamp),
        f"{hf_config.nbu} {hf_config.ift} {hf_config.flo} {hf_config.fhi}",
        "{seed}",
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
        # If running v5.4.5 it stops reading input here and so
        # these parameters are unused. It is harmless to add them
        # regardless of version
        f"{hf_config.stress_parameter_adjustment_fault_area or -1} "
        f"{hf_config.stress_parameter_adjustment_target_magnitude or -1} "
        f"{hf_config.stress_parameter_adjustment_tect_type or -1}",
        0,  # seek bytes to 0 (no binary offset for this output)
        "",
    ]
    return "\n".join(str(line) for line in hf_sim_input)


def hf_simulate_station(
    hf_sim_path: Path,
    hf_stdin_template: str,
    station_latitude: float,
    station_longitude: float,
    station_name: str,
    seed: int,
) -> tuple[str, float, np.ndarray]:
    """Simulate a seismic station using the HF (High-Frequency) simulation tool.

    Parameters
    ----------
    hf_sim_path : Path
        The path to the HF simulation binary.
    hf_stdin_template : str
        The stdin input template for the HF simulation binary.
    station_latitude : float
        The station latitude.
    station_longitude : float
        The station longitude.
    station_name : str
        The station name.
    seed : int
        The seed for this HF simulation.

    Returns
    -------
    str
        The completed station name.
    float
        The epicentre distance obtained from the simulation output.
    array of floats
        The simulation waveform.

    Raises
    ------
    ValueError
        If the output does not contain exactly one epicentre distance value.
    CalledProcessError
        If the HF binary throws an error. A note to the exception is
        added with the stderr.
    """
    with (
        tempfile.NamedTemporaryFile(mode="w") as input_file,
        tempfile.NamedTemporaryFile() as output_file,
    ):
        input_file.write(f"{station_longitude} {station_latitude} {station_name}\n")
        input_file.flush()

        hf_sim_input_str = hf_stdin_template.format(
            station_input_file=input_file.name, output_file=output_file.name, seed=seed
        )

        logger = log_utils.get_logger(__name__)
        logger.info("running hf", station=station_name, input=hf_sim_input_str)

        try:
            output = subprocess.run(
                str(hf_sim_path),
                input=hf_sim_input_str,
                check=True,
                text=True,
                stderr=subprocess.PIPE,
            )
        except subprocess.CalledProcessError as e:
            logger.error(
                "hf failed", station=station_name, stdout=e.stdout, stderr=e.stderr
            )
            e.add_note(e.stderr)
            raise

        epicentre_distance = float(output.stderr.strip())

        logger.info(
            "hf succeeded",
            station=station_name,
            epicentre_distance=epicentre_distance,
            stderr=output.stderr,
        )

        station_waveform = np.fromfile(output_file, dtype=np.float32).reshape((-1, 3))

        return station_name, epicentre_distance, station_waveform


@cli.from_docstring(app)
@log_utils.log_call()
def run_hf(
    realisation_ffp: Annotated[Path, typer.Argument()],
    stoch_ffp: Annotated[
        Path,
        typer.Argument(exists=True),
    ],
    station_file: Annotated[Path, typer.Argument(exists=True)],
    out_file: Annotated[Path, typer.Argument(file_okay=False)],
    hf_sim_path: Annotated[Path, typer.Option()] = Path(
        "/EMOD3D/tools/hb_high_binmod_v6.0.3"
    ),
    work_directory: Annotated[
        Path,
        typer.Option(exists=True, writable=True, file_okay=False),
    ] = Path("/out"),
) -> None:
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
    work_directory : Path, optional
        Directory for intermediate files. Must be writable.

    Returns
    -------
    None
        The function does not return any value. It writes the HF output directly to `out_file`.
    """
    seeds = Seeds.read_from_realisation_or_defaults(realisation_ffp)
    rng = np.random.default_rng(seeds.hf_seed)
    domain_parameters = DomainParameters.read_from_realisation(realisation_ffp)
    metadata = RealisationMetadata.read_from_realisation(realisation_ffp)
    velocity_model = VelocityModel1D.read_from_realisation_or_defaults(
        realisation_ffp, metadata.defaults_version
    )
    hf_config = HFConfig.read_from_realisation_or_defaults(
        realisation_ffp, metadata.defaults_version
    )

    stations = pd.read_csv(
        station_file,
        delimiter=r"\s+",
        header=None,
        names=["longitude", "latitude", "name"],
    ).set_index("name")
    int_bounds = np.iinfo(np.int32)
    stations["seed"] = rng.integers(
        low=int_bounds.min,
        high=int_bounds.max,
        endpoint=True,
        size=len(stations),
        dtype=np.int32,
    )
    velocity_model_path = work_directory / "velocity_model"
    velocity_model.write_velocity_model(velocity_model_path)
    nt = int(domain_parameters.duration / hf_config.dt)
    waveform = np.empty((3, len(stations), nt), dtype=np.float32)

    hf_input_template = build_hf_input(
        stoch_ffp, velocity_model_path, hf_config, seeds, domain_parameters
    )

    stations["epicentre_distance"] = np.nan

    with ThreadPoolExecutor(max_workers=utils.get_available_cores()) as executor:
        station_index = {station: i for i, station in enumerate(stations.index)}
        futures = [
            executor.submit(
                hf_simulate_station,
                hf_sim_path,
                hf_input_template,
                station["latitude"],
                station["longitude"],
                name,
                station["seed"],
            )
            for name, station in stations.iterrows()
        ]
        for future in concurrent.futures.as_completed(futures):
            station, epicentre, station_waveform = future.result()
            stations.loc[station]["epicentre_distance"] = epicentre
            i = station_index[station]

            for component in range(3):
                waveform[component, i] = station_waveform[:, component]

    vs = velocity_model.model["Vs"].iloc[0] * 1000
    stations["vs"] = vs

    start_sec = 0.0
    time = start_sec + np.arange(nt) * hf_config.dt
    xr.Dataset(
        {
            "waveform": (["component", "station", "time"], waveform),
            "epicentre_distance": (["station"], stations["epicentre_distance"]),
            "seed": (["station"], stations["seed"]),
            "vref": (["station"], stations["vs"]),
        },
        coords={
            "station": stations.index,
            "time": time,
            "lat": (["station"], stations["latitude"]),
            "lon": (["station"], stations["longitude"]),
        },
        attrs={
            "start_sec": start_sec,
            "nt": nt,
            "dt": hf_config.dt,
            "units": "cm/s^2",
        },
    ).to_netcdf(out_file, engine="h5netcdf")
    realisations.append_log_entry(realisation_ffp)
