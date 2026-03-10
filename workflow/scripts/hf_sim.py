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
from collections.abc import Iterable
from concurrent.futures.thread import ThreadPoolExecutor
from pathlib import Path
from typing import Annotated

import numpy as np
import numpy.typing as npt
import pandas as pd
import typer
import xarray as xr

from qcore import cli
from workflow import log_utils, realisations, utils
from workflow.realisations import (
    DomainParameters,
    HFConfig,
    HFVelocityModel1D,
    RealisationMetadata,
    Resolution,
    RuptureVelocity,
    Seeds,
)

app = typer.Typer()


def rupture_velocity_hf_transition_bands(
    rupture_velocity: RuptureVelocity,
) -> tuple[float, float, float, float]:
    """Produce transition bands for rupture velocity parameters.

    Converts median-centred description into bounds description.

    Parameters
    ----------
    rupture_velocity : RuptureVelocity
        Rupture velocity configuration


    Returns
    -------
    tuple[float, float, float, float]
        The shallow min/max, deep min/max transition depths.
    """
    deep = rupture_velocity.deep_depth
    deep_range = rupture_velocity.deep_transition_range
    shallow = rupture_velocity.shallow_depth
    shallow_range = rupture_velocity.shallow_transition_range
    deep_min = deep - deep_range
    deep_max = deep + deep_range
    shallow_min = shallow - shallow_range
    shallow_max = shallow + shallow_range
    return shallow_min, shallow_max, deep_min, deep_max


def build_hf_input(
    stoch_ffp: Path,
    velocity_model: Path,
    resolution: Resolution,
    hf_config: HFConfig,
    rupture_velocity: RuptureVelocity,
    domain_parameters: DomainParameters,
) -> str:
    """Build a high-frequency input template string.

    Parameters
    ----------
    stoch_ffp : Path
        The path to the stoch file.
    velocity_model : Path
        The path to the velocity model.
    resolution : Resolution
        HF simulation resolution.
    hf_config : HFConfig
        The high-frequency config.
    rupture_velocity : RuptureVelocity
        The rupture velocity settings.
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
    shallow_min, shallow_max, deep_min, deep_max = rupture_velocity_hf_transition_bands(
        rupture_velocity
    )
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
        f"{domain_parameters.duration} {resolution.dt} {hf_config.fmax} {hf_config.kappa} {hf_config.qfexp}",
        f"{rupture_velocity.rvfrac} {rupture_velocity.rvfrac_shal} {rupture_velocity.rvfrac_deep} {hf_config.czero} {hf_config.calpha}",
        # TODO: This requires PR from EMOD3D to merge before we can do this!
        # f"{shallow_min} {shallow_max} {deep_min} {deep_max}",
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


def stable_hash(station: str) -> int:
    """Compute stable hashes for station names.

    The HF binary expects seeds. We want the provided seed to be
    independent of the order of stations in the stations lists. This
    is so setting HF seed reproduces the same outputs, even for
    different orders or subsets of the original station file. To do
    that, we generate stable hashes based on the station name.


    Parameters
    ----------
    station : str
        The station name.

    Returns
    -------
    int
        A hash of the station name. This is guaranteed to be in the
        range of a signed 32-bit integer.
    """
    return int.from_bytes(
        hashlib.blake2b(station.encode("utf-8"), digest_size=4).digest(), signed=True
    )


def station_seeds(seed: int, stations: Iterable[str]) -> npt.NDArray[np.int32]:
    """Create a list of per-station seeds in an order-invariant fashion with a root seed.

    Parameters
    ----------
    seed : int
        The root seed.
    stations : Iterable[str]
        The stations to seed. The order and number of stations should
        not matter. The station seeds are based on their name only.

    Returns
    -------
    npt.NDArray[np.int32]
        A list of station seeds.
    """
    station_hashes = np.array([stable_hash(name) for name in stations], dtype=np.int32)
    # Rather than add (which could overflow and cause annoying numpy
    # warnings), we just xor the hf seed with the station hashes.
    # Since this is invertible, we ensure that the same hf seed gives
    # the same station seeds.
    return np.int32(seed) ^ station_hashes


def create_hf_dataset(
    # array-like used here to reduce the number of times we have to
    # change the types if the downstream function inputs change.
    waveform: npt.ArrayLike,
    latitude: npt.ArrayLike,
    longitude: npt.ArrayLike,
    names: npt.ArrayLike,
    epicentre_distance: npt.ArrayLike,
    seed: npt.ArrayLike,
    vref: npt.ArrayLike,
    dt: float,
    start_sec: float,
) -> xr.Dataset:
    """
    Create a structured xarray Dataset for HF simulation data.

    Parameters
    ----------
    waveform : ArrayLike
        The waveform data. Expected shape is (3, n_stations, nt),
        representing the three components (x, y, z).
    latitude : ArrayLike
        Latitude coordinates for each station. Shape (n_stations,).
    longitude : ArrayLike
        Longitude coordinates for each station. Shape (n_stations,).
    names : ArrayLike
        Names/IDs for each station. Shape (n_stations,). Used as the
        primary index for the 'station' dimension.
    epicentre_distance : ArrayLike
        Distance from the station to the epicentre. Shape (n_stations,).
    seed : ArrayLike
        Random seed values associated with each station. Shape (n_stations,).
    vref : ArrayLike
        Reference velocity (Vs30 or similar) for each station. Shape (n_stations,).
    dt : float
        Time step increment in seconds.
    start_sec : float
        The start time of the simulation in seconds.

    Returns
    -------
    xr.Dataset
        A dataset containing the waveforms and associated station metadata,
        indexed by station, component, and time.

    Notes
    -----
    The dataset follows specific dimensional mapping:
    * **waveform**: mapped to (component, station, time).
    * **coordinates**: 'lat' and 'lon' are non-index coordinates tied to
      the 'station' dimension.
    * **attributes**: global metadata includes 'units' (fixed to cm/s^2),
      'nt', and 'dt'.
    """
    waveform = np.asarray(waveform)
    nt = waveform.shape[-1]
    time = np.arange(nt) * dt
    return xr.Dataset(
        {
            "waveform": (["component", "station", "time"], waveform),
            "epicentre_distance": (["station"], epicentre_distance),
            "seed": (["station"], seed),
            "vref": (["station"], vref),
        },
        coords={
            "station": ("station", names),
            "component": ("component", ["x", "y", "z"]),
            "time": ("time", time),
            "lat": (["station"], latitude),
            "lon": (["station"], longitude),
        },
        attrs={
            "start_sec": start_sec,
            "nt": nt,
            "dt": dt,
            "units": "cm/s^2",
        },
    )


@cli.from_docstring(app)
@log_utils.log_call()
def run_hf(
    realisation_ffp: Annotated[Path, typer.Argument()],
    stoch_ffp: Annotated[
        Path,
        typer.Argument(exists=True),
    ],
    station_file: Annotated[Path, typer.Argument(exists=True)],
    out_file: Annotated[Path, typer.Argument()],
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
    seeds = Seeds.read_from_realisation_or_random(realisation_ffp)

    domain_parameters = DomainParameters.read_from_realisation(realisation_ffp)
    metadata = RealisationMetadata.read_from_realisation(realisation_ffp)
    velocity_model = HFVelocityModel1D.read_from_realisation_or_defaults(
        realisation_ffp, metadata.defaults_version
    )
    hf_config = HFConfig.read_from_realisation_or_defaults(
        realisation_ffp, metadata.defaults_version
    )
    rupture_velocity = RuptureVelocity.read_from_realisation_or_defaults(
        realisation_ffp, metadata.defaults_version
    )
    resolution = Resolution.read_from_realisation_or_defaults(
        realisation_ffp, metadata.defaults_version
    )

    stations = pd.read_csv(
        station_file,
        delimiter=r"\s+",
        header=None,
        names=["longitude", "latitude", "name"],
    ).set_index("name")
    stations["seed"] = station_seeds(seeds.hf_seed, stations.index)
    velocity_model_path = work_directory / "velocity_model"
    velocity_model.write_velocity_model(velocity_model_path)
    nt = int(
        np.float32(domain_parameters.duration) / np.float32(resolution.dt)
    )  # Match Fortran's single-precision for consistent nt calculation
    waveform = np.empty((3, len(stations), nt), dtype=np.float32)

    hf_input_template = build_hf_input(
        stoch_ffp,
        velocity_model_path,
        resolution,
        hf_config,
        rupture_velocity,
        domain_parameters,
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
                str(name),
                int(station["seed"]),
            )
            for name, station in stations.iterrows()
        ]
        for future in concurrent.futures.as_completed(futures):
            station, epicentre, station_waveform = future.result()
            stations.loc[station, "epicentre_distance"] = epicentre
            i = station_index[station]

            for component in range(3):
                waveform[component, i] = station_waveform[:, component]

    vs = velocity_model.model["Vs"].iloc[0] * 1000
    stations["vs"] = vs

    ds = create_hf_dataset(
        waveform=waveform,
        latitude=stations["latitude"],
        longitude=stations["longitude"],
        names=stations.index,
        epicentre_distance=stations["epicentre_distance"],
        seed=stations["seed"],
        vref=stations["vs"],
        dt=resolution.dt,
        start_sec=hf_config.t_sec,
    )
    ds.to_netcdf(out_file, engine="h5netcdf")
    realisations.append_log_entry(realisation_ffp)
