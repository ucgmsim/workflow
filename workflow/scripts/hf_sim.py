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
Can be run in the cybershake container. Can also be run from your own computer using the
`hf-sim` command after `pip install workflow@git+https://github.com/ucgmsim/workflow`.

Two simulators are available, chosen with `--simulator`. `rust` (the default) is the
`hf-simulation` package. `fortran` is EMOD3D's `hb_high_binmod`, which must be installed
(see `--hf-sim-path`); the cybershake container ships it. Both read the same `hf`
configuration, but they draw different random numbers, so their waveforms differ.

Usage
-----
`hf-sim [OPTIONS] REALISATION_FFP STOCH_FFP STATION_FILE OUT_FILE`

For More Help
-------------
See the output of `hf-sim --help`.
"""

import subprocess
import tempfile
from collections.abc import Iterable
from enum import StrEnum, auto
from pathlib import Path
from typing import Annotated, Any

import dask
import dask.array as da
import numpy as np
import numpy.typing as npt
import pandas as pd
import typer
import xarray as xr
from tqdm.dask import TqdmCallback

import hf_simulation
from hf_simulation import (
    COMPONENTS,
    FaultSegment,
    HfConfig,
    PathDurationModel,
    PathParameters,
    Ray,
    RecordParameters,
    Simulator,
    SiteParameters,
    SlipModel,
    SourceParameters,
    VelocityModel1D,
)
from hf_simulation import (
    RuptureVelocity as RuptureVelocityTaper,
)
from qcore import cli
from source_modelling.stoch import StochFile
from workflow import log_utils, realisations, utils
from workflow.realisations import (
    DomainParameters,
    HFVelocityModel1D,
    RealisationMetadata,
    RuptureVelocity,
    Seeds,
)
from workflow.realisations import (
    HFConfig as HFConfigDefaults,
)
from workflow.waveforms import Component

app = typer.Typer()

TARGET_CHUNK_BYTES = 128 * 2**20
"""Target size of a dask chunk (all components for a batch of stations)."""

FORTRAN_COMPONENTS = (Component.EAST, Component.NORTH, Component.UP)
"""Component of each column in `hb_high`'s output, which writes 090, 000, ver."""


class HFSimulator(StrEnum):
    """The high-frequency simulator to run."""

    RUST = auto()
    """The `hf-simulation` package."""
    FORTRAN = auto()
    """EMOD3D's `hb_high_binmod` binary."""


def _build_config(
    hf_config: HFConfigDefaults,
    rupture_velocity: RuptureVelocity,
    domain_parameters: DomainParameters,
) -> HfConfig:
    """Translate the realisation's configuration into the simulation's."""
    source: dict[str, Any] = hf_config.source | {
        "rupture_velocity": RuptureVelocityTaper(
            fraction=rupture_velocity.rvfrac,
            shallow=rupture_velocity.rvfrac_shal,
            deep=rupture_velocity.rvfrac_deep,
            **hf_config.source["rupture_velocity"],
        )
    }
    path: dict[str, Any] = hf_config.path | {
        "rayset": tuple(Ray(ray) for ray in hf_config.path["rayset"]),
        "path_duration_model": PathDurationModel(hf_config.path["path_duration_model"]),
    }
    return HfConfig(
        source=SourceParameters(**source),
        path=PathParameters(**path),
        site=SiteParameters(**hf_config.site),
        record=RecordParameters(
            duration_s=domain_parameters.duration, **hf_config.record
        ),
    )


def _build_slip_model(stoch_ffp: Path) -> SlipModel:
    """Read a stoch file into a simulation slip model."""
    stoch = StochFile.from_file(stoch_ffp)
    return SlipModel(
        [
            FaultSegment(
                longitude_deg=plane.header.longitude,
                latitude_deg=plane.header.latitude,
                strike_deg=plane.header.strike,
                dip_deg=plane.header.dip,
                rake_deg=plane.header.average_rake,
                top_depth_km=plane.header.dtop,
                subfault_length_km=plane.header.dx,
                subfault_width_km=plane.header.dy,
                hypocentre_along_strike_km=plane.header.shypo,
                hypocentre_down_dip_km=plane.header.dhypo,
                # (down-dip, along-strike), which is how the stoch format stores them.
                slip=plane.slip.astype(np.float32),
                rise_time_s=plane.rise.astype(np.float32),
                rupture_time_s=plane.trup.astype(np.float32),
            )
            for plane in stoch.data
        ]
    )


def _build_velocity_model(velocity_model: HFVelocityModel1D) -> VelocityModel1D:
    """Convert the realisation's 1D velocity model into the simulation's."""
    model = velocity_model.model
    return VelocityModel1D(
        thickness_km=model["thickness"].to_numpy(np.float32),
        vp_km_s=model["Vp"].to_numpy(np.float64),
        vsh_km_s=model["Vs"].to_numpy(np.float64),
        density_g_cm3=model["rho"].to_numpy(np.float64),
        quality_factor_p=model["Qp"].to_numpy(np.float32),
        quality_factor_s=model["Qs"].to_numpy(np.float32),
        vs_moho_km_s=velocity_model.vs_moho,
    )


def fortran_station_seeds(seed: int, stations: Iterable[str]) -> npt.NDArray[np.int32]:
    """Create per-station seeds for the Fortran simulator.

    `hb_high` reads a 32-bit seed, so it cannot take `hf_simulation.station_seeds`.
    These are the seeds the workflow gave it before the Rust rewrite, so Fortran runs
    reproduce those.

    Parameters
    ----------
    seed : int
        The root seed.
    stations : Iterable[str]
        The stations to seed. The seeds depend on the station names only, not on their
        order or number.

    Returns
    -------
    npt.NDArray[np.int32]
        A seed for each station.
    """
    station_hashes = np.array(
        [utils.stable_hash(name) for name in stations], dtype=np.int32
    )
    # xor rather than add, which could overflow. It is invertible, so the same root seed
    # always gives the same station seeds.
    return np.int32(seed) ^ station_hashes


def build_hf_input(
    hf_config: HFConfigDefaults,
    rupture_velocity: RuptureVelocity,
    velocity_model: HFVelocityModel1D,
    duration: float,
    stoch_ffp: Path,
    velocity_model_ffp: Path,
) -> str:
    """Build the stdin deck `hb_high` reads for one station.

    Parameters
    ----------
    hf_config : HFConfigDefaults
        The high-frequency configuration.
    rupture_velocity : RuptureVelocity
        The rupture velocity multipliers.
    velocity_model : HFVelocityModel1D
        The 1D velocity model, for its `vs_moho`.
    duration : float
        The record duration, seconds.
    stoch_ffp : Path
        The stoch file.
    velocity_model_ffp : Path
        Where the 1D velocity model has been written.

    Returns
    -------
    str
        A deck with `station_input_file`, `output_file` and `seed` format placeholders.
    """
    source = hf_config.source
    path = hf_config.path
    site = hf_config.site
    rayset = path["rayset"]
    # Values the configuration does not carry are fixed at what every realisation set
    # them to before the Rust rewrite; -1 tells `hb_high` to compute the value itself.
    deck = [
        "",
        source["stress_drop_bars"],
        "{station_input_file}",
        "{output_file}",
        f"{len(rayset)} {' '.join(str(ray) for ray in rayset)}",
        1,  # BJ97 site amplification on
        "4 0 0.02 19.9",  # nbu, ift, flo, fhi
        "{seed}",
        1,  # one station in the input
        (
            f"{duration} {hf_config.dt} {site['fmax_hz']} {site['kappa_s']} "
            f"{path['q_frequency_exponent']}"
        ),
        (
            f"{rupture_velocity.rvfrac} {rupture_velocity.rvfrac_shal} "
            f"{rupture_velocity.rvfrac_deep} {source['corner_frequency_constant']} "
            f"{source['corner_frequency_alpha']}"
        ),
        "-1 -1",  # seismic moment and rupture velocity
        stoch_ffp,
        velocity_model_ffp,
        velocity_model.vs_moho,
        "-99 0.0 0.0 0.0 0.0 1",  # nl_skip, vp/vsh/rho/qs sigmas, ic_flag
        "-1",  # velocity name
        f"0.0 0.0 {source['rupture_velocity']['sigma']}",  # fa_sig1, fa_sig2, rv_sig1
        path["path_duration_model"],
        0,  # log path duration perturbation
        "-1 -1 -1",  # stress parameter adjustment off
        0,  # no binary offset into the output
        "",
    ]
    return "\n".join(str(line) for line in deck)


def _simulate_station_fortran(
    hf_sim_path: Path,
    hf_input: str,
    latitude: float,
    longitude: float,
    station_name: str,
    seed: int,
    nt: int,
) -> np.ndarray:
    """Run `hb_high` for one station, returning its (nt, 3) waveform in 090/000/ver."""
    with (
        tempfile.NamedTemporaryFile(mode="w") as input_file,
        tempfile.NamedTemporaryFile() as output_file,
    ):
        input_file.write(f"{longitude} {latitude} {station_name}\n")
        input_file.flush()
        deck = hf_input.format(
            station_input_file=input_file.name, output_file=output_file.name, seed=seed
        )
        try:
            subprocess.run(
                str(hf_sim_path),
                input=deck,
                check=True,
                text=True,
                stderr=subprocess.PIPE,
            )
        except subprocess.CalledProcessError as e:
            log_utils.get_logger(__name__).error(
                "hf failed", station=station_name, input=deck, stderr=e.stderr
            )
            e.add_note(e.stderr)
            raise
        waveform = np.fromfile(output_file.name, dtype=np.float32).reshape((-1, 3))
    if len(waveform) != nt:
        raise RuntimeError(
            f"hb_high wrote {len(waveform)} samples for {station_name}, expected {nt}."
        )
    return waveform


def _simulate_chunk_fortran(
    station_chunk: xr.Dataset,
    time: np.ndarray,
    hf_sim_path: Path,
    hf_input: str,
) -> xr.DataArray:
    """Simulate one dask block's worth of stations, one `hb_high` call per station."""
    station_names = station_chunk["station"].values
    waveform = np.empty((3, len(station_names), len(time)), dtype=np.float32)
    for i, (name, latitude, longitude, seed) in enumerate(
        zip(
            station_names,
            station_chunk["latitude"].values,
            station_chunk["longitude"].values,
            station_chunk["seed"].values,
        )
    ):
        waveform[:, i] = _simulate_station_fortran(
            hf_sim_path,
            hf_input,
            float(latitude),
            float(longitude),
            str(name),
            int(seed),
            len(time),
        ).T
    return xr.DataArray(
        waveform,
        dims=["component", "station", "time"],
        coords={
            "component": list(FORTRAN_COMPONENTS),
            "station": station_names,
            "time": time,
        },
    ).sel(component=list(Component))


def _simulate_chunk(
    station_chunk: xr.Dataset,
    time: np.ndarray,
    simulator: Simulator,
) -> xr.DataArray:
    """Simulate one dask block's worth of stations in a single call."""
    station_names = station_chunk["station"].values
    waveform = simulator.run_stations(
        latitude_deg=station_chunk["latitude"].values.astype(np.float32),
        longitude_deg=station_chunk["longitude"].values.astype(np.float32),
        station_seed=station_chunk["seed"].values.astype(np.uint64),
    )
    # `hf_simulation` already uses the workflow's component labels, but orders them
    # 090, 000, ver; waveform files store them in `Component` order.
    return xr.DataArray(
        waveform,
        dims=["component", "station", "time"],
        coords={
            "component": list(COMPONENTS),
            "station": station_names,
            "time": time,
        },
    ).sel(component=list(Component))


@cli.from_docstring(app)
@log_utils.log_call()
def run_hf(
    realisation_ffp: Annotated[Path, typer.Argument()],
    stoch_ffp: Annotated[Path, typer.Argument(exists=True)],
    station_file: Annotated[Path, typer.Argument(exists=True)],
    out_file: Annotated[Path, typer.Argument()],
    simulator: Annotated[HFSimulator, typer.Option()] = HFSimulator.RUST,
    hf_sim_path: Annotated[Path, typer.Option()] = Path(
        "/EMOD3D/tools/hb_high_binmod_v6.0.3"
    ),
) -> None:
    """Run the HF simulation and write the HF output file.

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
    simulator : HFSimulator
        The simulator to run: the `hf-simulation` package (rust) or EMOD3D's
        `hb_high_binmod` (fortran).
    hf_sim_path : Path
        Path to the `hb_high_binmod` binary. Only used by the fortran simulator.
    """
    metadata = RealisationMetadata.read_from_realisation(realisation_ffp)
    seeds = Seeds.read_from_realisation_or_random(realisation_ffp)
    domain_parameters = DomainParameters.read_from_realisation(realisation_ffp)
    hf_config = HFConfigDefaults.read_from_realisation_or_defaults(
        realisation_ffp, metadata.defaults_version
    )
    rupture_velocity = RuptureVelocity.read_from_realisation_or_defaults(
        realisation_ffp, metadata.defaults_version
    )
    velocity_model_1d = HFVelocityModel1D.read_from_realisation_or_defaults(
        realisation_ffp, metadata.defaults_version
    )

    stations = pd.read_csv(
        station_file,
        delimiter=r"\s+",
        header=None,
        names=["longitude", "latitude", "station"],
    ).set_index("station")

    if simulator == HFSimulator.RUST:
        stations["seed"] = hf_simulation.station_seeds(
            seeds.hf_seed, stations.index.to_list()
        )
    else:
        stations["seed"] = fortran_station_seeds(seeds.hf_seed, stations.index)
    stations = stations.sort_values("seed")

    # float32 throughout: this mirrors how both simulators truncate duration/dt to a
    # sample count, so the dask template matches what comes back. Rounding instead
    # overcounts by one whenever duration/dt has a fractional part >= 0.5.
    nt = int(np.float32(domain_parameters.duration) / np.float32(hf_config.dt))
    # The record starts at the origin time. This was a configurable `t_sec` that every
    # realisation set to zero.
    time = np.arange(nt) * hf_config.dt

    num_workers = utils.get_available_cores()
    memory_chunk = TARGET_CHUNK_BYTES // (len(COMPONENTS) * nt * np.float32().itemsize)
    chunk_size = max(1, min(memory_chunk, -(-len(stations) // (4 * num_workers))))
    logger = log_utils.get_logger(__name__)
    logger.info(
        "concurrency settings",
        simulator=simulator,
        num_workers=num_workers,
        memory_bound_stations=memory_chunk,
        chunk_size=chunk_size,
    )
    with (
        tempfile.TemporaryDirectory() as work_directory,
        dask.config.set(scheduler="threads", num_workers=num_workers),
        TqdmCallback(desc="Station chunks"),
    ):
        if simulator == HFSimulator.RUST:
            simulate_chunk = _simulate_chunk
            kwargs: dict[str, Any] = {
                "simulator": Simulator(
                    _build_slip_model(stoch_ffp),
                    _build_velocity_model(velocity_model_1d),
                    _build_config(hf_config, rupture_velocity, domain_parameters),
                )
            }
        else:
            velocity_model_ffp = Path(work_directory) / "velocity_model"
            velocity_model_1d.write_velocity_model(velocity_model_ffp)
            simulate_chunk = _simulate_chunk_fortran
            kwargs = {
                "hf_sim_path": hf_sim_path,
                "hf_input": build_hf_input(
                    hf_config,
                    rupture_velocity,
                    velocity_model_1d,
                    domain_parameters.duration,
                    stoch_ffp,
                    velocity_model_ffp,
                ),
            }

        template = xr.DataArray(
            da.empty(
                (len(Component), len(stations), nt),
                dtype=np.float32,
                chunks=(len(Component), chunk_size, nt),
            ),
            dims=["component", "station", "time"],
            coords={
                "component": list(Component),
                "station": stations.index,
                "time": time,
            },
        )

        station_inputs = stations.to_xarray().chunk({"station": chunk_size})
        waveform = station_inputs.map_blocks(
            simulate_chunk,
            template=template,
            kwargs={"time": time} | kwargs,
        ).rename("waveform")

        # The Vs30 the HF waveforms are simulated at, which bb-sim amplifies from.
        station_inputs["vref"] = xr.full_like(
            station_inputs["latitude"], velocity_model_1d.model["Vs"].iloc[0] * 1000
        )
        dataset = xr.merge([waveform, station_inputs])
        dataset.attrs = {
            "start_sec": 0.0,
            "dt": hf_config.dt,
            "nt": nt,
            "units": "cm/s^2",
            "simulator": str(simulator),
        }
        dataset.to_netcdf(out_file, engine="h5netcdf")
    realisations.append_log_entry(realisation_ffp)
