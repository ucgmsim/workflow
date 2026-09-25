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

Usage
-----
`hf-sim [OPTIONS] REALISATION_FFP STOCH_FFP STATION_FILE OUT_FILE`

For More Help
-------------
See the output of `hf-sim --help`.
"""

from pathlib import Path
from typing import Annotated, Any

import dask
import dask.array as da
import numpy as np
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

    stations["seed"] = hf_simulation.station_seeds(
        seeds.hf_seed, stations.index.to_list()
    )
    stations = stations.sort_values("seed")

    simulator = Simulator(
        _build_slip_model(stoch_ffp),
        _build_velocity_model(velocity_model_1d),
        _build_config(hf_config, rupture_velocity, domain_parameters),
    )

    # float32 throughout: this mirrors how the simulation truncates duration/dt to a
    # sample count, so the dask template matches what comes back.
    nt = round(domain_parameters.duration / hf_config.dt)
    # The record starts at the origin time. This was a configurable `t_sec` that every
    # realisation set to zero.
    time = np.arange(nt) * hf_config.dt

    # Also bound by parallelism: chunk size set from memory alone gives 3 tasks for a
    # 900-station run, so most of the allocation idles. Peak memory is
    # num_workers * chunk_bytes, which this only ever lowers.
    num_workers = utils.get_available_cores()
    memory_chunk = TARGET_CHUNK_BYTES // (len(COMPONENTS) * nt * np.float32().itemsize)
    chunk_size = max(1, min(memory_chunk, -(-len(stations) // (4 * num_workers))))
    logger = log_utils.get_logger(__name__)
    logger.info(
        "concurrency settings",
        num_workers=num_workers,
        memory_bound_stations=memory_chunk,
        chunk_size=chunk_size,
    )
    with (
        dask.config.set(scheduler="threads", num_workers=num_workers),
        TqdmCallback(desc="Station chunks"),
    ):
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
            _simulate_chunk,
            template=template,
            kwargs={"time": time, "simulator": simulator},
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
        }
        dataset.to_netcdf(out_file, engine="h5netcdf")
    realisations.append_log_entry(realisation_ffp)
