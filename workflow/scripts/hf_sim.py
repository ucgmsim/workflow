"""High Frequency Simulation.

Description
-----------
Generate stochastic high frequency ground acceleration data for a number of stations.

Inputs
------
1. A site table of the stations to simulate (`stations.parquet`, see `workflow.sites`),
2. A 1D velocity model,
3. A stoch slip model (HDF5) from `generate-stoch`,
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
`hf-sim [OPTIONS] REALISATION_FFP STOCH_FFP SITES_FFP OUT_FILE`

For More Help
-------------
See the output of `hf-sim --help`.
"""

from pathlib import Path
from typing import Annotated, Any

import dask
import dask.array as da
import numpy as np
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
from workflow import log_utils, realisations, sites, utils
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
    """Translate the realisation's configuration into the simulation's.

    The realisation carries `stress_drop_bars` as a run-wide constant, but the simulation
    takes it per subfault. It is left out here: the stress parameter comes from the slip
    model `generate-stoch` writes.
    """
    source: dict[str, Any] = _without(hf_config.source, "stress_drop_bars") | {
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


def _without(group: dict[str, Any], key: str) -> dict[str, Any]:
    """Return a configuration group without one of its keys."""
    return {name: value for name, value in group.items() if name != key}


def read_stoch(stoch_ffp: Path) -> tuple[float, float, list[xr.Dataset]]:
    """Read a stoch slip model written by `generate-stoch`.

    Parameters
    ----------
    stoch_ffp : Path
        Path to the slip model.

    Returns
    -------
    dx : float
        Subfault length along strike shared by every plane (km).
    dy : float
        Subfault width down dip shared by every plane (km).
    planes : list of xr.Dataset
        Each plane of the slip model, in SRF order.
    """
    with xr.open_datatree(stoch_ffp, engine="h5netcdf") as tree:
        # HDF5 lists groups by name, so plane_10 would come before plane_2.
        names = sorted(tree.children, key=lambda name: int(name.removeprefix("plane_")))
        planes = [tree[name].to_dataset().load() for name in names]
        return float(tree.attrs["dx"]), float(tree.attrs["dy"]), planes


def _build_slip_model(stoch_ffp: Path) -> SlipModel:
    """Read a stoch slip model written by `generate-stoch` into a simulation slip model."""
    dx, dy, planes = read_stoch(stoch_ffp)

    def grid(plane: xr.Dataset, name: str) -> np.ndarray:
        # (down-dip, along-strike), which is how the simulation takes them.
        return plane[name].transpose("dip", "strike").values.astype(np.float32)

    return SlipModel(
        [
            FaultSegment(
                longitude_deg=plane.attrs["longitude"],
                latitude_deg=plane.attrs["latitude"],
                strike_deg=plane.attrs["strike"],
                dip_deg=plane.attrs["dip"],
                rake_deg=plane.attrs["rake"],
                top_depth_km=plane.attrs["dtop"],
                subfault_length_km=dx,
                subfault_width_km=dy,
                hypocentre_along_strike_km=plane.attrs["shypo"],
                hypocentre_down_dip_km=plane.attrs["dhypo"],
                slip=grid(plane, "slip"),
                rise_time_s=grid(plane, "rise_time"),
                rupture_time_s=grid(plane, "rupture_time"),
                stress_drop_bars=grid(plane, "stress_drop"),
            )
            for plane in planes
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
    station_chunk: xr.Dataset, time: np.ndarray, simulator: Simulator
) -> xr.DataArray:
    """Simulate one dask block's worth of stations in a single call."""
    station_names = station_chunk["station"].values
    waveform = simulator.run_stations(
        latitude_deg=station_chunk["latitude"].values.astype(np.float32),
        longitude_deg=station_chunk["longitude"].values.astype(np.float32),
        station_seed=station_chunk["seed"].values.astype(np.uint64),
        kappa_s=station_chunk["kappa_s"].values.astype(np.float32),
        q_frequency_exponent=station_chunk["q_frequency_exponent"].values.astype(
            np.float32
        ),
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
    sites_ffp: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    out_file: Annotated[Path, typer.Argument()],
) -> None:
    """Run the HF simulation and write the HF output file.

    Parameters
    ----------
    realisation_ffp : Path
        Path to the JSON file containing realisation data.
    stoch_ffp : Path
        Path to the stoch slip model written by `generate-stoch`.
    sites_ffp : Path
        Path to the site table of the stations to simulate (see `workflow.sites`).
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

    stations = sites.read_sites(sites_ffp)[
        ["latitude", "longitude", "kappa_s", "q_frequency_exponent"]
    ]

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
            station_inputs["latitude"], velocity_model_1d.vs30
        )
        dataset = xr.merge(
            [waveform, station_inputs[["latitude", "longitude", "seed", "vref"]]]
        )
        dataset.attrs = {
            "start_sec": 0.0,
            "dt": hf_config.dt,
            "nt": nt,
            "units": "cm/s^2",
        }
        dataset.to_netcdf(out_file, engine="h5netcdf")
    realisations.append_log_entry(realisation_ffp)
