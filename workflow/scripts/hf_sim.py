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
Can be run in the cybershake container. Can also be run from your own computer using the
`hf-sim` command after `pip install workflow@git+https://github.com/ucgmsim/workflow`.

Unlike previous versions this needs no `hb_high_binmod` binary and no writable work
directory: the simulation is the `hf-simulation` package, called in-process.

Usage
-----
`hf-sim [OPTIONS] REALISATION_FFP STOCH_FFP STATION_FILE OUT_FILE`

For More Help
-------------
See the output of `hf-sim --help`.
"""

from pathlib import Path
from typing import Annotated

import dask.array as da
import numpy as np
import pandas as pd
import typer
import xarray as xr
from hf_simulation import (
    COMPONENTS,
    FaultSegment,
    HfConfig,
    SlipModel,
    VelocityModel1D,
    simulate_stations,
    station_seeds,
)

from qcore import cli
from source_modelling.stoch import StochFile
from workflow import log_utils, realisations
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

app = typer.Typer()

TARGET_CHUNK_BYTES = 128 * 2**20
"""Target size of a dask chunk (all components for a batch of stations)."""


def build_config(
    hf_config: HFConfigDefaults,
    rupture_velocity: RuptureVelocity,
    domain_parameters: DomainParameters,
) -> HfConfig:
    """Translate the realisation's configuration into the simulation's.

    Parameters
    ----------
    hf_config : HFConfigDefaults
        The realisation's high-frequency configuration.
    rupture_velocity : RuptureVelocity
        The realisation's rupture velocity settings.
    domain_parameters : DomainParameters
        Supplies the record duration.

    Returns
    -------
    HfConfig
        The simulation configuration.
    """
    return HfConfig(
        duration_s=domain_parameters.duration,
        dt=hf_config.dt,
        stress_drop_bars=hf_config.sdrop,
        fmax_hz=hf_config.fmax,
        kappa_s=hf_config.kappa,
        q_frequency_exponent=hf_config.qfexp,
        rayset=tuple(hf_config.rayset),
        site_amplification=not hf_config.no_siteamp,
        rupture_velocity_fraction=rupture_velocity.rvfrac,
        rupture_velocity_shallow=rupture_velocity.rvfrac_shal,
        rupture_velocity_deep=rupture_velocity.rvfrac_deep,
        rupture_velocity_override=hf_config.rupv,
        rupture_velocity_sigma=hf_config.rv_sig1,
        corner_frequency_constant=hf_config.czero,
        corner_frequency_alpha=hf_config.calpha,
        moment=hf_config.mom,
        fourier_amplitude_sigma_1=hf_config.fa_sig1,
        fourier_amplitude_sigma_2=hf_config.fa_sig2,
        path_duration_model=hf_config.path_dur,
        stress_adjust_model=hf_config.stress_parameter_adjustment_tect_type or 0,
        target_magnitude=hf_config.stress_parameter_adjustment_target_magnitude,
        fault_area_km2=hf_config.stress_parameter_adjustment_fault_area,
    )


def build_slip_model(stoch_ffp: Path) -> SlipModel:
    """Read a stoch file into a simulation slip model.

    Parameters
    ----------
    stoch_ffp : Path
        Path to the stoch file.

    Returns
    -------
    SlipModel
        The slip model, one segment per stoch plane.
    """
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


def build_velocity_model(
    velocity_model: HFVelocityModel1D, vs_moho: float
) -> VelocityModel1D:
    """Convert the realisation's 1D velocity model into the simulation's.

    Parameters
    ----------
    velocity_model : HFVelocityModel1D
        The realisation's layered model.
    vs_moho : float
        Shear velocity at which to truncate the model, km/s.

    Returns
    -------
    VelocityModel1D
        The velocity model, already truncated at the Moho.
    """
    model = velocity_model.model
    return VelocityModel1D(
        thickness_km=model["thickness"].to_numpy(np.float32),
        vp_km_s=model["Vp"].to_numpy(np.float64),
        vsh_km_s=model["Vs"].to_numpy(np.float64),
        density_g_cm3=model["rho"].to_numpy(np.float64),
        quality_factor_p=model["Qp"].to_numpy(np.float32),
        quality_factor_s=model["Qs"].to_numpy(np.float32),
        vs_moho_km_s=vs_moho,
    )


def simulate_chunk(
    station_chunk: xr.Dataset,
    time: np.ndarray,
    slip_model: SlipModel,
    velocity_model: VelocityModel1D,
    config: HfConfig,
) -> xr.DataArray:
    """Simulate one dask block's worth of stations in a single call.

    Parameters
    ----------
    station_chunk : xr.Dataset
        Stations in this block, with `latitude`, `longitude` and `seed`.
    time : np.ndarray
        The shared time axis.
    slip_model : SlipModel
        The fault.
    velocity_model : VelocityModel1D
        The velocity structure.
    config : HfConfig
        The simulation configuration.

    Returns
    -------
    xr.DataArray
        Waveforms over (component, station, time).
    """
    # The block's own station order, not a sorted one: map_blocks requires the output to
    # line up with the template block. Station order does not affect any waveform -- the
    # simulation guarantees that and tests it -- but the LABELS still have to match.
    station_names = station_chunk["station"].values
    waveform = simulate_stations(
        slip_model,
        velocity_model,
        config,
        latitude_deg=station_chunk["latitude"].values.astype(np.float32),
        longitude_deg=station_chunk["longitude"].values.astype(np.float32),
        station_seed=station_chunk["seed"].values.astype(np.uint64),
    )
    return xr.DataArray(
        waveform,
        dims=["component", "station", "time"],
        coords={
            "component": list(COMPONENTS),
            "station": station_names,
            "time": time,
        },
    )


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
    # Name-derived and order-invariant, so adding a station leaves every other station's
    # waveform untouched and re-running a subset reproduces it exactly.
    stations["seed"] = station_seeds(seeds.hf_seed, stations.index)

    config = build_config(hf_config, rupture_velocity, domain_parameters)
    slip_model = build_slip_model(stoch_ffp)
    velocity_model = build_velocity_model(velocity_model_1d, hf_config.vs_moho)

    # float32 throughout: this mirrors how the simulation truncates duration/dt to a
    # sample count, so the dask template matches what comes back.
    nt = int(np.float32(domain_parameters.duration) / np.float32(hf_config.dt))
    time = hf_config.t_sec + np.arange(nt) * hf_config.dt

    # Chunk over stations only, so every chunk holds complete time series.
    chunk_size = max(
        1, TARGET_CHUNK_BYTES // (len(COMPONENTS) * nt * np.float32().itemsize)
    )
    template = xr.DataArray(
        da.empty(
            (len(COMPONENTS), len(stations), nt),
            dtype=np.float32,
            chunks=(len(COMPONENTS), chunk_size, nt),
        ),
        dims=["component", "station", "time"],
        coords={"component": list(COMPONENTS), "station": stations.index, "time": time},
    )

    station_inputs = stations.to_xarray().chunk({"station": chunk_size})
    waveform = station_inputs.map_blocks(
        simulate_chunk,
        template=template,
        kwargs={
            "time": time,
            "slip_model": slip_model,
            "velocity_model": velocity_model,
            "config": config,
        },
    ).rename("waveform")

    station_inputs["vs"] = xr.full_like(
        station_inputs["latitude"], velocity_model_1d.model["Vs"].iloc[0] * 1000
    )
    dataset = xr.merge([waveform, station_inputs])
    dataset.attrs = {
        "start_sec": float(time[0]),
        "dt": hf_config.dt,
        "nt": nt,
        "units": "cm/s^2",
    }
    dataset.to_netcdf(out_file, engine="h5netcdf")
    realisations.append_log_entry(realisation_ffp)
