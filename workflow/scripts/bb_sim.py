"""Broadband Simulation.

Description
-----------
Combine high-frequency and low-frequency simulation waveforms for each station into a broadband simulation file.

Inputs
------
1. A realisation file containing:
   - Realisation metadata,
   - Domain parameters.
2. Station list (latitude, longitude, name),
3. Stations VS30 reference values,
4. Low frequency waveform directory,
5. High frequency output file,

Outputs
-------
An output broadband file in the HDF5 format.

Environment
-----------
Can be run in the cybershake container. Can also be run from your own
computer using the `bb-sim` command which is installed after running
`pip install workflow@git+https://github.com/ucgmsim/workflow`. If
running on your own computer, you need to configure a work directory
(`--work-directory`).

Usage
-----
`bb-sim REALISATION_FFP STATION_FFP STATION_VS30_FFP LOW_FREQUENCY_WAVEFORM_DIRECTORY HIGH_FREQUENCY_WAVEFORM_FILE OUTPUT_FFP`

For More Help
-------------
See the output of `bb-sim --help`.
"""

from pathlib import Path
from typing import Annotated

import numpy as np
import numpy.typing as npt
import pandas as pd
import typer
import xarray as xr

from qcore import cli, siteamp_models, timeseries
from workflow import log_utils, realisations
from workflow.realisations import (
    BroadbandParameters,
    RealisationMetadata,
)

app = typer.Typer()

G = 1 / 981.0


def align_waveforms(
    lf_waveform: npt.NDArray[np.floating],
    hf_waveform: npt.NDArray[np.floating],
    lf_start: float,
    hf_start: float,
    dt: float,
) -> tuple[
    npt.NDArray[np.floating], npt.NDArray[np.floating], npt.NDArray[np.floating]
]:
    """Align LF and HF waveforms to a common time axis.

    Parameters
    ----------
    lf_waveform : array of floats
        The low-frequency waveform to align.
    hf_waveform : array of floats
        The high-frequency waveform to align.
    lf_start : float
        The start of the LF simulation.
    hf_start : float
        The start of the HF simulation.
    dt : float
        The timestep for the simulation.

    Returns
    -------
    array of floats
        The aligned low-frequency results.
    array of floats
        The aligned high-frequency results.
    array of floats
        The new time values.
    """

    lf_nt = lf_waveform.shape[1]
    hf_nt = hf_waveform.shape[1]

    lf_time = lf_start + np.arange(lf_nt) * dt
    hf_time = hf_start + np.arange(hf_nt) * dt

    start = min(lf_time[0], hf_time[0])
    end = max(lf_time[-1], hf_time[-1])
    common_time = np.arange(start, end + dt / 2, dt)

    lf_aligned = np.zeros((lf_waveform.shape[0], common_time.size), dtype=np.float32)
    hf_aligned = np.zeros((hf_waveform.shape[0], common_time.size), dtype=np.float32)

    lf_indices = ((lf_time - start) / dt).round().astype(int)
    hf_indices = ((hf_time - start) / dt).round().astype(int)

    lf_aligned[:, lf_indices] = lf_waveform
    hf_aligned[:, hf_indices] = hf_waveform

    return lf_aligned, hf_aligned, common_time


@cli.from_docstring(app)
@log_utils.log_call()
def combine_hf_and_lf(
    realisation_ffp: Annotated[Path, typer.Argument(dir_okay=False, exists=True)],
    station_vs30_ffp: Annotated[Path, typer.Argument(dir_okay=False, exists=True)],
    low_frequency_waveform_file: Annotated[
        Path, typer.Argument(dir_okay=False, exists=True)
    ],
    high_frequency_waveform_file: Annotated[
        Path, typer.Argument(exists=True, dir_okay=False)
    ],
    output_ffp: Annotated[Path, typer.Argument(dir_okay=False, writable=True)],
) -> None:
    """Combine low-frequency and high-frequency seismic waveforms.

    Parameters
    ----------
    realisation_ffp : Path
        Path to the realisation file containing parameters for the simulation.
    station_vs30_ffp : Path
        Path to the file containing VS30 reference values for stations.
    low_frequency_waveform_file : Path
        File containing low-frequency waveform data.
    high_frequency_waveform_file : Path
        File containing high-frequency waveform data.
    output_ffp : Path
        Path to the output file where the combined broadband waveforms will be saved.
    """
    metadata = RealisationMetadata.read_from_realisation(realisation_ffp)
    broadband_config = BroadbandParameters.read_from_realisation_or_defaults(
        realisation_ffp, metadata.defaults_version
    )
    bb_dt = broadband_config.dt

    # load data stores
    lf = xr.open_dataset(low_frequency_waveform_file)
    hf = xr.open_dataset(high_frequency_waveform_file)
    common_stations = list(
        set(map(str, hf.station.values)) & set(map(str, lf.station.values))
    )
    hf = hf.sel(station=common_stations)
    lf = lf.sel(station=common_stations)

    vs30_df = pd.read_csv(
        station_vs30_ffp, sep=r"\s+", header=None, names=["station", "vsite"]
    ).set_index("station")
    vs30_df["vsite"] = vs30_df["vsite"].astype(np.float32)
    vs30_df = vs30_df.loc[common_stations]
    vs30_df["vref"] = 500.0
    vs30_df["vpga"] = 500.0

    bb_waveforms = []
    new_time_coords = None

    for i, (lf_component, hf_component) in enumerate(zip(lf.component, hf.component)):
        hf_waveform_raw = hf.sel(component=hf_component).waveform.values
        lf_waveform_raw = lf.sel(component=lf_component).waveform.values

        temp_lf_padded, temp_hf_padded, new_time_coords = align_waveforms(
            lf_waveform_raw,
            hf_waveform_raw,
            lf.attrs["start_sec"],
            hf.attrs["start_sec"],
            bb_dt,
        )
        bb_nt = temp_lf_padded.shape[1]

        vs30_df["pga"] = np.abs(temp_hf_padded).max(axis=1) * G

        hf_amp_val = siteamp_models.cb_amp_multi(vs30_df)
        hf_amp_fas_vals = siteamp_models.cb2014_to_fas_amplification_factors(
            hf_amp_val, bb_dt, bb_nt
        )
        hf_waveform_amped = timeseries.ampdeamp(
            temp_hf_padded, hf_amp_fas_vals, amplify=True
        )
        hf_filtered = timeseries.bwfilter(
            hf_waveform_amped, bb_dt, 1.0, timeseries.Band.HIGHPASS
        )
        lf_filtered = timeseries.bwfilter(
            temp_lf_padded, bb_dt, 1.0, timeseries.Band.LOWPASS
        )
        bb_waveforms.append((hf_filtered + lf_filtered) * G)

    bb_waveform = np.stack(bb_waveforms, dtype=np.float32)

    xr.Dataset(
        {"waveform": (["component", "station", "time"], bb_waveform)},
        coords={
            "component": ("component", ["x", "y", "z"]),
            "station": ("station", common_stations),
            "time": ("time", new_time_coords),
            "x": ("station", lf.x.values),
            "y": ("station", lf.y.values),
            "latitude": ("station", lf.lat.values),
            "longitude": ("station", lf.lon.values),
        },
        attrs={
            "units": "g",
        },
    ).to_netcdf(
        output_ffp,
        engine="h5netcdf",
        encoding={
            "waveform": {
                "fletcher32": True,  # Add Fletcher-32 checksums for long-term storage.
            }
        },
    )
    realisations.append_log_entry(realisation_ffp)
